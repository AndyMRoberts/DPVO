import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from functools import partial

from . import altcorr, fastba, lietorch
from . import projective_ops as pops
from .lietorch import SE3
from .net import VONet
from .patchgraph import PatchGraph
from .utils import *

mp.set_start_method('spawn', True)


autocast = partial(torch.amp.autocast, "cuda")


class DPVO_images_only:

    def __init__(self, cfg, network, ht=480, wd=640, viz=False, onnx_dir=None):
        self.cfg = cfg
        self._onnx_fnet = None
        self._onnx_inet = None
        self.load_weights(network, onnx_dir=onnx_dir)
        self.is_initialized = False
        self.enable_timing = False
        torch.set_num_threads(2)

        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.ht = ht    # image height
        self.wd = wd    # image width

        DIM = self.DIM
        RES = self.RES

        ### state attributes ###
        self.tlist = []
        self.counter = 0

        ht = ht // RES
        wd = wd // RES

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        ### network attributes ###
    
    def load_weights(self, network, onnx_dir=None):
        # load network from checkpoint file
        if isinstance(network, str):
            from collections import OrderedDict
            state_dict = torch.load(network, weights_only=True)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace('module.', '')] = v

            self.network = VONet()
            self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes
        self.DIM = self.network.DIM
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

         # optional ONNX encoders (fnet, inet) for hybrid PyTorch+ONNX
        if onnx_dir:
            self._load_onnx_encoders(onnx_dir)

    def _load_onnx_encoders(self, onnx_dir):
        import os
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnx_dir requires onnxruntime. Install with: pip install onnxruntime-gpu")
        fnet_path = os.path.join(onnx_dir, "fnet.onnx")
        inet_path = os.path.join(onnx_dir, "inet.onnx")
        if not os.path.isfile(fnet_path) or not os.path.isfile(inet_path):
            raise FileNotFoundError(f"ONNX encoder files not found in {onnx_dir}. Run andy/onnx_conversion.ipynb first.")
        # Quantized (int8) ONNX models use ConvInteger, which is only implemented on CPU.
        def _model_uses_conv_integer(path):
            try:
                import onnx
                m = onnx.load(path)
                for node in m.graph.node:
                    if node.op_type == "ConvInteger":
                        return True
                return False
            except Exception:
                return False
        onnx_dir_str = os.path.normpath(str(onnx_dir))
        is_quantized = (
            _model_uses_conv_integer(fnet_path)
            or _model_uses_conv_integer(inet_path)
            or "int8" in onnx_dir_str
            or "quant" in onnx_dir_str.lower()
        )
        # Quantized models use ConvInteger: CUDA EP doesn't implement it; CPU EP in onnxruntime-gpu
        # may not either. TensorRT EP can run INT8 on GPU. Prefer TensorRT > CUDA > CPU for quantized.
        if is_quantized:
            available = ort.get_available_providers()
            if "TensorrtExecutionProvider" in available:
                print("Tensorrt available")
                providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                # TensorRT not installed; try CPU only (requires full CPU build for ConvInteger)
                providers = ["CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._onnx_fnet = ort.InferenceSession(fnet_path, providers=providers)
        self._onnx_inet = ort.InferenceSession(inet_path, providers=providers)


    def terminate(self):
        tstamps = np.array(self.tlist, dtype=np.float64)
        return tstamps
  

    def __call__(self, tstamp, image, intrinsics):
        """ track new frame """

        image = 2 * (image[None,None] / 255.0) - 0.5

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            if self._onnx_fnet is not None and self._onnx_inet is not None:
                # Hybrid: run fnet/inet with ONNX, rest with PyTorch
                feed = {"images": image.cpu().numpy().astype(np.float32)}
                fmap = torch.from_numpy(self._onnx_fnet.run(None, feed)[0]).cuda().to(image.dtype)
                imap = torch.from_numpy(self._onnx_inet.run(None, feed)[0]).cuda().to(image.dtype)
            else:
                fmap, gmap, imap, patches, _, clr = \
                    self.network.patchify(image,
                        patches_per_image=self.cfg.PATCHES_PER_FRAME,
                        centroid_sel_strat=self.cfg.CENTROID_SEL_STRAT,
                        return_color=True)
        self.tlist.append(tstamp)
        

import datetime
import glob
import json
import os
import os.path as osp
from pathlib import Path

import cv2
import evo.main_ape as main_ape
import numpy as np
import torch
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from dpvo.config import cfg
from dpvo.data_readers.tartan import test_split as val_split
from dpvo.plot_utils import plot_trajectory
from dpvo.utils import Timer

# test_split = \
#     ["MH%03d"%i for i in range(8)] + \
#     ["ME%03d"%i for i in range(8)]

test_split = ['ME005']

STRIDE = 1
fx, fy, cx, cy = [320, 320, 320, 240]


def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

def video_iterator(imagedir, ext=".png", preload=True):
    imfiles = glob.glob(osp.join(imagedir, "*{}".format(ext)))

    data_list = []
    for imfile in sorted(imfiles)[::STRIDE]:
        image = torch.from_numpy(cv2.imread(imfile)).permute(2,0,1)
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        data_list.append((image, intrinsics))

    for (image, intrinsics) in data_list:
        yield image.cuda(), intrinsics.cuda()

@torch.no_grad()
def run(imagedir, cfg, network, viz=False, show_img=False, onnx_dir=None):
    slam = DPVO_images_only(cfg, network, ht=480, wd=640, viz=viz, onnx_dir=onnx_dir)
    n_frames = len(glob.glob(osp.join(imagedir, "*.png"))) // STRIDE

    for t, (image, intrinsics) in enumerate(video_iterator(imagedir)):
        if show_img:
            show_image(image, 1)
        if t == 0 or (t + 1) % 50 == 0 or t == n_frames - 1:
            print(f"  Frame {t + 1}/{n_frames}")

        with Timer("SLAM", enabled=False):
            slam(t, image, intrinsics)

    return slam.terminate()


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--show_img', action="store_true")
    parser.add_argument('--id', type=int, default=-1)
    parser.add_argument('--weights', default="dpvo.pth")
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--split', default="validation")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--backend_thresh', type=float, default=18.0)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--run_dir', type=str, default=None,
                        help="Directory to write ate_results.json and run artifacts")
    parser.add_argument('--datapath', type=str, default=None,
                        help="Data root (overrides default TartanAir path)")
    parser.add_argument('--gt_path', type=str, default=None,
                        help="Ground truth root or file (overrides default)")
    parser.add_argument('--backend', choices=['pytorch', 'onnx'], default='pytorch',
                        help='Run with pure PyTorch or PyTorch+ONNX (encoders via ONNX)'),
    parser.add_argument('--onnx_dir', type=str, default='andy/onnx',
                        help='Directory containing fnet.onnx and inet.onnx (used when --backend onnx)')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.BACKEND_THRESH = args.backend_thresh
    cfg.merge_from_list(args.opts)

    onnx_dir = os.path.abspath(args.onnx_dir) if args.backend == 'onnx' else None
    if onnx_dir and not os.path.isdir(onnx_dir):
        raise FileNotFoundError(f"ONNX dir not found: {onnx_dir}. Run andy/onnx_conversion.ipynb and set --onnx_dir.")

    print("Running with config...")
    print(cfg)
    print("Backend:", args.backend, f"(onnx_dir={onnx_dir})" if onnx_dir else "")

    torch.manual_seed(1234)

    data_root = args.datapath or "/mnt/data/datasets/agricultural/tartanair"
    gt_root = args.gt_path or osp.join(data_root, "mono_gt")
    scene_path = os.path.join(data_root, "tartanair_mono_track", test_split[args.id])
    tstamps = run(scene_path, cfg, args.weights, viz=args.viz, show_img=args.show_img, onnx_dir=onnx_dir)

    if args.run_dir:
        os.makedirs(args.run_dir, exist_ok=True)
        ate_results = {
            "total_frames": len(tstamps),
            "results": {"ATE": "NA"},
            "all_results": "NA",
            "per_scene": "NA",
        }
        with open(osp.join(args.run_dir, "ate_results.json"), "w") as f:
            json.dump(ate_results, f, indent=2)

  
