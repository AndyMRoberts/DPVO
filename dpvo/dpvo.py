import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from functools import partial

from . import altcorr, fastba, lietorch
from . import projective_ops as pops
from .lietorch import SE3
from .net import VONet
import dpvo.net as net
from .patchgraph import PatchGraph
from .utils import *

import os
try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("onnx_dir requires onnxruntime. Install with: pip install onnxruntime-gpu")

import pandas as pd # temporary, for logging

so = ort.SessionOptions()
so.log_severity_level = 2  # 0 = verbose


verbose = False
logging = True


mp.set_start_method('spawn', True)


autocast = partial(torch.amp.autocast, "cuda")
Id = SE3.Identity(1, device="cuda")


class DPVO:

    def __init__(self, cfg, network, ht=480, wd=640, viz=False, onnx_dir=None, onnx_type='patchify',
                 record_update_dummy_inputs=True,
                 record_update_dummy_inputs_once=False,
                 record_update_dummy_inputs_path='/home/campus.ncl.ac.uk/c4071391/Projects/DPVO/andy/onnx/input_payload.pth'):
        # onnx additions
        self._onnx_fnet = None
        self._onnx_inet = None
        self._onnx_patchify = None
        self._onnx_update = None
        self.use_edges_padding = False
        self.max_edges_count = []
        self.edges_padded_value = 50000 # used to avoid dynamic axes issue with onnx export
        self.logging = False
        self.log_start_buffer = 100
        self.log_count = 0

        # Snapshot net/ctx/corr/ii/jj/kk for ONNX export dummy inputs (PyTorch update path only).
        self.record_update_dummy_inputs = record_update_dummy_inputs
        self.record_update_dummy_inputs_once = record_update_dummy_inputs_once
        self.record_update_dummy_inputs_path = record_update_dummy_inputs_path
        self._update_dummy_inputs = None

        self.cfg = cfg
        self.load_weights(network, onnx_dir=onnx_dir, onnx_type=onnx_type)
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

        # keep track of global-BA calls
        self.ran_global_ba = np.zeros(100000, dtype=bool)

        ht = ht // RES
        wd = wd // RES

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        ### network attributes ###
        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}

        ### frame memory size ###
        self.pmem = self.mem = 36 # 32 was too small given default settings
        if self.cfg.LOOP_CLOSURE:
            self.last_global_ba = -1000 # keep track of time since last global opt
            self.pmem = self.cfg.MAX_EDGE_AGE # patch memory

        self.imap_ = torch.zeros(self.pmem, self.M, DIM, **kwargs)
        self.gmap_ = torch.zeros(self.pmem, self.M, 128, self.P, self.P, **kwargs)

        self.pg = PatchGraph(self.cfg, self.P, self.DIM, self.pmem, **kwargs)

        # classic backend
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.load_long_term_loop_closure()

        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.viewer = None
        if viz:
            self.start_viewer()

    def load_long_term_loop_closure(self):
        try:
            from .loop_closure.long_term import LongTermLoopClosure
            self.long_term_lc = LongTermLoopClosure(self.cfg, self.pg)
        except ModuleNotFoundError as e:
            self.cfg.CLASSIC_LOOP_CLOSURE = False
            print(f"WARNING: {e}")

    def load_weights(self, network, onnx_dir=None, onnx_type='patchify'):
        # load network from checkpoint file
        if onnx_type == 'all' or onnx_type == 'all_modular':
            # only these values and corrblock are needed when using full onnx implementation
            self.DIM = net.DIM
            self.RES = net.RES
            self.P = net.P
        else:
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
            if onnx_type == 'features':
                self._load_onnx_encoders_features(onnx_dir)
            elif onnx_type == 'patchify':
                self._load_onnx_encoders_patchify(onnx_dir)
            elif onnx_type == 'all':
                self.use_edges_padding = True
                self._load_onnx_encoders_patchify(onnx_dir)
                self._load_onnx_encoders_update(onnx_dir)
            elif onnx_type == 'all_modular':
                self.use_edges_padding = False
                self._load_onnx_encoders_patchify(onnx_dir)
                self._load_onnx_encoders_update_modular(onnx_dir)
    
    def _model_uses_conv_integer(self, path):
            try:
                import onnx
                m = onnx.load(path)
                for node in m.graph.node:
                    if node.op_type == "ConvInteger":
                        return True
                return False
            except Exception:
                return False

    def _load_onnx_encoders_update_modular(self, onnx_dir):
        if verbose: print(f'Loading onnx update modular model')
        
        module_paths = {'corr': os.path.join(onnx_dir, "corr.onnx"),
                        'norm': os.path.join(onnx_dir, "norm.onnx"),
                        'c1': os.path.join(onnx_dir, "c1.onnx"),
                        'c2': os.path.join(onnx_dir, "c2.onnx"),
                        'agg_kk': os.path.join(onnx_dir, "agg_kk.onnx"),
                        'agg_ij': os.path.join(onnx_dir, "agg_ij.onnx"),
                        'gru': os.path.join(onnx_dir, "gru.onnx"),
                        'w': os.path.join(onnx_dir, "w.onnx"),
                        'd': os.path.join(onnx_dir, "d.onnx")}
        module_sessions = {}
        for name, module_path in module_paths.items():
            if not os.path.isfile(module_path):
                raise FileNotFoundError(f"ONNX encoder file not found in {onnx_dir} for {name}. Run andy/onnx_conversion.ipynb first.")
            # Quantized (int8) ONNX models use ConvInteger, which is only implemented on CPU.
               
            onnx_dir_str = os.path.normpath(str(onnx_dir))
            is_quantized = (
                self._model_uses_conv_integer(module_path)
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
                # providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                providers = ["CUDAExecutionProvider"]
            module_sessions[name] = ort.InferenceSession(module_path, sess_options=so, providers=providers)
            if verbose: print(f'Onnx {name} module loaded: {module_sessions[name]}')
        self._onnx_update_modular = module_sessions

    def _load_onnx_encoders_update(self, onnx_dir):
        if verbose: print(f'Loading onnx update model')
        update_path = os.path.join(onnx_dir, "update.onnx")

        if not os.path.isfile(update_path):
            raise FileNotFoundError(f"ONNX encoder file not found in {onnx_dir}. Run andy/onnx_conversion.ipynb first.")
        # Quantized (int8) ONNX models use ConvInteger, which is only implemented on CPU.
        
        onnx_dir_str = os.path.normpath(str(onnx_dir))
        is_quantized = (
            self._model_uses_conv_integer(update_path)
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
            # providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            providers = ["CUDAExecutionProvider"]
        self._onnx_update = ort.InferenceSession(update_path, sess_options=so, providers=providers)
        if verbose: print(f'Onnx Update Loaded: {self._onnx_update}')
    

    def _load_onnx_encoders_patchify(self, onnx_dir):
        if verbose: print(f'Loading onnx patchify model')
        patchify_path = os.path.join(onnx_dir, "patchify.onnx")
        if not os.path.isfile(patchify_path):
            raise FileNotFoundError(f"ONNX encoder file not found in {onnx_dir}. Run andy/onnx_conversion.ipynb first.")
        # Quantized (int8) ONNX models use ConvInteger, which is only implemented on CPU.

        onnx_dir_str = os.path.normpath(str(onnx_dir))
        is_quantized = (
            self._model_uses_conv_integer(patchify_path)
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
            # providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            providers = ["CUDAExecutionProvider"]
        self._onnx_patchify = ort.InferenceSession(patchify_path, sess_options=so, providers=providers)
        if verbose: print(f'Onnx Patchify Loaded: {self._onnx_patchify}')
    
    def _load_onnx_encoders_features(self, onnx_dir):
        if verbose: print(f'Loading onnx features only models')
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
        self._onnx_fnet = ort.InferenceSession(fnet_path, sess_options=so, providers=providers)
        self._onnx_inet = ort.InferenceSession(inet_path, sess_options=so, providers=providers)

    def start_viewer(self):
        from dpviewer import Viewer

        intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")

        self.viewer = Viewer(
            self.image_,
            self.pg.poses_,
            self.pg.points_,
            self.pg.colors_,
            intrinsics_)

    @property
    def poses(self):
        return self.pg.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.pg.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.pg.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.pg.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.pmem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.pmem * self.M, 128, 3, 3)

    @property
    def n(self):
        return self.pg.n

    @n.setter
    def n(self, val):
        self.pg.n = val

    @property
    def m(self):
        return self.pg.m

    @m.setter
    def m(self, val):
        self.pg.m = val

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.pg.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self):

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.terminate(self.n)

        if self.cfg.LOOP_CLOSURE:
            self.append_factors(*self.pg.edges_loop())

        for _ in range(12):
            self.ran_global_ba[self.n] = False
            self.update()

        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.pg.tstamps_[i]] = self.pg.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=np.float64)
        if self.viewer is not None:
            self.viewer.join()

        # Poses: x y z qx qy qz qw
        return poses, tstamps

    def corr(self, coords, indicies=None):
        """ local correlation volume """
        ii, jj = indicies if indicies is not None else (self.pg.kk, self.pg.jj)
        ii1 = ii % (self.M * self.pmem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.pg.ii, self.pg.jj, self.pg.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj):
        self.pg.jj = torch.cat([self.pg.jj, jj])
        self.pg.kk = torch.cat([self.pg.kk, ii])
        self.pg.ii = torch.cat([self.pg.ii, self.ix[ii]])

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        self.pg.net = torch.cat([self.pg.net, net], dim=1)

    def remove_factors(self, m, store: bool):
        assert self.pg.ii.numel() == self.pg.weight.shape[1]
        if store:
            self.pg.ii_inac = torch.cat((self.pg.ii_inac, self.pg.ii[m]))
            self.pg.jj_inac = torch.cat((self.pg.jj_inac, self.pg.jj[m]))
            self.pg.kk_inac = torch.cat((self.pg.kk_inac, self.pg.kk[m]))
            self.pg.weight_inac = torch.cat((self.pg.weight_inac, self.pg.weight[:,m]), dim=1)
            self.pg.target_inac = torch.cat((self.pg.target_inac, self.pg.target[:,m]), dim=1)
        self.pg.weight = self.pg.weight[:,~m]
        self.pg.target = self.pg.target[:,~m]

        self.pg.ii = self.pg.ii[~m]
        self.pg.jj = self.pg.jj[~m]
        self.pg.kk = self.pg.kk[~m]
        self.pg.net = self.pg.net[:,~m]
        assert self.pg.ii.numel() == self.pg.weight.shape[1]

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.imap[:,kk % (self.M * self.pmem)]
            net, delta, weight = self.update_inner(net, corr, ctx, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        k = (self.pg.ii == i) & (self.pg.jj == j)
        ii = self.pg.ii[k]
        jj = self.pg.jj[k]
        kk = self.pg.kk[k]

        flow, _ = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self):

        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.KEYFRAME_INDEX + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)
 
        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.pg.tstamps_[k-1]
            t1 = self.pg.tstamps_[k]

            dP = SE3(self.pg.poses_[k]) * SE3(self.pg.poses_[k-1]).inv()
            self.pg.delta[t1] = (t0, dP)

            to_remove = (self.pg.ii == k) | (self.pg.jj == k)
            self.remove_factors(to_remove, store=False)

            self.pg.kk[self.pg.ii > k] -= self.M
            self.pg.ii[self.pg.ii > k] -= 1
            self.pg.jj[self.pg.jj > k] -= 1

            for i in range(k, self.n-1):
                self.pg.tstamps_[i] = self.pg.tstamps_[i+1]
                self.pg.colors_[i] = self.pg.colors_[i+1]
                self.pg.poses_[i] = self.pg.poses_[i+1]
                self.pg.patches_[i] = self.pg.patches_[i+1]
                self.pg.intrinsics_[i] = self.pg.intrinsics_[i+1]

                self.imap_[i % self.pmem] = self.imap_[(i+1) % self.pmem]
                self.gmap_[i % self.pmem] = self.gmap_[(i+1) % self.pmem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]

            self.n -= 1
            self.m-= self.M

            if self.cfg.CLASSIC_LOOP_CLOSURE:
                self.long_term_lc.keyframe(k)

        to_remove = self.ix[self.pg.kk] < self.n - self.cfg.REMOVAL_WINDOW # Remove edges falling outside the optimization window
        if self.cfg.LOOP_CLOSURE:
            # ...unless they are being used for loop closure
            lc_edges = ((self.pg.jj - self.pg.ii) > 30) & (self.pg.jj > (self.n - self.cfg.OPTIMIZATION_WINDOW))
            to_remove = to_remove & ~lc_edges
        self.remove_factors(to_remove, store=True)

    def __run_global_BA(self):
        """ Global bundle adjustment
         Includes both active and inactive edges """
        full_target = torch.cat((self.pg.target_inac, self.pg.target), dim=1)
        full_weight = torch.cat((self.pg.weight_inac, self.pg.weight), dim=1)
        full_ii = torch.cat((self.pg.ii_inac, self.pg.ii))
        full_jj = torch.cat((self.pg.jj_inac, self.pg.jj))
        full_kk = torch.cat((self.pg.kk_inac, self.pg.kk))

        self.pg.normalize()
        lmbda = torch.as_tensor([1e-4], device="cuda")
        t0 = self.pg.ii.min().item()
        fastba.BA(self.poses, self.patches, self.intrinsics,
            full_target, full_weight, lmbda, full_ii, full_jj, full_kk, t0, self.n, M=self.M, iterations=2, eff_impl=True)
        self.ran_global_ba[self.n] = True


    def update_inner(self, net, corr, ctx, ii, jj, kk):
        def bind_torch_inputs(io_binding, inputs: dict, device="cuda", device_id=0):
            for name, tensor in inputs.items():
                if tensor is None:
                    continue

                assert tensor.is_cuda, f"{name} must be on GPU for zero-copy"

                io_binding.bind_input(
                    name=name,
                    device_type=device,
                    device_id=device_id,
                    element_type=(
                        np.float32 if tensor.dtype in (torch.float32, torch.float16)
                        else np.int64
                    ),
                    shape=tuple(tensor.shape),
                    buffer_ptr=tensor.data_ptr(),
                )

        # Determine real edge count and (optionally) pad to fixed ONNX size
        B, E_real, D = net.shape
        _, _, Cc = corr.shape

        if self.logging:
            self.log_count += 1
            if self.log_count == self.log_start_buffer:
                pd.DataFrame(np.array(net.squeeze(0).cpu())).to_csv('net.csv')
                pd.DataFrame(np.array(ctx.squeeze(0).cpu())).to_csv('ctx.csv')
                pd.DataFrame(np.array(corr.squeeze(0).cpu())).to_csv('corr.csv')
                pd.DataFrame(np.array(ii.cpu())).to_csv('ii.csv')
                pd.DataFrame(np.array(jj.cpu())).to_csv('jj.csv')
                pd.DataFrame(np.array(kk.cpu())).to_csv('kk.csv')


        if self.use_edges_padding:
            if verbose: print(f'E_real = {E_real}')
            pad_value = self.edges_padded_value - E_real

            if pad_value < 0:
                raise ValueError(
                    f"Number of edges exceeds the padding size for ONNX. "
                    f"Re-export update.onnx with edges_padded_value increased by {-pad_value}."
                )

            # create dummy pad variables
            net_pad = torch.zeros(B, pad_value, D, device=net.device, dtype=net.dtype)
            ctx_pad = torch.zeros(B, pad_value, D, device=ctx.device, dtype=ctx.dtype)
            corr_pad = torch.zeros(B, pad_value, Cc, device=corr.device, dtype=corr.dtype)
            #pad 0 method - repeats # 6.68
            # ii_pad = ii[-1].repeat(pad_value)  # repeats last frame
            # jj_pad = jj[-1].repeat(pad_value)  # repeats last frame
            # # generates brand new, unconnected edges that should not affect the algorithm
            # kk_pad = kk.max() + torch.arange(1, pad_value + 1, device=kk.device, dtype=kk.dtype)

            # pad 2 method - zeros # 8.74
            ii_pad = torch.zeros(pad_value, device=net.device, dtype=ii.dtype)  
            jj_pad = torch.zeros(pad_value, device=net.device, dtype=jj.dtype)  
            kk_pad = torch.zeros(pad_value, device=net.device, dtype=kk.dtype)

            # concatenate pad variables onto real values to create a valid padded parameter
            # # original concatenation
            net = torch.cat([net, net_pad], dim=1)
            ctx = torch.cat([ctx, ctx_pad], dim=1)
            corr = torch.cat([corr, corr_pad], dim=1)
            ii = torch.cat([ii, ii_pad], dim=0)
            jj = torch.cat([jj, jj_pad], dim=0)
            kk = torch.cat([kk, kk_pad], dim=0)

            # reverse order of padding (latest values more important?) # 8.94
            # net_padded = torch.cat([net_pad, net], dim=1)
            # ctx_padded = torch.cat([ctx_pad, ctx], dim=1)
            # corr_padded = torch.cat([corr_pad,corr], dim=1)
            # ii_padded = torch.cat([ii_pad, ii], dim=0)
            # jj_padded = torch.cat([jj_pad, jj], dim=0)
            # kk_padded = torch.cat([kk_pad, kk], dim=0)

        else:
            # No padding – run with current edge count
            net_padded = net
            ctx_padded = ctx
            corr_padded = corr
            ii_padded = ii
            jj_padded = jj
            kk_padded = kk

        if self._onnx_update is not None:
            if verbose: print(f'Running onnx update')

            # ONNX model is exported in fp32; ensure inputs are fp32 / int64.
            net_input = net.to(torch.float32, copy=False)
            ctx_input = ctx.to(torch.float32, copy=False)
            corr_input = corr.to(torch.float32, copy=False)

            # Prepare outputs as CUDA tensors and bind with IO binding for zero-copy.
            Bp, Ep, Dp = net_input.shape
            net_out = torch.empty((Bp, Ep, Dp), device=net_input.device, dtype=torch.float32)
            delta_out = torch.empty((Bp, Ep, 2), device=net_input.device, dtype=torch.float32)
            weight_out = torch.empty((Bp, Ep, 2), device=net_input.device, dtype=torch.float32)

            io_binding = self._onnx_update.io_binding()

            feed_tensors = {
                'net_in': net_input,
                'inp': ctx_input,
                'corr': corr_input,
                'flow': None,   # optional input in the graph
                'ii': ii,
                'jj': jj,
                'kk': kk,
            }
            bind_torch_inputs(io_binding, feed_tensors)

            io_binding.bind_output(
                name='net_out',
                device_type="cuda",
                device_id=0,
                element_type=np.float32,
                shape=tuple(net_out.shape),
                buffer_ptr=net_out.data_ptr(),
            )
            io_binding.bind_output(
                name='delta_out',
                device_type="cuda",
                device_id=0,
                element_type=np.float32,
                shape=tuple(delta_out.shape),
                buffer_ptr=delta_out.data_ptr(),
            )
            io_binding.bind_output(
                name='weight_out',
                device_type="cuda",
                device_id=0,
                element_type=np.float32,
                shape=tuple(weight_out.shape),
                buffer_ptr=weight_out.data_ptr(),
            )

            # Execute ONNX model with bound CUDA tensors
            self._onnx_update_.run_with_iobinding(io_binding)

            # Unpad back to the real edge count for downstream PyTorch code
            net = net_out[:, :E_real, :].to(net.dtype)
            # Original PyTorch update returns delta/weight with shape (B, E, 2)
            delta = delta_out[:, :E_real, :]
            weight = weight_out[:, :E_real, :]

        elif self._onnx_update_modular is not None:
            if verbose: print(f'Running onnx update modular')

            # ONNX model is exported in fp32; ensure inputs are fp32 / int64.
            net_input = net.to(torch.float32, copy=False)
            ctx_input = ctx.to(torch.float32, copy=False)
            corr_input = corr.to(torch.float32, copy=False)
            ii_input = ii.to(torch.int64, copy=False)

            # Prepare outputs as CUDA tensors and bind with IO binding for zero-copy.
            Bp, Ep, Dp = net_input.shape
            corr_out = torch.empty((Bp, Ep, Dp), device=net_input.device, dtype=torch.float32)
            norm_out = torch.empty((Bp, Ep, Dp), device=net_input.device, dtype=torch.float32)
            c1_out = torch.empty((Bp, Ep, Dp), device=net_input.device, dtype=torch.float32)
            c2_out = torch.empty((Bp, Ep, Dp), device=net_input.device, dtype=torch.float32)
            agg_kk_out = torch.empty((Bp, Ep, Dp), device=net_input.device, dtype=torch.float32)
            agg_ij_out = torch.empty((Bp, Ep, Dp), device=net_input.device, dtype=torch.float32)
            net_out = torch.empty((Bp, Ep, Dp), device=net_input.device, dtype=torch.float32)
            weight_out = torch.empty((Bp, Ep, 2), device=net_input.device, dtype=torch.float32)
            delta_out = torch.empty((Bp, Ep, 2), device=net_input.device, dtype=torch.float32)

            corr_io_binding = self._onnx_update_modular['corr'].io_binding()  
            norm_io_binding = self._onnx_update_modular['norm'].io_binding()
            c1_io_binding = self._onnx_update_modular['c1'].io_binding()
            c2_io_binding = self._onnx_update_modular['c2'].io_binding()
            agg_kk_io_binding = self._onnx_update_modular['agg_kk'].io_binding()
            agg_ij_io_binding = self._onnx_update_modular['agg_ij'].io_binding()
            gru_io_binding = self._onnx_update_modular['gru'].io_binding()
            weight_io_binding = self._onnx_update_modular['w'].io_binding()
            delta_io_binding = self._onnx_update_modular['d'].io_binding()

            # --------------corr--------------
            bind_torch_inputs(corr_io_binding, {'corr_input': corr_input})
            corr_io_binding.bind_output(
                name='corr_out',
                device_type="cuda",
                device_id=0,
                element_type=np.float32,
                shape=tuple(corr_out.shape),
                buffer_ptr=corr_out.data_ptr(),
            )
            self._onnx_update_modular['corr'].run_with_iobinding(corr_io_binding)
            # -------------------------------
            net_input = net + ctx_input + corr_out
            # --------------norm-------------
            bind_torch_inputs(norm_io_binding, {'net_input': net_input})
            norm_io_binding.bind_output(
                name='norm_out',
                device_type="cuda",
                device_id=0,
                element_type=np.float32,
                shape=tuple(norm_out.shape),
                buffer_ptr=norm_out.data_ptr(),
            )
            self._onnx_update_modular['norm'].run_with_iobinding(norm_io_binding)
            # -------------------------------------
            ix, jx = fastba.neighbors(kk, jj)
            mask_ix = (ix >= 0).float().reshape(1, -1, 1)
            mask_jx = (jx >= 0).float().reshape(1, -1, 1)
            # -----------------c1--------------------
            c1_input = mask_ix * norm_out[:,ix]
            bind_torch_inputs(c1_io_binding, {'c1_input': c1_input})
            c1_io_binding.bind_output(
                name='c1_out',
                device_type="cuda",
                device_id=0,
                element_type=np.float32,
                shape=tuple(c1_out.shape),
                buffer_ptr=c1_out.data_ptr(),
            )
            self._onnx_update_modular['c1'].run_with_iobinding(c1_io_binding)
            # -----------------c2--------------------
            net = net + c1_out
            c2_input = mask_jx * net[:,jx]
            bind_torch_inputs(c2_io_binding, {'c2_input': c2_input})
            c2_io_binding.bind_output(
                name='c2_out',
                device_type="cuda",
                device_id=0,
                element_type=np.float32,
                shape=tuple(c2_out.shape),
                buffer_ptr=c2_out.data_ptr(),
            )
            self._onnx_update_modular['c2'].run_with_iobinding(c2_io_binding)
            net = net + c2_out
            # -----------------agg_kk--------------------
            bind_torch_inputs(agg_kk_io_binding, {'agg_kk_input': net, 'kk_input': kk})
            agg_kk_io_binding.bind_output(
                name='agg_kk_out',
                device_type="cuda",
                device_id=0,
                element_type=np.float32,
                shape=tuple(agg_kk_out.shape),
                buffer_ptr=agg_kk_out.data_ptr(),
            )
            self._onnx_update_modular['agg_kk'].run_with_iobinding(agg_kk_io_binding)
            net = net + agg_kk_out
            # -----------------agg_ij--------------------
            iijj_input = ii*12345 + jj
            bind_torch_inputs(agg_ij_io_binding, {'agg_ij_input': net, 'iijj_input': iijj_input})
            agg_ij_io_binding.bind_output(
                name='agg_ij_out',
                device_type="cuda",
                device_id=0,
                element_type=np.float32,
                shape=tuple(agg_ij_out.shape),
                buffer_ptr=agg_ij_out.data_ptr(),
            )
            self._onnx_update_modular['agg_ij'].run_with_iobinding(agg_ij_io_binding)
            net = net + agg_ij_out
            # -----------------gru--------------------
            bind_torch_inputs(gru_io_binding, {'gru_input': net})
            gru_io_binding.bind_output(
                name='net_out',
                device_type="cuda",
                device_id=0,
                element_type=np.float32,
                shape=tuple(net_out.shape),
                buffer_ptr=net_out.data_ptr(),
            )
            self._onnx_update_modular['gru'].run_with_iobinding(gru_io_binding)
            # -----------------weight--------------------
            bind_torch_inputs(weight_io_binding, {'net_input': net_out})
            weight_io_binding.bind_output(
                name='weight_out',
                device_type="cuda",
                device_id=0,
                element_type=np.float32,
                shape=tuple(weight_out.shape),
                buffer_ptr=weight_out.data_ptr(),
            )
            self._onnx_update_modular['w'].run_with_iobinding(weight_io_binding)
            # -----------------delta--------------------
            bind_torch_inputs(delta_io_binding, {'net_input': net_out})
            delta_io_binding.bind_output(
                name='delta_out',
                device_type="cuda",
                device_id=0,
                element_type=np.float32,
                shape=tuple(delta_out.shape),
                buffer_ptr=delta_out.data_ptr(),
            )
            self._onnx_update_modular['d'].run_with_iobinding(delta_io_binding)
            # -------------------------------------
            net = net_out
            delta = delta_out
            weight = weight_out
                    
        else: 
            if verbose: print(f'Running pytorch update')
            self._maybe_record_update_dummy_inputs(net, ctx, corr, ii, jj, kk)
            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk)
        
        return net, delta, weight

    def _maybe_record_update_dummy_inputs(self, net, ctx, corr, ii, jj, kk):
        """When record_update_dummy_inputs is True, save tensors for torch.onnx.export."""
        if not self.record_update_dummy_inputs:
            return
        if self.record_update_dummy_inputs_once and self._update_dummy_inputs is not None:
            return
        payload = {
            'net_in': net.detach().cpu().float().clone(),
            'inp': ctx.detach().cpu().float().clone(),
            'corr': corr.detach().cpu().float().clone(),
            'flow': None,
            'ii': ii.detach().cpu().clone(),
            'jj': jj.detach().cpu().clone(),
            'kk': kk.detach().cpu().clone(),
        }
        self._update_dummy_inputs = payload
        if self.record_update_dummy_inputs_path:
            torch.save(payload, self.record_update_dummy_inputs_path)
            if verbose:
                print(f'[DPVO] Saved update dummy inputs to {self.record_update_dummy_inputs_path}')

    def update(self):
        with Timer("other", enabled=self.enable_timing):
            coords = self.reproject()

            ### record max e dimensions for use in onnx exporting ###
            self.max_edges_count.append(int(self.pg.net.shape[1]))

            with autocast(enabled=True):
                corr = self.corr(coords)
                ctx = self.imap[:, self.pg.kk % (self.M * self.pmem)]
                self.pg.net, delta, weight = self.update_inner(self.pg.net, corr, ctx, self.pg.ii, self.pg.jj, self.pg.kk)

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            target = coords[...,self.P//2,self.P//2] + delta.float()

        self.pg.target = target
        self.pg.weight = weight

        with Timer("BA", enabled=self.enable_timing):
            try:
                # run global bundle adjustment if there exist long-range edges
                if (self.pg.ii < self.n - self.cfg.REMOVAL_WINDOW - 1).any() and not self.ran_global_ba[self.n]:
                    self.__run_global_BA()
                else:
                    t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
                    t0 = max(t0, 1)
                    fastba.BA(self.poses, self.patches, self.intrinsics, 
                        target, weight, lmbda, self.pg.ii, self.pg.jj, self.pg.kk, t0, self.n, M=self.M, iterations=2, eff_impl=False)
            except:
                print("Warning BA failed...")

            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.pg.points_[:len(points)] = points[:]

    def __edges_forw(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')

    def __edges_back(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-r, 0), self.n, device="cuda"), indexing='ij')

    def __call__(self, tstamp, image, intrinsics):
        """ track new frame """

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc(image, self.n)

        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--opts BUFFER_SIZE={self.N*2}"')

        if self.viewer is not None:
            self.viewer.update_image(image.contiguous())

        image = 2 * (image[None,None] / 255.0) - 0.5
        # stop h/w being dynamic in pathcify to help with onnx export

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            if self._onnx_fnet is not None and self._onnx_inet is not None:
                if verbose: print(f'Running onnx features only')
                # Hybrid: run fnet/inet with ONNX, rest with PyTorch
                feed = {"images": image.cpu().numpy().astype(np.float32)}
                fmap = torch.from_numpy(self._onnx_fnet.run(None, feed)[0]).cuda().to(image.dtype)
                imap = torch.from_numpy(self._onnx_inet.run(None, feed)[0]).cuda().to(image.dtype)
                fmap, gmap, imap, patches, _, clr = self.network.patchify.forward_from_maps(
                    fmap, imap, image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME,
                    centroid_sel_strat=self.cfg.CENTROID_SEL_STRAT,
                    return_color=True
                    )
            elif self._onnx_patchify is not None:
                if verbose: print(f'Running onnx patchify')
                feed = {"images": image.cpu().numpy().astype(np.float32),
                "patches_per_image": np.array(self.cfg.PATCHES_PER_FRAME, dtype=np.int64)}
                fmap, gmap, imap, patches, _, clr = \
                    self._onnx_patchify.run(output_names=None, input_feed=feed, run_options=None)
                fmap = torch.from_numpy(fmap).to('cuda')
                gmap = torch.from_numpy(gmap).to('cuda')
                imap = torch.from_numpy(imap).to('cuda')
                patches = torch.from_numpy(patches).to('cuda')
                clr = torch.from_numpy(clr).to('cuda')
            else:
                if verbose: print(f'Running pytorch patchify')
                fmap, gmap, imap, patches, _, clr = \
                    self.network.patchify(image,
                        patches_per_image=self.cfg.PATCHES_PER_FRAME,
                        centroid_sel_strat=self.cfg.CENTROID_SEL_STRAT,
                        return_color=True
                        )

        ### update state attributes ###
        self.tlist.append(tstamp)
        self.pg.tstamps_[self.n] = self.counter
        self.pg.intrinsics_[self.n] = intrinsics / self.RES

        # color info for visualization
        clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
        self.pg.colors_[self.n] = clr.to(torch.uint8)

        self.pg.index_[self.n + 1] = self.n + 1
        self.pg.index_map_[self.n + 1] = self.m + self.M

        if self.n > 1:
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.pg.poses_[self.n-1])
                P2 = SE3(self.pg.poses_[self.n-2])

                # To deal with varying camera hz
                *_, a,b,c = [1]*3 + self.tlist
                fac = (c-b) / (b-a)

                xi = self.cfg.MOTION_DAMPING * fac * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.pg.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n-1]
                self.pg.poses_[self.n] = tvec_qvec

        # TODO better depth initialization
        patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
        if self.is_initialized:
            s = torch.median(self.pg.patches_[self.n-3:self.n,:,2])
            patches[:,:,2] = s

        self.pg.patches_[self.n] = patches

        ### update network attributes ###
        self.imap_[self.n % self.pmem] = imap.squeeze()
        self.gmap_[self.n % self.pmem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1        
        if self.n > 0 and not self.is_initialized:
            if self.motion_probe() < 2.0:
                self.pg.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1
        self.m += self.M

        if self.cfg.LOOP_CLOSURE:
            if self.n - self.last_global_ba >= self.cfg.GLOBAL_OPT_FREQ:
                """ Add loop closure factors """
                lii, ljj = self.pg.edges_loop()
                if lii.numel() > 0:
                    self.last_global_ba = self.n
                    self.append_factors(lii, ljj)

        # Add forward and backward factors
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True

            for itr in range(12):
                self.update()

        elif self.is_initialized:
            self.update()
            self.keyframe()

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.attempt_loop_closure(self.n)
            self.long_term_lc.lc_callback()
