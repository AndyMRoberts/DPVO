import torch


def patchify_forward_pytorch(net, coords, radius):
    """
    Original Cuda Function

    std::vector<torch::Tensor> patchify_cuda_forward(
    torch::Tensor net, torch::Tensor coords, int radius)
    {
    const int B = coords.size(0);
    const int M = coords.size(1);
    const int C = net.size(1);
    const int D = 2 * radius + 2;

    auto opts = net.options();
    auto patches = torch::zeros({B, M, C, D, D}, opts);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(net.type(), "patchify_forward_kernel", ([&] {
        patchify_forward_kernel<scalar_t><<<BLOCKS(B * M * D * D), THREADS>>>(radius,
            net.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            coords.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            patches.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
    }));

    return { patches };
    } 

        const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> net,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> coords,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> patches)
{
  // diameter (R = 3)
  const int D = 2*R + 2; 8 

  const int B = coords.size(0); 1
  const int M = coords.size(1); 10 (patch index)
  const int C = net.size(1); 3 (channel index)
  const int H = net.size(2); 100 (
  const int W = net.size(3); 100  

  int n = blockIdx.x * blockDim.x + threadIdx.x;
  // n < 1*10*8*8 = 640
  if (n < B * M * D * D) { 
    // Each thread “decodes” its own n back into ii, jj, m, b.
    const int ii = n % D; n /= D; // in patch row
    const int jj = n % D; n /= D; // in patch column
    const int  m = n % M; n /= M;

    const float x = coords[n][m][0];
    const float y = coords[n][m][1];

    // get actual image coords to check for 
    const int i = static_cast<int>(floor(y)) + (ii - R);
    const int j = static_cast<int>(floor(x)) + (jj - R);

    if (within_bounds(i, j, H, W)) {
      for (int k=0; k<C; k++)
        patches[n][m][k][ii][jj] = net[n][k][i][j];
    }
  }
}
    """
    R = radius
    coords = coords.to(net.device)
    
    B = coords.shape[0]
    M = coords.shape[1]
    C = net.shape[1]
    # due to bilinear interpolation and scaling net
    # feature map is scaled down
    # coords is full size
    D = 2 * R + 2
    H = net.shape[2]
    W = net.shape[3]

    patches = torch.zeros(B, M, C, D, D, device=net.device, dtype=net.dtype)
    x = coords[..., 0]
    y = coords[..., 1]
    # get patch centres, still a vector
    j0 = torch.floor(x).long()
    i0 = torch.floor(y).long()

    # get offsets inside the patch
    offs = torch.arange(D, device=net.device, dtype=j0.dtype) - R
    # grid of each coords inside each patch
    off_ii, off_jj = torch.meshgrid(offs, offs, indexing="ij")
    # convert grid to actual pixel coords (not patch coords)
    i = i0[:,:, None, None] + off_ii[None, None, :, :]
    j = j0[:,:, None, None] + off_jj[None, None, :, :]
    # ensure use only values within image bounds
    # produces a boolean tensor the same shape as i or j
    valid = (i >= 0) & (i < H) & (j >= 0) & (j < W)  # (B, M, D, D)

    # clamp indices as extra redundancy
    i_clamp = i.clamp(0, H - 1)
    j_clamp = j.clamp(0, W - 1)

    # prepare b and m indices
    b_idx = torch.arange(B, device=net.device)[:, None, None, None]
    m_idx = torch.arange(M, device=net.device)[None, :, None, None]

    # get patches 
    patches = net[b_idx, :, i_clamp, j_clamp] # (B, M, C, D, D)
    # permute to correct shape
    patches = patches.permute(0, 1, 4, 2, 3)

    # final check that if patches are not valid replace with zeros tensor
    patches = torch.where(
      valid[:, :, None, :, :], 
      patches,
      torch.zeros_like(patches),
    )



    return patches