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
  // diameter
  const int D = 2*R + 2;

  const int B = coords.size(0);
  const int M = coords.size(1);
  const int C = net.size(1);
  const int H = net.size(2);
  const int W = net.size(3);

  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < B * M * D * D) {
    const int ii = n % D; n /= D;
    const int jj = n % D; n /= D;
    const int  m = n % M; n /= M;

    const float x = coords[n][m][0];
    const float y = coords[n][m][1];
    const int i = static_cast<int>(floor(y)) + (ii - R);
    const int j = static_cast<int>(floor(x)) + (jj - R);

    if (within_bounds(i, j, H, W)) {
      for (int k=0; k<C; k++)
        patches[n][m][k][ii][jj] = net[n][k][i][j];
    }
  }
}
    """
    
    B = coords.shape[0]
    M = coords.shape[1]
    C = net.shape[1]
    D = 2 * radius + 2
    H = net.shape[2]
    W = net.shape[3]

    opts = net.options()
    patches = torch.zeros(B, M, C, D, D, device=net.device, dtype=net.dtype)



    return patches