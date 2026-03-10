# defines various functions in pytorch from cuda/other libraries that are not exportable to onnx
import torch

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    """
    Taken from torch_scatter
    """
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def torch_scatter_max(src, index, dim, dim_size=None):
    """
    To replace the torch_scatter implementation
    """
    index = broadcast(index, src, dim)

    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)

    out = out.scatter_reduce_(dim ,index, src, reduce="amax", include_self=False)
    out = broadcast(out, index, dim)
    
    return out

def torch_scatter_sum(src, index, dim, dim_size=None):
    """
    To replace the torch_scatter implementation:
    class ScatterSum : public torch::autograd::Function<ScatterSum> {
        public:
        static variable_list forward(AutogradContext *ctx, Variable src,
                               Variable index, int64_t dim,
                               std::optional<Variable> optional_out,
                               std::optional<int64_t> dim_size) {
        dim = dim < 0 ? src.dim() + dim : dim;
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["src_shape"] = src.sizes();
        index = broadcast(index, src, dim);
        auto result = scatter_fw(src, index, dim, optional_out, dim_size, "sum");
        auto out = std::get<0>(result);
        ctx->save_for_backward({index});
        if (optional_out.has_value())
        ctx->mark_dirty({optional_out.value()});
        return {out};
        }
    """
    index = broadcast(index, src, dim)

    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1

    out = torch.zeros(size, dtype=src.dtype, device=src.device)

    out = out.scatter_reduce_(dim, index, src, reduce="sum", include_self=False)
    out = broadcast(out, index, dim)
    return out





def torch_scatter_softmax(src, index, dim=-1, dim_size=None):
    """
    --------
    To replace the torch_scatter implementation
    --------
    def scatter_softmax(src: torch.Tensor, index: torch.Tensor,
                    dim: int = -1,
                    dim_size: Optional[int] = None) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise ValueError('`scatter_softmax` can only be computed over tensors '
                         'with floating point data types.')

    index = broadcast(index, src, dim)

    max_value_per_index = scatter_max(
        src, index, dim=dim, dim_size=dim_size)[0]
    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp_()

    sum_per_index = scatter_sum(
        recentered_scores_exp, index, dim, dim_size=dim_size)
    normalizing_constants = sum_per_index.gather(dim, index)

    return recentered_scores_exp.div(normalizing_constants)
    """

    if dim_size is None:
        dim_size = int(index.max()) + 1

    if not torch.is_floating_point(src):
        raise ValueError('`scatter_softmax` can only be computed over tensors '
                         'with floating point data types.')

    index = broadcast(index, src, dim)

    max_value_per_index = torch_scatter_max(
        src, index, dim=dim, dim_size=dim_size)[0]

    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp_()

    sum_per_index = torch_scatter_sum(
        recentered_scores_exp, index, dim, dim_size=dim_size)
    normalizing_constants = sum_per_index.gather(dim, index)

    return recentered_scores_exp.div(normalizing_constants)


