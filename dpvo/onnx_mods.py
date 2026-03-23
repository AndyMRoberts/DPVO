# defines various functions in pytorch from cuda/other libraries that are not exportable to onnx
import torch


def neighbors(kk: torch.Tensor, jj: torch.Tensor):
    """
    PyTorch implementation of fastba.neighbors. For each element, returns the
    previous (ix) and next (jx) temporal neighbor index within the same kk group,
    when ordering by jj. -1 means no previous/next neighbor.
    Fully vectorized (no Python loops) so ONNX export is fast and trace-friendly.
    """
    # Sort by kk first, then jj (PyTorch has no lexsort; use composite key).
    jj_max = jj.max() + 1
    key = kk * jj_max + jj
    order = torch.argsort(key)
    n = kk.shape[0]
    device = kk.device
    ix = torch.full((n,), -1, dtype=torch.long, device=device)
    jx = torch.full((n,), -1, dtype=torch.long, device=device)
    ix[order[1:]] = order[:-1]
    jx[order[:-1]] = order[1:]
    return ix, jx


def pad_update_inputs_like_dpvo(
    net,
    ctx,
    corr,
    ii,
    jj,
    kk,
    edges_padded_value,
    flow=None,
):
    """
    Match DPVO.update_inner when use_edges_padding is True: pad edge dimension
    to a fixed size (zeros for net/ctx/corr/flow; zeros for ii/jj/kk indices).

    Shapes: net, ctx [B, E, D], corr [B, E, Cc], ii/jj/kk [E], flow [B, E, F] or None.
    Returns (net_p, ctx_p, corr_p, flow_p, ii_p, jj_p, kk_p). flow_p is None if flow is None.
    """
    B, E_real, D = net.shape
    _, _, Cc = corr.shape
    pad_value = int(edges_padded_value) - int(E_real)
    if pad_value < 0:
        raise ValueError(
            f"Number of edges exceeds padding: edges_padded_value={edges_padded_value} "
            f"< E_real={E_real}. Increase edges_padded_value by {-pad_value}."
        )

    net_pad = torch.zeros(B, pad_value, D, device=net.device, dtype=net.dtype)
    ctx_pad = torch.zeros(B, pad_value, D, device=ctx.device, dtype=ctx.dtype)
    corr_pad = torch.zeros(B, pad_value, Cc, device=corr.device, dtype=corr.dtype)
    ii_pad = torch.zeros(pad_value, device=ii.device, dtype=ii.dtype)
    jj_pad = torch.zeros(pad_value, device=jj.device, dtype=jj.dtype)
    kk_pad = torch.zeros(pad_value, device=kk.device, dtype=kk.dtype)

    net_p = torch.cat([net, net_pad], dim=1)
    ctx_p = torch.cat([ctx, ctx_pad], dim=1)
    corr_p = torch.cat([corr, corr_pad], dim=1)
    ii_p = torch.cat([ii, ii_pad], dim=0)
    jj_p = torch.cat([jj, jj_pad], dim=0)
    kk_p = torch.cat([kk, kk_pad], dim=0)

    if flow is None:
        flow_p = None
    else:
        _, _, Ff = flow.shape
        flow_pad = torch.zeros(B, pad_value, Ff, device=flow.device, dtype=flow.dtype)
        flow_p = torch.cat([flow, flow_pad], dim=1)

    return net_p, ctx_p, corr_p, flow_p, ii_p, jj_p, kk_p


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
    To replace the torch_scatter implementation.
    For the scatter_softmax use case we want an output with the SAME SHAPE
    as src, where entries that share the same index along `dim` contain the
    group's maximum. The caller is responsible for broadcasting `index`
    to match src's shape.
    """
    # Make dim non-negative
    if dim < 0:
        dim = src.dim() + dim

    # Ensure index has same shape/rank as src along dim
    index = broadcast(index, src, dim)

    # Start from the identity for max (-inf) and ignore the existing values via
    # include_self=False so we only reduce over `src`.
    out = torch.full_like(src, float("-inf"))
    out = out.scatter_reduce_(dim, index, src, reduce="amax")

    # Return a dummy second value for compatibility with torch_scatter API.
    return out, index

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
    # Make dim non-negative
    if dim < 0:
        dim = src.dim() + dim

    # Same semantics as torch_scatter_max above: output has the SAME SHAPE as
    # src. Allow index to be 1-D or already broadcast; normalize to src shape.
    index = broadcast(index, src, dim)

    out = torch.zeros_like(src)
    out = out.scatter_reduce_(dim, index, src, reduce="sum")
    return out





def torch_scatter_softmax(src, index, dim=1, dim_size=None):
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


