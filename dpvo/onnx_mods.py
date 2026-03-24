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


def _dim_size_from_index(index: torch.Tensor, dim: int, dim_size=None):
    if dim_size is not None:
        return int(dim_size)
    return int(index.max().item()) + 1


def torch_scatter_max(src, index, dim, dim_size=None):
    """
    Match torch_scatter.scatter_max reduced output along `dim`:
    out[..., g, ...] = max over src positions that map to g.

    Returns (out_reduced, index_broadcast) where out_reduced.shape[dim] == dim_size.
    Second value mirrors torch_scatter (callers in onnx_mods only use [0]).
    """
    if dim < 0:
        dim = src.dim() + dim
    index = broadcast(index, src, dim)
    idx = index.long()
    ds = _dim_size_from_index(idx, dim, dim_size)
    out_shape = list(src.shape)
    out_shape[dim] = ds
    out = torch.full(
        out_shape, float("-inf"), dtype=src.dtype, device=src.device
    )
    # out is -inf so amax(-inf, x) == x; duplicate indices aggregate like torch_scatter.scatter_max
    out.scatter_reduce_(dim, idx, src, reduce="amax")
    return out, index


def torch_scatter_sum(src, index, dim, dim_size=None):
    """
    Match torch_scatter.scatter_sum: reduced tensor along `dim` (same as torch_scatter),
    not same shape as src. Use .gather(dim, index) to broadcast to per-edge layout.
    """
    if dim < 0:
        dim = src.dim() + dim
    index = broadcast(index, src, dim)
    idx = index.long()
    ds = _dim_size_from_index(idx, dim, dim_size)
    out_shape = list(src.shape)
    out_shape[dim] = ds
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    out.scatter_add_(dim, idx, src)
    return out


def torch_scatter_softmax(src, index, dim=1, dim_size=None):
    """
    Match torch_scatter.scatter_softmax (same math as PyG); output shape == src.shape.
    """
    if not torch.is_floating_point(src):
        raise ValueError(
            "`torch_scatter_softmax` can only be computed over tensors "
            "with floating point data types."
        )

    if dim < 0:
        dim = src.dim() + dim
    index = broadcast(index, src, dim)
    idx = index.long()
    if dim_size is None:
        dim_size = int(idx.max().item()) + 1

    max_value_per_index = torch_scatter_max(src, idx, dim=dim, dim_size=dim_size)[0]
    max_per_src_element = max_value_per_index.gather(dim, idx)

    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp()

    sum_per_index = torch_scatter_sum(
        recentered_scores_exp, idx, dim=dim, dim_size=dim_size
    )
    normalizing_constants = sum_per_index.gather(dim, idx)

    return recentered_scores_exp.div(normalizing_constants)
