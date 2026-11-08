import torch
import torch.distributed as dist

from sklearn import metrics


def distributed_concat(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Concatenate a tensor across distributed ranks, supporting uneven sizes."""
    if not dist.is_available() or not dist.is_initialized():
        return x

    x = x.contiguous()
    dim = dim % x.ndim
    device = x.device
    local_size = torch.tensor([x.shape[dim]], device=device, dtype=torch.long)
    size_tensors = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(size_tensors, local_size)
    sizes = [int(size.item()) for size in size_tensors]
    max_size = max(sizes)

    if x.shape[dim] < max_size:
        pad_shape = list(x.shape)
        pad_shape[dim] = max_size - x.shape[dim]
        padding = torch.zeros(pad_shape, dtype=x.dtype, device=device)
        x = torch.cat([x, padding], dim=dim)

    gathered = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, x)
    gathered = [
        tensor.narrow(dim, 0, size) for tensor, size in zip(gathered, sizes) if size > 0
    ]
    return torch.cat(gathered, dim=dim) if gathered else x.narrow(dim, 0, 0)


def _as_numpy(x: torch.Tensor):
    x = x.detach().cpu()
    if x.ndim == 2 and x.size(1) == 1:
        x = x[:, 0]
    return x.numpy()


def auc(t: torch.Tensor, p: torch.Tensor, threshold=0.5) -> torch.Tensor:
    if not torch.isfinite(t).all() or not torch.isfinite(p).all():
        raise ValueError("Metric targets and predictions must be finite")
    t = (t >= threshold).float()
    if len(t.unique()) < 2:
        return torch.tensor(float("nan"))
    return torch.tensor(
        metrics.roc_auc_score(y_true=_as_numpy(t), y_score=_as_numpy(p))
    )


def avp(t: torch.Tensor, p: torch.Tensor, threshold=0.5) -> torch.Tensor:
    if not torch.isfinite(t).all() or not torch.isfinite(p).all():
        raise ValueError("Metric targets and predictions must be finite")
    t = (t >= threshold).float()
    if len(t.unique()) < 2:
        return torch.tensor(float("nan"))
    return torch.tensor(
        metrics.average_precision_score(y_true=_as_numpy(t), y_score=_as_numpy(p))
    )


def kappa(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    if len(t.unique()) == 1:
        return torch.tensor(0)
    t, p = _as_numpy(t), _as_numpy(p)
    return torch.tensor(metrics.cohen_kappa_score(y1=t, y2=p))


def qwk(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    if len(t.unique()) == 1:
        return torch.tensor(0)
    t, p = _as_numpy(t), _as_numpy(p)
    return torch.tensor(metrics.cohen_kappa_score(y1=t, y2=p, weights="quadratic"))


def mae(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(t - p))


def mse(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return torch.mean((t - p) ** 2)


def accuracy(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return torch.mean((t == p).float())


def gather_states(state):
    """Gather possibly empty rank states without guessing target dimensions."""
    local = (
        state.detach().cpu()
        if isinstance(state, torch.Tensor)
        else (torch.cat(state).detach().cpu() if state else None)
    )
    parts = [local]
    if dist.is_available() and dist.is_initialized():
        parts = [None] * dist.get_world_size()
        dist.all_gather_object(parts, local)
    parts = [p for p in parts if p is not None]
    if not parts:
        raise ValueError("No validation samples on any rank")
    return torch.cat(parts)


def score_summary(values, name):
    scores = torch.stack(values)
    valid = torch.isfinite(scores)
    return {
        f"{name}_mean": scores.mean(),
        f"{name}_valid_classes": valid.sum(),
        f"{name}_undefined_classes": (~valid).sum(),
    }
