"""Optional feature pooling and explicit encoder chunk contracts."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class AttentionPool(nn.Module):
    def __init__(self, features, heads=1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(features, features), nn.Tanh(), nn.Linear(features, heads)
        )

    def forward(self, features, mask=None):
        scores = self.attention(features)
        if mask is not None:
            if mask.shape != features.shape[:2] or not mask.any(dim=1).all():
                raise ValueError("Every bag needs a valid slice and an aligned mask")
            scores = scores.masked_fill(~mask.bool().unsqueeze(-1), -torch.inf)
        return torch.einsum("bdh,bdf->bhf", scores.softmax(dim=1), features)


class GRUPool(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.gru = nn.GRU(features, features, batch_first=True)

    def forward(self, features, mask=None):
        if mask is not None:
            lengths = mask.sum(1)
            expected = (
                torch.arange(features.size(1), device=mask.device)[None]
                < lengths[:, None]
            )
            if (lengths == 0).any() or not torch.equal(expected, mask.bool()):
                raise ValueError("GRU masks must be nonempty contiguous valid prefixes")
            features = pack_padded_sequence(
                features, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        _, hidden = self.gru(features)
        return hidden[-1]


def encode_chunks(encoder, slices, chunk_size=None):
    """Chunk size changes training normalization. Reject unequal DDP call shapes."""
    if chunk_size is None:
        return encoder(slices)
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("encoder_chunk_size must be a positive integer")
    if encoder.training and torch.distributed.is_initialized():
        local = torch.tensor([len(slices), chunk_size], device=slices.device)
        shapes = [
            torch.empty_like(local) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(shapes, local)
        if any(not torch.equal(shape, local) for shape in shapes):
            raise ValueError(
                "Chunked distributed training requires equal slice counts and chunk sizes"
            )
    return torch.cat([encoder(chunk) for chunk in slices.split(chunk_size)], dim=0)
