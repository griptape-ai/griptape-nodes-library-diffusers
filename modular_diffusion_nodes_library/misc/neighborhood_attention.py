import math

import torch  # type: ignore[reportMissingImports]
from diffusers.models.autoencoders.ltx2_diffusion_decoder import (  # type: ignore[reportMissingImports]
    LTX2VideoVaeNeighborhoodAttention,
)

DEFAULT_MAX_WINDOW_ELEMENTS = 2**27


def _window_indices(
    query_index: torch.Tensor, grid: tuple[int, int, int], kernel_size: tuple[int, int, int]
) -> torch.Tensor:
    """Flattened key indices of each query's window, shape `(len(query_index), prod(kernel_size))`."""
    num_frames, height, width = grid
    kernel_t, kernel_h, kernel_w = kernel_size
    hw = height * width
    q_t = query_index // hw
    q_h = (query_index % hw) // width
    q_w = query_index % width

    # Shift the window inward at grid edges so it always holds exactly `kernel_size` positions.
    start_t = torch.clamp(q_t - kernel_t // 2, 0, num_frames - kernel_t)
    start_h = torch.clamp(q_h - kernel_h // 2, 0, height - kernel_h)
    start_w = torch.clamp(q_w - kernel_w // 2, 0, width - kernel_w)

    device = query_index.device
    t = start_t[:, None] + torch.arange(kernel_t, device=device)
    h = start_h[:, None] + torch.arange(kernel_h, device=device)
    w = start_w[:, None] + torch.arange(kernel_w, device=device)
    indices = t[:, :, None, None] * hw + h[:, None, :, None] * width + w[:, None, None, :]
    return indices.reshape(query_index.shape[0], -1)


class ChunkedNeighborhoodAttnProcessor:
    """Neighborhood attention that gathers each query's window in chunks, avoiding a dense score matrix."""

    def __init__(self, max_window_elements: int = DEFAULT_MAX_WINDOW_ELEMENTS) -> None:
        self.max_window_elements = max_window_elements

    def __call__(
        self, attn: LTX2VideoVaeNeighborhoodAttention, hidden_states: torch.Tensor, block_mask=None
    ) -> torch.Tensor:
        batch_size, num_frames, height, width, _ = hidden_states.shape
        grid = (num_frames, height, width)
        seq_len = num_frames * height * width
        query, key, value = attn.project_qkv(hidden_states)
        query = query.reshape(batch_size, seq_len, attn.heads, attn.head_dim)
        key = key.reshape(batch_size, seq_len, attn.heads, attn.head_dim)
        value = value.reshape(batch_size, seq_len, attn.heads, attn.head_dim)

        window_size = math.prod(attn.kernel_size)
        chunk_size = max(1, self.max_window_elements // (batch_size * window_size * attn.heads * attn.head_dim))
        outputs = []
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            window = _window_indices(torch.arange(start, end, device=hidden_states.device), grid, attn.kernel_size)
            window_key = key[:, window].float()
            window_value = value[:, window].float()
            scores = torch.einsum("bqhd,bqkhd->bqhk", query[:, start:end].float(), window_key).softmax(dim=-1)
            outputs.append(torch.einsum("bqhk,bqkhd->bqhd", scores, window_value).to(value.dtype))

        hidden_states = torch.cat(outputs, dim=1)
        hidden_states = hidden_states.reshape(batch_size, num_frames, height, width, attn.heads * attn.head_dim)
        return attn.to_out[0](hidden_states)
