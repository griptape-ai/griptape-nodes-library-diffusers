"""Memory-safe neighborhood attention.

Neighborhood attention: every token in a (T, H, W) grid attends only to the `kernel_size` window around itself.
The default diffusers processor builds a dense (N x N) score matrix over all N = T*H*W tokens, which runs out of
memory at production sizes. This processor gives similar results in bounded memory:

    1. Split the grid into small TILES of queries.
    2. Each tile only needs keys/values from a slightly larger HALO region (tile + kernel - 1 per axis).
    3. Run ordinary attention per tile, masked to each query's exact window.
    4. Stitch the tile outputs back into the full grid.

Shape legend used below: B = batch, T/H/W = frames/height/width, Q = queries per tile, K = keys per tile (halo).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch  # type: ignore[reportMissingImports]
    from diffusers.models.autoencoders.ltx2_diffusion_decoder import (  # type: ignore[reportMissingImports]
        LTX2VideoVaeNeighborhoodAttention,
    )

DEFAULT_MAX_WINDOW_ELEMENTS = 2**27
TARGET_TILE_QUERIES = 128


def _default_tile_size(grid: tuple[int, int, int], kernel_size: tuple[int, int, int]) -> tuple[int, int, int]:
    """Pick a (t, h, w) tile shape of about `TARGET_TILE_QUERIES` queries, shaped roughly like the kernel."""
    tile = [1, 1, 1]
    while tile[0] * tile[1] * tile[2] < TARGET_TILE_QUERIES:
        growable = [axis for axis in range(3) if tile[axis] < grid[axis]]
        if not growable:
            break
        axis = min(growable, key=lambda a: tile[a] / kernel_size[a])
        tile[axis] = min(tile[axis] * 2, grid[axis])
    return (tile[0], tile[1], tile[2])


def _axis_window_mask(query_pos: torch.Tensor, region_pos: torch.Tensor, kernel: int, size: int) -> torch.Tensor:
    """Per-tile membership of each region position in each query's window, shape `(tiles, queries, region)`.

    One axis only. True means "this halo key is inside this query's window on this axis", e.g. kernel 3:

        query 4 -> window [3, 5];  halo keys 3..8 -> [T T T F F F]
    """
    import torch  # type: ignore[reportMissingImports]

    # Shift the window inward at grid edges so it always holds exactly `kernel` positions.
    start = torch.clamp(query_pos - kernel // 2, 0, size - kernel)[:, :, None]
    region = region_pos[:, None, :]
    return (region >= start) & (region < start + kernel)


def _gather_tiles(x: torch.Tensor, pos: list[torch.Tensor]) -> torch.Tensor:
    """`x` is `(B, T, H, W, ...)`; returns `(B, tiles, t, h, w, ...)` gathered at per-tile axis positions.

    `pos[axis]` is `(tiles, n_axis)`; the three index tensors broadcast into a (tiles, t, h, w) cut-out per tile.
    """
    return x[:, pos[0][:, :, None, None], pos[1][:, None, :, None], pos[2][:, None, None, :]]


class ChunkedNeighborhoodAttnProcessor:
    """Neighborhood attention over small query tiles, each attending to its halo region with fused SDPA.

    Tiles are processed in batches so only `max_window_elements` of gathered keys are alive at once, and the dense
    score matrix is never materialised (SDPA's memory-efficient kernel streams it).

    Args:
        max_window_elements: Memory budget per batch of tiles; sets how many tiles run per SDPA call.
        tile_size: Fixed (t, h, w) query tile. `None` picks one automatically via `_default_tile_size`.
    """

    def __init__(
        self,
        max_window_elements: int = DEFAULT_MAX_WINDOW_ELEMENTS,
        tile_size: tuple[int, int, int] | None = None,
    ) -> None:
        self.max_window_elements = max_window_elements
        self.tile_size = tile_size

    def __call__(
        self, attn: LTX2VideoVaeNeighborhoodAttention, hidden_states: torch.Tensor, block_mask=None
    ) -> torch.Tensor:
        import torch  # type: ignore[reportMissingImports]
        import torch.nn.functional as F  # type: ignore[reportMissingImports]  # noqa: N812

        batch_size, num_frames, height, width, _ = hidden_states.shape
        grid = (num_frames, height, width)
        kernel = attn.kernel_size
        if len(kernel) != 3:
            raise ValueError(f"Neighborhood attention requires a 3D kernel, got {kernel}.")
        kernel_size = (kernel[0], kernel[1], kernel[2])
        device = hidden_states.device
        # Each is (B, T, H, W, heads, head_dim).
        query, key, value = attn.project_qkv(hidden_states)

        # --- 1. Plan the tiling ---
        # tile = queries per tile, region = halo of keys per tile, both (t, h, w) and clamped to the grid.
        requested = self.tile_size or _default_tile_size(grid, kernel_size)
        tile = tuple(min(requested[axis], grid[axis]) for axis in range(3))
        region = tuple(min(tile[axis] + kernel_size[axis] - 1, grid[axis]) for axis in range(3))

        # Ceiling division: the last tile on an axis may overhang the grid.
        tiles_per_axis = tuple(-(-grid[axis] // tile[axis]) for axis in range(3))
        num_tiles = tiles_per_axis[0] * tiles_per_axis[1] * tiles_per_axis[2]
        queries_per_tile = tile[0] * tile[1] * tile[2]
        keys_per_tile = region[0] * region[1] * region[2]

        # How many tiles fit in the memory budget at once (at least 1).
        gathered_per_tile = batch_size * keys_per_tile * attn.heads * attn.head_dim
        mask_per_tile = queries_per_tile * keys_per_tile
        tiles_per_chunk = max(
            1, min(self.max_window_elements // gathered_per_tile, self.max_window_elements // mask_per_tile)
        )

        # --- 2. Process tiles one chunk at a time ---
        offsets = [torch.arange(tile[axis], device=device) for axis in range(3)]
        region_offsets = [torch.arange(region[axis], device=device) for axis in range(3)]
        hw_tiles = tiles_per_axis[1] * tiles_per_axis[2]
        outputs = []
        for first in range(0, num_tiles, tiles_per_chunk):
            flat = torch.arange(first, min(first + tiles_per_chunk, num_tiles), device=device)
            chunk_tiles = flat.shape[0]
            tile_index = (flat // hw_tiles, (flat % hw_tiles) // tiles_per_axis[2], flat % tiles_per_axis[2])

            # Grid positions (per axis, per tile) of this chunk's queries and of their halo keys.
            query_pos = []
            region_pos = []
            for axis in range(3):
                tile_start = tile_index[axis] * tile[axis]
                # Ragged edge tiles repeat the last row; those outputs are cropped below.
                query_pos.append((tile_start[:, None] + offsets[axis]).clamp_max(grid[axis] - 1))
                # Halo starts where the tile's first query's window starts, slid back if it would leave the grid.
                window_start = torch.clamp(tile_start - kernel_size[axis] // 2, 0, grid[axis] - kernel_size[axis])
                region_start = torch.clamp(window_start, max=grid[axis] - region[axis])
                region_pos.append(region_start[:, None] + region_offsets[axis])

            # A key is in a query's 3D window only if it is in the window on all three axes.
            mask_t, mask_h, mask_w = (
                _axis_window_mask(query_pos[axis], region_pos[axis], kernel_size[axis], grid[axis]) for axis in range(3)
            )
            # Broadcast the three axis masks to (tiles, qt, qh, qw, kt, kh, kw), then flatten to (tiles, Q, K).
            mask = (
                mask_t[:, :, None, None, :, None, None]
                & mask_h[:, None, :, None, None, :, None]
                & mask_w[:, None, None, :, None, None, :]
            ).reshape(1, chunk_tiles, 1, queries_per_tile, keys_per_tile)
            # Share the mask across batch items and heads: (B * tiles, 1, Q, K).
            mask = mask.expand(batch_size, -1, -1, -1, -1).reshape(-1, 1, queries_per_tile, keys_per_tile)

            # Cut out each tile's queries and halo keys/values; fold tiles into the batch: (B * tiles, Q or K, heads, head_dim).
            batched = batch_size * chunk_tiles
            query_tile = _gather_tiles(query, query_pos).reshape(batched, queries_per_tile, attn.heads, attn.head_dim)
            key_tile = _gather_tiles(key, region_pos).reshape(batched, keys_per_tile, attn.heads, attn.head_dim)
            value_tile = _gather_tiles(value, region_pos).reshape(batched, keys_per_tile, attn.heads, attn.head_dim)

            # Ordinary attention per tile; the mask limits each query to its exact window.
            # `scale=1.0`: the query is already scaled in `project_qkv`.
            out = F.scaled_dot_product_attention(
                query_tile.transpose(1, 2),
                key_tile.transpose(1, 2),
                value_tile.transpose(1, 2),
                attn_mask=mask,
                scale=1.0,
            )
            outputs.append(out.transpose(1, 2).reshape(batch_size, chunk_tiles, *tile, attn.heads, attn.head_dim))

        # --- 3. Stitch tiles back into the full grid ---
        tiled = torch.cat(outputs, dim=1)
        tiled = tiled.reshape(batch_size, *tiles_per_axis, *tile, attn.heads, attn.head_dim)
        # Interleave per axis -> (B, nT, t, nH, h, nW, w, ...) so merging each pair gives a global coordinate.
        tiled = tiled.permute(0, 1, 4, 2, 5, 3, 6, 7, 8)
        padded = tiled.reshape(
            batch_size,
            tiles_per_axis[0] * tile[0],
            tiles_per_axis[1] * tile[1],
            tiles_per_axis[2] * tile[2],
            attn.heads * attn.head_dim,
        )
        # Drop the padding added by ragged edge tiles.
        hidden_states = padded[:, :num_frames, :height, :width]
        return attn.to_out[0](hidden_states)
