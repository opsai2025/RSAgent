"""
Cutting a large scene into tiles (V02). Pure functions, no models.

    make_tiles(scene_size, tile_size, strategy) -> [Tile]      row-major, 1-based
    crop(image, tile) -> PIL image of exactly tile.size        never padded
    vlm_context(tile) -> the line that tells the VLM it sees one tile

Rules
  * both sides <= tile_size  -> ONE tile = the whole image, processed unsplit
  * otherwise each side longer than tile_size is cut, the other side is kept whole

No tile is ever padded. PIL's Image.crop() silently fills anything outside the
image with black, which would hand the tools and the VLM fake "no-data" borders
(and change the image size the VLM is told about). Every window produced here
lies inside the scene, and crop() asserts it. Two ways to get there:

  "balanced" (default)  n = ceil(side / tile_size) tiles of (almost) equal
                        length, edges at round(i * side / n). Every pixel is in
                        exactly one tile: no overlap, so independent tile answers
                        never count the same object twice, and no thin sliver
                        tiles (2010 px -> 2 x 1005, not 2000 + 10).
                        7000 x 3000 -> 4 x 2 tiles of 1750 x 1500.
  "fixed"               tiles of exactly tile_size; the last row / column starts
                        at side - tile_size, i.e. it is shifted back inside the
                        image and OVERLAPS its neighbour instead of being padded.
                        7000 x 3000 -> 4 x 2 tiles of 2000 x 2000, column 4 at
                        x 5000-6999 (shares x 5000-5999 with column 3).
"""

import math
from typing import List, Tuple

from PIL import Image

import config
from schemas import Tile

STRATEGIES = ("balanced", "fixed")


def axis_spans(length: int, tile_size: int, strategy: str = "balanced") -> List[Tuple[int, int]]:
    """Half-open [start, end) windows covering 0..length along one axis."""
    if length <= 0 or tile_size <= 0:
        raise ValueError("length and tile_size must be positive (got %r, %r)" % (length, tile_size))
    if strategy not in STRATEGIES:
        raise ValueError("unknown tiling strategy %r (use one of %s)" % (strategy, STRATEGIES))
    if length <= tile_size:
        return [(0, length)]
    n = math.ceil(length / tile_size)
    if strategy == "balanced":
        edges = [round(i * length / n) for i in range(n + 1)]
        return [(edges[i], edges[i + 1]) for i in range(n)]
    starts = [i * tile_size for i in range(n - 1)] + [length - tile_size]
    return [(s, s + tile_size) for s in starts]


def needs_tiling(scene_size, tile_size: int = None) -> bool:
    tile_size = tile_size or config.TILE_SIZE
    return max(scene_size) > tile_size


def make_tiles(scene_size, tile_size: int = None, strategy: str = None) -> List[Tile]:
    """Row-major tiles (left to right, then top to bottom), named Tile 1..N."""
    tile_size = tile_size or config.TILE_SIZE
    strategy = strategy or config.TILE_STRATEGY
    w, h = scene_size
    xs, ys = axis_spans(w, tile_size, strategy), axis_spans(h, tile_size, strategy)
    count = len(xs) * len(ys)
    tiles = []
    for r, (y0, y1) in enumerate(ys, start=1):
        for c, (x0, x1) in enumerate(xs, start=1):
            tiles.append(Tile(index=len(tiles) + 1, count=count, row=r, col=c,
                              rows=len(ys), cols=len(xs), x0=x0, y0=y0, x1=x1, y1=y1,
                              scene_w=w, scene_h=h))
    return tiles


def crop(image: Image.Image, tile: Tile) -> Image.Image:
    """The tile's pixels, exactly tile.size -- refuses any window that would make
    PIL pad the crop."""
    w, h = image.size
    if not (0 <= tile.x0 < tile.x1 <= w and 0 <= tile.y0 < tile.y1 <= h):
        raise ValueError("%s window %s is outside the %dx%d image -- it would be padded"
                         % (tile.name, tile.box, w, h))
    if tile.is_whole_scene and tile.size == image.size:
        return image
    return image.crop(tile.box)


def overlap_note(tiles: List[Tile]) -> str:
    """Human-readable overlap summary ('' when tiles do not overlap)."""
    if len(tiles) < 2:
        return ""
    xs = sorted({(t.x0, t.x1) for t in tiles})
    ys = sorted({(t.y0, t.y1) for t in tiles})
    parts = ["columns share x %d–%d" % (b[0], a[1] - 1) for a, b in zip(xs, xs[1:]) if b[0] < a[1]]
    parts += ["rows share y %d–%d" % (b[0], a[1] - 1) for a, b in zip(ys, ys[1:]) if b[0] < a[1]]
    return "; ".join(parts)


def grid_summary(tiles: List[Tile]) -> str:
    """'8 tiles (4 columns × 2 rows) · 1750 × 1500 px' -- for captions. Balanced
    tiles can differ by 1 px (rounding): '1781 × 1897–1898 px'."""
    if not tiles:
        return "no tiles"

    def span(values):
        lo, hi = min(values), max(values)
        return "%d" % lo if lo == hi else "%d–%d" % (lo, hi)

    size_txt = "%s × %s" % (span([t.size[0] for t in tiles]), span([t.size[1] for t in tiles]))
    t = tiles[0]
    return "%d tile%s (%d column%s × %d row%s) · %s px" % (
        t.count, "s" if t.count != 1 else "", t.cols, "s" if t.cols != 1 else "",
        t.rows, "s" if t.rows != 1 else "", size_txt)


def vlm_context(tile: Tile) -> str:
    """Told to the VLM before the question, so it answers for the tile it sees.
    Empty for an unsplit image -- then the prompt is exactly V01's."""
    if tile.is_whole_scene:
        return ""
    w, h = tile.size
    return ("IMAGE CONTEXT: image 1 is %s of %d, cut WITHOUT resizing or padding from a "
            "larger %d x %d px scene (scene pixels x %d-%d, y %d-%d; grid row %d of %d, "
            "column %d of %d). Its size is %d x %d px and all boxes are relative to it. "
            "Answer the question for THIS tile only; the other tiles are answered "
            "separately. Objects at the tile's borders may be cut off -- judge them from "
            "their visible part."
            % (tile.name, tile.count, tile.scene_w, tile.scene_h, tile.x0, tile.x1 - 1,
               tile.y0, tile.y1 - 1, tile.row, tile.rows, tile.col, tile.cols, w, h))
