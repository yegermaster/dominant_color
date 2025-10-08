# color_name_lookup.py
# Find the closest human-readable color name for an (R, G, B) input.
# Uses a large union of color dictionaries (CSS4, XKCD, Tableau) from matplotlib
# and matches in CIELAB space for perceptual accuracy.

from math import pow, sqrt
from functools import lru_cache

try:
    from matplotlib.colors import CSS4_COLORS, XKCD_COLORS, TABLEAU_COLORS
except Exception as e:
    raise ImportError(
        "This module requires matplotlib. Install it with: pip install matplotlib"
    ) from e


# ---------- Color space utilities (sRGB -> XYZ -> Lab) ----------
def _hex_to_rgb_tuple(hex_str: str) -> tuple[int, int, int]:
    s = hex_str.lstrip("#")
    return tuple(int(s[i:i+2], 16) for i in (0, 2, 4))


def _srgb_to_linear(c: float) -> float:
    # c in [0,1]
    return c / 12.92 if c <= 0.04045 else pow((c + 0.055) / 1.055, 2.4)


def _rgb_to_xyz(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    # Normalize to [0,1]
    r, g, b = [x / 255.0 for x in rgb]
    # Linearize
    r, g, b = _srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)
    # sRGB D65 matrix
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    return x, y, z


def _f_lab(t: float) -> float:
    # CIE standard
    delta = 6/29
    return pow(t, 1/3) if t > delta**3 else (t / (3 * delta**2) + 4/29)


def _xyz_to_lab(xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    # D65 reference white
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x, y, z = xyz[0] / Xn, xyz[1] / Yn, xyz[2] / Zn
    fx, fy, fz = _f_lab(x), _f_lab(y), _f_lab(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return L, a, b


@lru_cache(maxsize=4096)
def _rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    return _xyz_to_lab(_rgb_to_xyz(rgb))


def _lab_distance(lab1, lab2) -> float:
    # CIE76 (Euclidean in Lab). Simple, fast, and robust for our use.
    return sqrt((lab1[0]-lab2[0])**2 + (lab1[1]-lab2[1])**2 + (lab1[2]-lab2[2])**2)


# ---------- Build the color name → RGB index ----------
@lru_cache(maxsize=1)
def _color_index():
    # Start with CSS4 names
    name_to_hex = dict(CSS4_COLORS)

    # Add XKCD colors (keys like 'xkcd:acid green'); strip 'xkcd:' for display
    for k, v in XKCD_COLORS.items():
        display = k.replace("xkcd:", "")
        # Prefer CSS4 canonical names if a collision occurs; otherwise add XKCD
        if display not in name_to_hex:
            name_to_hex[display] = v

    # Add Tableau (small set of distinct, well-known colors)
    for k, v in TABLEAU_COLORS.items():
        if k not in name_to_hex:
            name_to_hex[k] = v

    # Build final list of (name, rgb, lab) for fast nearest-neighbor search
    index = []
    seen_rgbs = set()
    for name, hexv in name_to_hex.items():
        rgb = _hex_to_rgb_tuple(hexv)
        if rgb in seen_rgbs:
            continue
        seen_rgbs.add(rgb)
        lab = _rgb_to_lab(rgb)
        index.append((name, rgb, lab))

    return index


# ---------- Public API ----------
def closest_color_name(rgb: tuple[int, int, int]) -> str:
    """
    Return the closest color name (CSS4 + XKCD + Tableau) for an (R,G,B) input.
    Matching is performed in CIELAB space (CIE76 distance).
    """
    if not (isinstance(rgb, (tuple, list)) and len(rgb) == 3):
        raise ValueError("rgb must be a 3-tuple/list of integers 0..255")

    r, g, b = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    for c in (r, g, b):
        if c < 0 or c > 255:
            raise ValueError("RGB components must be in the 0..255 range")

    target_lab = _rgb_to_lab((r, g, b))
    best_name, best_dist = None, float("inf")

    for name, _, lab in _color_index():
        d = _lab_distance(target_lab, lab)
        if d < best_dist:
            best_dist = d
            best_name = name

    return best_name


def closest_color_name_with_meta(rgb: tuple[int, int, int]) -> dict:
    """
    Like closest_color_name, but also returns distance and the matched RGB.
    Returns: {"name": str, "matched_rgb": (r,g,b), "distance_lab": float}
    """
    target_lab = _rgb_to_lab((int(rgb[0]), int(rgb[1]), int(rgb[2])))
    best = {"name": None, "matched_rgb": None, "distance_lab": float("inf")}

    for name, matched_rgb, lab in _color_index():
        d = _lab_distance(target_lab, lab)
        if d < best["distance_lab"]:
            best = {"name": name, "matched_rgb": matched_rgb, "distance_lab": d}

    return best


# ---------- Example ----------
if __name__ == "__main__":
    tests = [(255, 55, 111), (250, 128, 114), (123, 104, 238), (35, 180, 90)]
    for t in tests:
        print(t, "→", closest_color_name(t))
