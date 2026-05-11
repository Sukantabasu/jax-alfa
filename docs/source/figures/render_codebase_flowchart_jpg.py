from PIL import Image, ImageDraw, ImageFont
import math
from pathlib import Path
import textwrap


OUT = Path(__file__).with_name("JAXALFA_Codebase_Flowchart.jpg")
W, H = 2600, 1700
MARGIN = 70


def font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


F_TITLE = font(44, True)
F_GROUP = font(28, True)
F_NODE = font(24)
F_NODE_B = font(24, True)
F_SMALL = font(21)
F_CAPTION = font(22)

COL = {
    "bg": (255, 255, 255),
    "ink": (35, 38, 42),
    "line": (82, 88, 96),
    "setup": (231, 244, 255),
    "loop": (235, 249, 241),
    "sgs": (255, 247, 229),
    "kernel": (242, 238, 255),
    "out": (246, 246, 246),
    "node": (255, 255, 255),
    "node_blue": (238, 247, 255),
    "node_green": (238, 249, 242),
    "node_orange": (255, 247, 232),
    "node_purple": (245, 242, 255),
}

img = Image.new("RGB", (W, H), COL["bg"])
draw = ImageDraw.Draw(img)


def rounded_rect(xy, fill, outline=(170, 176, 184), radius=18, width=3):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def group_box(xy, title, fill):
    rounded_rect(xy, fill=fill, outline=(195, 202, 210), radius=24, width=3)
    x1, y1, x2, _ = xy
    draw.text((x1 + 22, y1 + 16), title, fill=COL["ink"], font=F_GROUP)


def wrap_lines(text, max_chars):
    lines = []
    for part in text.split("\n"):
        if not part:
            lines.append("")
        else:
            lines.extend(textwrap.wrap(part, width=max_chars, break_long_words=False))
    return lines


def node(name, x, y, w, h, text, fill=None, bold_first=True, max_chars=22):
    fill = fill or COL["node"]
    rounded_rect((x, y, x + w, y + h), fill=fill, outline=(120, 130, 140), radius=14, width=3)
    lines = wrap_lines(text, max_chars)
    line_heights = [F_NODE_B.getbbox(line)[3] - F_NODE_B.getbbox(line)[1] if i == 0 and bold_first else F_NODE.getbbox(line)[3] - F_NODE.getbbox(line)[1] for i, line in enumerate(lines)]
    total_h = sum(line_heights) + 8 * (len(lines) - 1)
    yy = y + (h - total_h) / 2 - 2
    for i, line in enumerate(lines):
        f = F_NODE_B if i == 0 and bold_first else F_NODE
        bbox = draw.textbbox((0, 0), line, font=f)
        draw.text((x + (w - (bbox[2] - bbox[0])) / 2, yy), line, fill=COL["ink"], font=f)
        yy += line_heights[i] + 8
    return (x, y, x + w, y + h)


def diamond(name, cx, cy, w, h, text):
    pts = [(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)]
    draw.polygon(pts, fill=COL["node_orange"], outline=(120, 130, 140))
    draw.line(pts + [pts[0]], fill=(120, 130, 140), width=3)
    lines = wrap_lines(text, 16)
    yy = cy - (len(lines) * 28) / 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=F_SMALL)
        draw.text((cx - (bbox[2] - bbox[0]) / 2, yy), line, fill=COL["ink"], font=F_SMALL)
        yy += 28
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def side(box, where):
    x1, y1, x2, y2 = box
    if where == "r":
        return (x2, (y1 + y2) / 2)
    if where == "l":
        return (x1, (y1 + y2) / 2)
    if where == "t":
        return ((x1 + x2) / 2, y1)
    return ((x1 + x2) / 2, y2)


def arrow(p1, p2, color=None, width=4, dashed=False):
    color = color or COL["line"]
    if dashed:
        x1, y1 = p1
        x2, y2 = p2
        n = max(1, int(math.hypot(x2 - x1, y2 - y1) // 18))
        for i in range(n):
            if i % 2 == 0:
                a = i / n
                b = (i + 1) / n
                draw.line((x1 + (x2 - x1) * a, y1 + (y2 - y1) * a, x1 + (x2 - x1) * b, y1 + (y2 - y1) * b), fill=color, width=width)
    else:
        draw.line((*p1, *p2), fill=color, width=width)
    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    length = 18
    spread = 0.45
    pts = [
        p2,
        (p2[0] - length * math.cos(angle - spread), p2[1] - length * math.sin(angle - spread)),
        (p2[0] - length * math.cos(angle + spread), p2[1] - length * math.sin(angle + spread)),
    ]
    draw.polygon(pts, fill=color)


def poly_arrow(points, color=None, width=4, dashed=False):
    for a, b in zip(points[:-2], points[1:-1]):
        if dashed:
            arrow(a, b, color=color, width=width, dashed=True)
        else:
            draw.line((*a, *b), fill=color or COL["line"], width=width)
    arrow(points[-2], points[-1], color=color, width=width, dashed=dashed)


draw.text((MARGIN, 36), "JAX-ALFA Solver Workflow and Codebase Structure", fill=COL["ink"], font=F_TITLE)

group_box((60, 120, 560, 1025), "Run Setup", COL["setup"])
group_box((620, 120, 1110, 1560), "Main Time Loop", COL["loop"])
group_box((1170, 120, 2050, 1110), "SGS Closure", COL["sgs"])
group_box((2110, 600, 2535, 1015), "Shared Kernels", COL["kernel"])
group_box((1170, 1160, 2050, 1560), "Output Products", COL["out"])

boxes = {}
boxes["cfg"] = node("cfg", 130, 205, 360, 90, "Run directory\nConfig.py", COL["node_blue"])
boxes["loader"] = node("loader", 130, 325, 360, 90, "ConfigLoader\nload run parameters", COL["node_blue"])
boxes["derived"] = node("derived", 130, 445, 360, 130, "DerivedVars\nscales, grid, filters\ninitial SGS fields", COL["node_blue"])
boxes["pre"] = node("pre", 130, 610, 360, 100, "Preprocess\nwavenumbers\nzero arrays", COL["node_blue"])
boxes["init"] = node("init", 130, 745, 360, 130, "Initialization\nu, v, w, TH\nforcing and damping", COL["node_blue"])

loop_names = [
    ("filter", "Explicit spectral\nfiltering"),
    ("surface", "Surface fluxes\nand MOST"),
    ("grad", "Velocity and scalar\ngradients"),
    ("adv", "Advection\nmomentum and scalar"),
    ("buoy", "Buoyancy\nforcing"),
    ("sgs", "Subgrid-scale\nclosure"),
    ("rhs", "Momentum and scalar\nright-hand sides"),
    ("pressure", "Pressure projection\nPoisson solve"),
    ("ab2", "AB2 time\nadvancement"),
    ("diag", "Diagnostics\nand output"),
]
for i, (key, text) in enumerate(loop_names):
    boxes[key] = node(key, 700, 200 + i * 126, 330, 82, text, COL["node_green"], max_chars=22)

boxes["dynq"] = diamond("dynq", 1300, 380, 240, 150, "Dynamic\nupdate step?")
boxes["family"] = diamond("family", 1580, 280, 220, 135, "Model\nfamily")
boxes["sm"] = node("sm", 1775, 175, 230, 120, "SM variants\noptSgs=1 LASDD\noptSgs=3 LAD", COL["node_orange"], max_chars=20)
boxes["wl"] = node("wl", 1775, 340, 230, 120, "WL variants\noptSgs=2 LASDD\noptSgs=4 LAD", COL["node_orange"], max_chars=20)
boxes["dynmom"] = node("dynmom", 1285, 535, 320, 115, "DynamicSGS\nstresses and 3D\nmomentum coefficient", COL["node_orange"], max_chars=24)
boxes["dynscal"] = node("dynscal", 1660, 535, 320, 115, "DynamicSGSscalar\nscalar fluxes and 3D\nscalar coefficient", COL["node_orange"], max_chars=24)
boxes["cache"] = node("cache", 1455, 710, 330, 95, "Cached coefficients\nlatest 3D fields", (255, 252, 239), max_chars=24)
boxes["staticsgs"] = node("staticsgs", 1225, 865, 330, 115, "StaticSGS\nStaticSGSscalar\nreuse cached fields", COL["node_orange"], max_chars=24)
boxes["divsgs"] = node("divsgs", 1635, 880, 330, 100, "Stress and scalar-flux\ndivergence", COL["node_orange"], max_chars=25)

boxes["kernels"] = node("kernels", 2165, 705, 315, 180, "Shared JAX kernels\nFFT/dealiasing\ntest filters\nstrain rates\nlocal averages\npolynomial roots", COL["node_purple"], max_chars=22)

boxes["stats"] = node("stats", 1235, 1260, 330, 85, "Statistics files\nALFA_Statistics_*.npz", COL["node"], max_chars=25)
boxes["fields"] = node("fields", 1645, 1260, 330, 85, "3D field files\nALFA_3DFields_*.npz", COL["node"], max_chars=25)
boxes["notebooks"] = node("notebooks", 1435, 1410, 340, 85, "Postprocessing\nnotebooks and figures", COL["node"], max_chars=25)

# Setup arrows
for a, b in [("cfg", "loader"), ("loader", "derived"), ("derived", "pre"), ("pre", "init")]:
    arrow(side(boxes[a], "b"), side(boxes[b], "t"))
arrow(side(boxes["init"], "r"), side(boxes["filter"], "l"))

# Main loop arrows
for a, b in zip([x[0] for x in loop_names[:-1]], [x[0] for x in loop_names[1:]]):
    arrow(side(boxes[a], "b"), side(boxes[b], "t"))
poly_arrow([side(boxes["ab2"], "l"), (655, side(boxes["ab2"], "l")[1]), (655, side(boxes["filter"], "l")[1]), side(boxes["filter"], "l")], color=(85, 125, 105))

# SGS arrows
arrow(side(boxes["sgs"], "r"), side(boxes["dynq"], "l"))
arrow((1410, 342), side(boxes["family"], "l"))
draw.text((1418, 310), "yes", font=F_SMALL, fill=COL["ink"])
arrow(side(boxes["family"], "r"), side(boxes["sm"], "l"))
arrow(side(boxes["family"], "r"), side(boxes["wl"], "l"))
draw.text((1665, 198), "SM", font=F_SMALL, fill=COL["ink"])
draw.text((1665, 390), "WL", font=F_SMALL, fill=COL["ink"])
arrow(side(boxes["sm"], "b"), side(boxes["dynmom"], "t"))
arrow(side(boxes["wl"], "b"), side(boxes["dynmom"], "t"))
arrow(side(boxes["dynmom"], "r"), side(boxes["dynscal"], "l"))
arrow(side(boxes["dynscal"], "b"), side(boxes["cache"], "t"))
arrow(side(boxes["cache"], "b"), side(boxes["divsgs"], "t"))
arrow(side(boxes["dynq"], "b"), side(boxes["staticsgs"], "t"))
draw.text((1320, 602), "no", font=F_SMALL, fill=COL["ink"])
arrow(side(boxes["cache"], "l"), side(boxes["staticsgs"], "r"), dashed=True)
draw.text((1440, 825), "reuse", font=F_SMALL, fill=COL["ink"])
arrow(side(boxes["staticsgs"], "r"), side(boxes["divsgs"], "l"))
poly_arrow([side(boxes["divsgs"], "l"), (1130, center(boxes["divsgs"])[1]), (1130, center(boxes["rhs"])[1]), side(boxes["rhs"], "r")])

# Kernels
for target in ["filter", "grad", "adv", "dynmom", "dynscal", "staticsgs", "pressure"]:
    arrow(side(boxes["kernels"], "l"), side(boxes[target], "r"), color=(120, 110, 155), width=3, dashed=True)

# Outputs
arrow(side(boxes["diag"], "r"), side(boxes["stats"], "l"))
arrow(side(boxes["stats"], "r"), side(boxes["fields"], "l"))
arrow(side(boxes["stats"], "b"), side(boxes["notebooks"], "t"))
arrow(side(boxes["fields"], "b"), side(boxes["notebooks"], "t"))

caption = (
    "Workflow of the JAX-ALFA solver. Configuration and derived variables initialize the run, "
    "then the main loop advances the velocity and scalar fields through filtering, surface fluxes, "
    "gradients, advection, buoyancy, SGS closure, pressure projection, and AB2 time stepping. "
    "For optSgs = 1..4, SGS coefficients are recomputed at dynamic update instants; between those "
    "updates, the most recently computed 3D coefficients are reused."
)
cap_lines = wrap_lines(caption, 150)
yy = 1600
for line in cap_lines:
    draw.text((80, yy), line, fill=(50, 54, 60), font=F_CAPTION)
    yy += 30

img.save(OUT, quality=95, subsampling=0)
print(OUT)
