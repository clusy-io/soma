"""Figure 1: a move as a two-lane swimlane timeline (Source / Destination).

Four equal phase columns share one time axis. Every phase boundary carries one
event: admission closes (1|2), the capsule moves (2|3), the journal commit (3|4).
Laid out in inches (axes units == page inches), so a font size here is the size
on the page. Every label is measured and checked against its space and against
every other label and shape before saving.
"""
from pathlib import Path
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

OUT = Path(__file__).resolve().parent / 'out'   # figures/out/
OUT.mkdir(parents=True, exist_ok=True)
INK = '#12161d'
GREEN, GREEN_DARK, GREEN_PALE = '#17D1A6', '#0B8A6C', '#E4F8F2'
SLATE, SLATE_MID, SLATE_LIGHT = '#3d4a5c', '#8a96a8', '#d5dbe3'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK, 'pdf.fonttype': 42,
                     'hatch.linewidth': .5, 'hatch.color': GREEN_DARK,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})

PAD = .02                         # savefig pad_inches; the tight box is the whole axes
W_PAGE = 3.33                     # final width on the page (one ACM column)
W = W_PAGE - 2 * PAD
LABEL, TITLE = 6.5, 6.8           # box / annotation labels; phase and lane titles (bold)
MEASURE_DPI = 600                 # hinting at low dpi distorts widths; measure near true metrics
fig = plt.figure(figsize=(W, 2.0), dpi=MEASURE_DPI)
ax = fig.add_axes([0, 0, 1, 1])


def textw(s, size, weight='normal'):
    t = ax.text(0, 0, s, fontsize=size, weight=weight)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


# ---- horizontal grid: a narrow gutter for the vertical lane titles, then four equal phase columns ----
LANE_T = 5.8                                     # lane titles, set vertically so the columns get the width
X_TITLE = .07                                    # centre of the vertical lane titles
GUT = .15
X_END = W - .005
COL = (X_END - GUT) / 4
xs = [GUT + i * COL for i in range(5)]           # column edges; xs[1], xs[2], xs[3] are the events

# ---- vertical grid, bottom up ----
LANE_H, GAP = 0.31, 0.35
Y_JW = 0.055                                      # 'journal write' tag under the commit marker
yd0 = Y_JW + 0.12
yd1 = yd0 + LANE_H                                # destination lane
ys0 = yd1 + GAP
ys1 = ys0 + LANE_H                                # source lane
Y_ABORT = ys1 + 0.14                              # dashed return path
Y_HEAD = Y_ABORT + 0.18                           # phase header row
RC = 0.050                                        # phase circle radius
H = Y_HEAD + RC + .006
assert H + 2 * PAD <= 2.0, H
fig.set_size_inches(W, H)
ax.set(xlim=(0, W), ylim=(0, H))
ax.axis('off')

texts = []        # (artist, container (x0, y0, x1, y1) in inches, or None)
shapes = []       # (name, (x0, y0, x1, y1)): boxes no foreign label may touch
STY = {'authority': dict(facecolor=GREEN, edgecolor=GREEN_DARK),
       'closed': dict(facecolor=GREEN_PALE, edgecolor=GREEN_DARK, hatch='//////'),
       'work': dict(facecolor=GREEN_PALE, edgecolor=GREEN_DARK),
       'idle': dict(facecolor='white', edgecolor=SLATE_MID, linestyle=(0, (2, 1.4))),
       'gone': dict(facecolor=SLATE_LIGHT, edgecolor=SLATE_MID)}


def label(x, y, s, size=LABEL, weight='normal', ha='center', box=None, bg=False, z=6, rot=0):
    t = ax.text(x, y, s, fontsize=size, weight=weight, ha=ha, va='center', zorder=z, rotation=rot,
                linespacing=1.15 if '\n' in s else None,
                bbox=dict(facecolor='white', edgecolor='none', pad=1.2) if bg else None)
    texts.append((t, box))
    return t


def seg(c0, c1, y0, style, s=None, bg=False):
    xa, xb = xs[c0], xs[c1]
    ax.add_patch(Rectangle((xa, y0), xb - xa, LANE_H, lw=.6, zorder=2, **STY[style]))
    r = (xa, y0, xb, y0 + LANE_H)
    shapes.append((s or style, r))
    if s:
        label((xa + xb) / 2, y0 + LANE_H / 2, s, box=r, bg=bg)


# ---- lanes ----
label(X_TITLE, (ys0 + ys1) / 2, 'Source', LANE_T, 'bold', rot=90, box=(0, ys0 - .1, GUT - .005, ys1 + .1))
label(X_TITLE, (yd0 + yd1) / 2, 'Destination', LANE_T, 'bold', rot=90, box=(0, yd0 - .16, GUT - .005, yd1 + .16))

seg(0, 1, ys0, 'authority', 'runs cells')
seg(1, 3, ys0, 'closed', 'new cells refused', bg=True)
seg(3, 4, ys0, 'gone', 'released')
seg(0, 1, yd0, 'work', 'boot,\ninspect')
seg(1, 2, yd0, 'idle', 'idle')
seg(2, 3, yd0, 'work', 'restore,\nvalidate')
seg(3, 4, yd0, 'authority', 'runs cells')

# ---- phase header: a numbered circle at the start of each phase, its name beside it ----
for i, name in enumerate(['Preflight', 'Capture', 'Restore', 'Commit']):
    x = xs[i]
    ax.add_patch(Circle((x, Y_HEAD), RC, facecolor='white', edgecolor=GREEN_DARK, lw=.8, zorder=4))
    shapes.append((f'circle {i + 1}', (x - RC, Y_HEAD - RC, x + RC, Y_HEAD + RC)))
    label(x, Y_HEAD - .002, str(i + 1), 6.3, 'bold', z=5, box=(x - RC, Y_HEAD - RC, x + RC, Y_HEAD + RC))
    # the container stops .02 short of the next circle, so a name never crowds it
    label(x + RC + .026, Y_HEAD, name, TITLE, 'bold', ha='left',
          box=(x + RC, Y_HEAD - .1, (xs[i + 1] - RC - .02) if i < 3 else W + .02, Y_HEAD + .1))

# ---- capsule: one vertical arrow from source to destination at the capture/restore boundary ----
xc, ym = xs[2], (ys0 + yd1) / 2
pw, ph = textw('capsule', LABEL) + .13, .155
pill = (xc - pw / 2, ym - ph / 2, xc + pw / 2, ym + ph / 2)
ax.add_patch(FancyBboxPatch(pill[:2], pw, ph, boxstyle='round,pad=0,rounding_size=.0775',
                            facecolor=GREEN_PALE, edgecolor=GREEN_DARK, lw=.7, zorder=4))
shapes.append(('capsule', pill))
label(xc, ym, 'capsule', box=pill)
CONTENTS = 'objects, files,\npackages, RNG'
t = ax.text(pill[0] - .06, ym, CONTENTS, fontsize=6.3, ha='right', va='center', linespacing=1.15, zorder=6)
texts.append((t, (xs[0], ys0 - GAP, pill[0], ys0)))
ax.plot([xc, xc], [ys0, pill[3]], color=GREEN_DARK, lw=.9, zorder=3, solid_capstyle='butt')
ax.add_patch(FancyArrowPatch((xc, pill[1]), (xc, yd1), arrowstyle='-|>', mutation_scale=6.5,
                             color=GREEN_DARK, lw=.9, shrinkA=0, shrinkB=0, zorder=3))

# ---- commit: one highlighted marker through both lanes, tagged with what it is ----
xk, d = xs[3], .056
ax.plot([xk, xk], [Y_JW + .06, Y_HEAD - RC], color=INK, lw=1.1, zorder=5, solid_capstyle='butt')
# authority moves down, from the source lane to the destination lane, at the commit
ax.add_patch(FancyArrowPatch((xk, ym + .05), (xk, ym - .05), arrowstyle='-|>', mutation_scale=7,
                             color=INK, lw=1.1, shrinkA=0, shrinkB=0, zorder=6))
label(xk + .05, ym, 'authority', ha='left', box=(xk + .02, ym - .1, W + .02, ym + .1))
label(xk, Y_JW, 'one journal write')

# ---- abort: dashed return path from just before commit back to the running source ----
xa_from, xa_to = xk - .10, (xs[0] + xs[1]) / 2
dash = dict(color=SLATE, lw=.8, ls=(0, (2.2, 1.6)))
ax.plot([xa_from, xa_from, xa_to], [ys1, Y_ABORT, Y_ABORT], zorder=3, **dash)
ax.add_patch(FancyArrowPatch((xa_to, Y_ABORT), (xa_to, ys1), arrowstyle='-|>', mutation_scale=6.5,
                             color=SLATE, lw=.8, linestyle=(0, (2.2, 1.6)), shrinkA=0, shrinkB=0, zorder=3))
# vertical strokes count as shapes too (the abort label deliberately interrupts its own dashed line)
for name, x, y0, y1 in [('commit line', xk, Y_JW + .06, Y_HEAD - RC), ('capsule line (out)', xc, pill[3], ys0),
                        ('capsule line (in)', xc, yd1, pill[1]),
                        ('abort leg (from)', xa_from, ys1, Y_ABORT), ('abort leg (to)', xa_to, ys1, Y_ABORT)]:
    shapes.append((name, (x - .012, y0, x + .012, y1)))
label((xa_from + xa_to) / 2, Y_ABORT, 'any failure: source reopens', bg=True,
      box=(xa_to + .05, Y_ABORT - .1, xa_from - .05, Y_ABORT + .1))


# ---- checks, at the PNG resolution and near true font metrics ----
def check(dpi):
    fig.set_dpi(dpi)
    fig.canvas.draw()
    inv = ax.transData.inverted()
    boxes = []
    for t, c in texts:
        b = t.get_window_extent(fig.canvas.get_renderer()).transformed(inv)
        boxes.append((t.get_text(), (b.x0, b.y0, b.x1, b.y1), c))
    for s, (x0, y0, x1, y1), c in boxes:
        assert 0 <= x0 and x1 <= W and 0 <= y0 and y1 <= H, ('outside figure', s, dpi)
        if c:
            pad = .012 if c[2] - c[0] < .15 else .025    # digits inside the phase circles
            assert x0 >= c[0] + pad and x1 <= c[2] - pad, ('too wide', s, round(x1 - x0, 3), dpi)
            assert y0 >= c[1] and y1 <= c[3], ('too tall', s, dpi)
        for name, r in shapes:
            if c and all(abs(a - b) < 1e-9 for a, b in zip(c, r)):
                continue            # its own box
            hit = x0 < r[2] and r[0] < x1 and y0 < r[3] and r[1] < y1
            assert not hit, ('label on a shape', s, name, dpi)
    for (s1, a, _), (s2, b, _) in itertools.combinations(boxes, 2):
        assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]), ('overlap', s1, s2, dpi)
    return boxes


for dpi in (300, 600):
    check(dpi)
fig.set_dpi(300)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'protocol.{ext}', dpi=300, bbox_inches='tight', pad_inches=PAD)
from PIL import Image
px = Image.open(OUT / 'protocol.png').size
print(f'page size {px[0] / 300:.3f} x {px[1] / 300:.3f} in; gutter {GUT:.3f}; column {COL:.3f}')
assert abs(px[0] / 300 - W_PAGE) < 1.5 / 300 and px[1] / 300 <= 2.0, px
