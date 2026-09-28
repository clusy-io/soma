"""Regenerate every paper figure from its named cohort; never alter raw records.

Figures (all single-column, drawn in real units so a point size here is the
point size on the page):
  protocol  the handoff as a two-lane timeline (schematic, no measured data)
  cost      estimated cost of one job under each way of getting a GPU for its
            training phase, at three CPU-phase lengths (executed runs only)
  lifecycle the combined CPU->T4->CPU run: which runtime accepted work, when,
            and what happened to every routed write (one executed run)

Usage: python figures/build_figures.py [protocol fidelity share cost model_day movetime lifecycle ...]
       (default: the five paper figures: protocol fidelity share cost model_day)
Inputs are the records under results/ in this repository; outputs go to figures/out/.
"""
from pathlib import Path
import hashlib
import json
import statistics
import subprocess
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent           # figures/
EXP = HERE.parent                                 # repository root (holds results/ and experiments/)
OUT = HERE / 'out'
OUT.mkdir(parents=True, exist_ok=True)

# One colour system for the whole set. Type is always INK.
INK = '#12161d'
GREEN, GREEN_DARK, GREEN_PALE = '#17D1A6', '#0B8A6C', '#E4F8F2'
SLATE, SLATE_MID, SLATE_LIGHT = '#3d4a5c', '#8a96a8', '#d5dbe3'
RULE = '#c9ced6'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.labelsize': 7,
    'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.edgecolor': INK, 'axes.linewidth': .6,
    'xtick.major.width': .6, 'ytick.major.width': .6, 'xtick.major.size': 2.5,
    'ytick.major.size': 2.5, 'grid.color': RULE, 'grid.linewidth': .5,
    'legend.frameon': False, 'legend.fontsize': 6.5, 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'hatch.linewidth': .5, 'hatch.color': GREEN_DARK,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
COL_W = 3.33  # ACM sigconf column width, inches
PROVENANCE = {}


def save(fig, name):
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, bbox_inches='tight', pad_inches=.02)
    plt.close(fig)


def inch_axes(w, h):
    """A figure whose data coordinates are inches, so text and shapes are laid out in page units."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, w), ylim=(0, h))
    ax.axis('off')
    return fig, ax


# ---------------------------------------------------------------------------
def _textw(ax, text, size, weight='normal'):
    """Rendered width of a label in inches (so layout never guesses)."""
    r = ax.figure.canvas.get_renderer()
    t = ax.text(0, 0, text, fontsize=size, weight=weight)
    w = t.get_window_extent(renderer=r).width / ax.figure.dpi
    t.remove()
    return w


def fig_protocol():
    """The handoff as two lanes on one time axis: who accepts cells, and when."""
    W, H = COL_W, 1.60
    fig, ax = inch_axes(W, H)
    x0, x1 = 0.03, W - 0.03
    # prepare | capture | restore+validate | commit | after
    cuts = [x0, 0.78, 1.40, 2.30, 2.42, x1]
    src_y, dst_y, lane_h = 1.00, 0.46, 0.25
    styles = {'authority': dict(facecolor=GREEN, edgecolor=GREEN_DARK),
              'closed': dict(facecolor=GREEN_PALE, edgecolor=GREEN_DARK, hatch='//////'),
              'work': dict(facecolor=GREEN_PALE, edgecolor=GREEN_DARK),
              'idle': dict(facecolor='white', edgecolor=SLATE_MID, linestyle=(0, (2, 1.5))),
              'gone': dict(facecolor=SLATE_LIGHT, edgecolor=SLATE_MID)}

    def seg(xa, xb, y, style, text=None):
        ax.add_patch(Rectangle((xa, y), xb - xa, lane_h, lw=.6, **styles[style]))
        if text:
            assert _textw(ax, text, 6.3) < xb - xa - .04, text
            ax.text((xa + xb) / 2, y + lane_h / 2, text, ha='center', va='center', fontsize=6.3,
                    bbox=dict(facecolor='white', edgecolor='none', pad=.8) if style == 'closed' else None)

    # Lane titles sit above their lanes, left-aligned, clear of the capsule arrow.
    ax.text(x0, src_y + lane_h + .045, 'Source runtime', fontsize=6.8, weight='bold', va='bottom')
    ax.text(x0, dst_y + lane_h + .045, 'Destination runtime', fontsize=6.8, weight='bold', va='bottom')
    # Phase header.
    for i, name in enumerate(['1 Prepare', '2 Capture', '3 Restore, check']):
        assert _textw(ax, name, 6.5, 'bold') < cuts[i + 1] - cuts[i] - .06, name
        ax.text(cuts[i] + .04, 1.50, name, ha='left', va='center', fontsize=6.5, weight='bold')
    ax.text(cuts[3] + .04, 1.50, '4 Commit', ha='left', va='center', fontsize=6.5, weight='bold')
    for c in cuts[1:4]:
        ax.plot([c, c], [1.43, 1.57], color=RULE, lw=.6)

    seg(cuts[0], cuts[1], src_y, 'authority', 'runs cells')
    seg(cuts[1], cuts[4], src_y, 'closed', 'new cells refused')
    seg(cuts[4], x1, src_y, 'gone', 'released')
    seg(cuts[0], cuts[1], dst_y, 'work', 'boot, inspect')
    seg(cuts[1], cuts[2], dst_y, 'idle')
    seg(cuts[2], cuts[3], dst_y, 'work', 'restore, check')
    seg(cuts[4], x1, dst_y, 'authority', 'runs cells')

    # Capsule: from the end of capture straight down to the start of restore.
    xa = cuts[2]
    ax.add_patch(FancyArrowPatch((xa, src_y), (xa, dst_y + lane_h), arrowstyle='-|>', mutation_scale=7,
                                 color=GREEN_DARK, lw=.9, shrinkA=0, shrinkB=0))
    ax.text(xa + .05, (src_y + dst_y + lane_h) / 2, 'capsule', ha='left', va='center', fontsize=6.3)
    # Commit: one journal write moves authority from the upper lane to the lower.
    cx = (cuts[3] + cuts[4]) / 2
    ax.add_patch(FancyArrowPatch((cx, src_y + lane_h), (cx, dst_y), arrowstyle='-|>', mutation_scale=7,
                                 color=INK, lw=.9, shrinkA=0, shrinkB=0))
    # Abort, under the destination lane: before commit the source simply reopens.
    ax.text((cuts[2] + cuts[3]) / 2, dst_y - .09, 'fails? abort: source reopens', ha='center', va='center',
            fontsize=6.0)
    # Key, laid out from measured label widths.
    ky, kx = 0.05, x0
    for style, label in [('authority', 'accepts cells'), ('closed', 'admission closed'),
                         ('work', 'reconstruction')]:
        ax.add_patch(Rectangle((kx, ky), .15, .09, lw=.6, **styles[style]))
        ax.text(kx + .19, ky + .045, label, va='center', fontsize=6.3)
        kx += .19 + _textw(ax, label, 6.3) + .14
    assert kx < W, kx
    save(fig, 'protocol')


# ---------------------------------------------------------------------------
def _e14_rows():
    path = EXP / 'results/e14/e14_arms.jsonl'
    PROVENANCE[str(path.relative_to(EXP))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r.get('outcome') == 'completed' and not r.get('local_simulation')]


def measured_costs():
    """{(passes, arm): [usd, ...]} for the crossover cohort (every arm executed at every length)."""
    out = {}
    for r in _e14_rows():
        if r['cohort'] == 'e14-v2-crossover' and r['prep_passes'] in (3, 400, 800):
            out.setdefault((r['prep_passes'], r['arm']), []).append(r['est_runtime_cost_usd']['server_list_price'])
    return out


def paired_ratios(passes):
    """switch/always cost ratio per repetition (attempt k of each arm)."""
    by = {}
    for r in _e14_rows():
        if r['cohort'] == 'e14-v2-crossover' and r['prep_passes'] == passes and r['arm'] in ('switch', 'always'):
            by.setdefault(r.get('attempt', 1), {})[r['arm']] = r['est_runtime_cost_usd']['server_list_price']
    return [v['switch'] / v['always'] for _, v in sorted(by.items()) if len(v) == 2]


def fig_cost():
    """Measured: one job's estimated cost, keeping the GPU vs switching vs never using one."""
    t = measured_costs()
    groups = [3, 400, 800]
    arms = [('always', 'Keep the GPU', dict(color=SLATE, edgecolor=SLATE)),
            ('switch', SYSTEM, dict(color=GREEN, edgecolor=GREEN_DARK)),
            ('uninterrupted', 'CPU only', dict(color='white', edgecolor=INK))]
    fig, ax = plt.subplots(figsize=(COL_W, 1.62))
    fig.subplots_adjust(left=.115, right=.995, bottom=.215, top=.87)
    w, gap = .25, .035
    top = max(max(v) for v in t.values()) * 100
    for gi, passes in enumerate(groups):
        for ai, (arm, label, style) in enumerate(arms):
            x = gi + (ai - 1) * (w + gap)
            vals = [v * 100 for v in t[(passes, arm)]]
            m = statistics.mean(vals)
            ax.bar(x, m, w, lw=.6, zorder=3, label=label if gi == 0 else None, **style)
            hi = m
            if len(vals) > 1:
                ax.plot([x, x], [min(vals), max(vals)], color=INK, lw=.7, zorder=4)
                for v in (min(vals), max(vals)):
                    ax.plot([x - .05, x + .05], [v, v], color=INK, lw=.7, zorder=4)
                hi = max(hi, max(vals))
            ax.text(x, hi + top * .02, f'{m:.1f}', ha='center', va='bottom', fontsize=6.0,
                    weight='bold' if arm == 'switch' else 'normal')
        # SOMA against keeping the GPU, over paired runs; one label per group, above everything
        d = sorted(((r - 1) * 100 for r in paired_ratios(passes)), key=abs)
        pct = (f'{d[0]:+.0f}%' if len(d) == 1 else f'{d[0]:+.0f}% to {d[-1]:+.0f}%').replace('-', '\u2212')
        ax.text(gi, top * 1.19, pct, ha='center', va='bottom', fontsize=6.6, weight='bold',
                color=GREEN_DARK if d[-1] < 0 else INK)
    ax.set_xticks(range(len(groups)), [str(g) for g in groups])
    ax.tick_params(axis='x', length=0)
    ax.set_xlabel('Preprocessing passes in the CPU phase', labelpad=2)
    ax.set_ylabel('Job cost (US cents)', labelpad=2)
    ax.set_xlim(-.55, len(groups) - .45)
    ax.set_ylim(0, top * 1.36)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.grid(axis='y', zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=3, fontsize=6.5, handlelength=1.2,
              columnspacing=1.2, handletextpad=.45, borderaxespad=.15, frameon=False)
    save(fig, 'cost')


# Cost model for a long interactive session. Every parameter is a measured
# value or a list price from the evaluation; the session shape is the model.
RATE = {'cpu': 0.5328, 'a100': 2.5446, 't4': 0.8123}  # $/h, the evaluated list prices
MOVE_S = 120.0     # charged per move at BOTH rates; measured median end-to-end move 66 s
REBUILD_H = 0.25   # restart baseline: time to rebuild in-memory state after each move
SESSION_H, EPISODES = 8.0, 4


def model_costs(g, gpu='a100', hours=None, episodes=None):
    hours = SESSION_H if hours is None else hours
    episodes = EPISODES if episodes is None else episodes
    rc, rg = RATE['cpu'], RATE[gpu]
    keep = hours * rg
    move = 2 * episodes * MOVE_S / 3600 * (rc + rg)
    switch = hours * ((1 - g) * rc + g * rg) + move
    restart = switch + episodes * REBUILD_H * (rc + rg)
    return keep, switch, restart


def fig_model():
    import numpy as np
    g = np.linspace(0, 1, 201)
    k, s, r = zip(*(model_costs(x) for x in g))
    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    fig.subplots_adjust(left=.13, right=.985, bottom=.2, top=.97)
    ax.plot(g * 100, k, color=SLATE, lw=1.4, label='Keep the A100 all session')
    ax.plot(g * 100, r, color=SLATE_MID, lw=1.2, ls=(0, (3, 1.6)), label='Restart and rebuild state')
    ax.plot(g * 100, s, color=GREEN_DARK, lw=1.8, label='SOMA')
    ax.fill_between(g * 100, s, k, color=GREEN_PALE, zorder=0)
    for x in (10, 25, 50):
        kk, ss, _ = model_costs(x / 100)
        ax.plot([x, x], [ss, kk], color=GREEN_DARK, lw=.6, ls=(0, (1, 1.2)))
        ax.text(x + 1.2, kk - 1.1, f'\u2212{(1 - ss / kk) * 100:.0f}%', fontsize=6.5, weight='bold',
                va='top', ha='left')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(r) * 1.04)
    ax.set_xlabel('Share of the 8-hour session that needs the GPU (%)')
    ax.set_ylabel('Session cost (US$)')
    ax.grid(axis='y')
    ax.legend(loc='lower right', handlelength=2.0)
    save(fig, 'model')


NOTEBOOK_GPU_SHARE = 19  # %: reserved notebook GPUs idle more than 81% of the time (NotebookOS, ASPLOS '26)


def fig_model_day():
    """The same model over a 24-hour session: dollars per day, with the GPU share seen in notebook traces."""
    import numpy as np
    H, E = 24.0, 8
    g = np.linspace(0, 1, 401)
    k, s, r = zip(*(model_costs(x, hours=H, episodes=E) for x in g))
    fig, ax = plt.subplots(figsize=(COL_W, 1.7))
    fig.subplots_adjust(left=.13, right=.985, bottom=.235, top=.97)
    ax.axvspan(0, NOTEBOOK_GPU_SHARE, color=SLATE_LIGHT, alpha=.45, lw=0, zorder=0)
    ax.text(NOTEBOOK_GPU_SHARE / 2, 2.0, 'notebook\nGPU use', ha='center', va='bottom', fontsize=6.0,
            linespacing=1.05)
    ax.plot(g * 100, k, color=SLATE, lw=1.4, label='Keep the A100 all day')
    ax.plot(g * 100, r, color=SLATE_MID, lw=1.2, ls=(0, (3, 1.6)), label='Restart and rebuild state')
    ax.plot(g * 100, s, color=GREEN_DARK, lw=1.8, label='SOMA')
    ax.fill_between(g * 100, s, k, color=GREEN_PALE, zorder=0, alpha=.9)
    for x in (10, 40):
        kk, ss, _ = model_costs(x / 100, hours=H, episodes=E)
        ax.plot([x, x], [ss, kk], color=GREEN_DARK, lw=.6, ls=(0, (1, 1.2)))
        ax.text(x + 1.2, kk - 2.5, f'\u2212{(1 - ss / kk) * 100:.0f}%\n\\${kk - ss:.0f}/day', fontsize=6.5,
                weight='bold', va='top', ha='left', linespacing=1.1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(r) * 1.04)
    ax.set_xlabel('Share of the 24-hour session that needs the GPU (%)')
    ax.set_ylabel('Session cost (US$)')
    ax.grid(axis='y')
    ax.legend(loc='lower right', handlelength=2.0)
    save(fig, 'model_day')


# ---------------------------------------------------------------------------
def fidelity_rows():
    """Max (and min) relative difference of a RECOMPUTED value after a move, per crossing.
    Stored state was compared exactly on every one of these moves and matched."""
    sys.path.insert(0, str(EXP / 'experiments'))
    import e13_analyse
    path = EXP / 'results/e13/e13_chains.jsonl'
    PROVENANCE[str(path.relative_to(EXP))] = hashlib.sha256(path.read_bytes()).hexdigest()
    cohort, chains, notes = e13_analyse.load_cohort(path, 'e13-v2')
    s = e13_analyse.summarise(chains, cohort, notes)
    assert s['comparisons']['same_vs_control_t4_a']['bitwise_equal']
    assert s['comparisons']['same_vs_control_t4_b']['bitwise_equal']
    d = {r['hop']: r for r in s['drift']['hetero']}
    isa = EXP / 'results/xsub/isa/xsub_isa.json'
    PROVENANCE[str(isa.relative_to(EXP))] = hashlib.sha256(isa.read_bytes()).hexdigest()
    rec = json.loads(isa.read_text())
    fin = [float.fromhex(x) for x in rec['final']['losses']]
    ctl = [float.fromhex(x) for x in rec['controls']['arm64']['losses']]
    isa_rel = [abs(a - b) / abs(b) for a, b in zip(fin, ctl)]
    return [
        ('T4 \u2192 T4, five moves\n(training losses)', [0.0]),
        ('arm64 \u2194 x86-64 CPU\n(float64 training losses)', [max(isa_rel)]),
        ('T4 \u2192 CPU\n(forward output)', [d[h]['forward_rel_default'] for h in (1, 5)]),
        ('T4 \u2194 A100, TF32 off\n(forward output)', [d[h]['forward_rel_tf32_off'] for h in (3, 4, 7)]),
        ('T4 \u2194 A100, default\n(forward output)', [d[h]['forward_rel_default'] for h in (3, 4, 7)]),
    ]


def _sci(v):
    m, e = f'{v:.1e}'.split('e')
    return f'${m}\\times10^{{{int(e)}}}$'


def fig_fidelity():
    rows = fidelity_rows()
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    fig.subplots_adjust(left=.37, right=.98, bottom=.21, top=.99)
    floor = 1e-17
    for i, (label, vals) in enumerate(rows):
        y = len(rows) - 1 - i
        hi = max(vals)
        if hi == 0:
            ax.text(floor * 2.5, y, 'bitwise identical: 0', va='center', fontsize=6.5, weight='bold')
            continue
        ax.barh(y, hi - floor, .58, left=floor, color=SLATE_LIGHT, edgecolor=SLATE, lw=.6, zorder=3)
        if len(vals) > 1 and min(vals) > 0:
            ax.plot([min(vals), hi], [y, y], color=INK, lw=.7, zorder=4)
            ax.plot([min(vals)] * 2, [y - .14, y + .14], color=INK, lw=.7, zorder=4)
        import math
        inside = math.log10(hi / floor) > 8
        ax.text(floor * 2.5 if inside else hi * 2.5, y, _sci(hi), va='center', fontsize=6.5, zorder=5,
                bbox=None if inside else dict(facecolor='white', edgecolor='none', pad=.5))
    ax.axvline(1e-4, color=INK, lw=.7, ls=(0, (2, 1.5)), zorder=5)
    ax.text(1e-4 * 1.4, len(rows) - .75, 'check\ntolerance', fontsize=6.0, va='center', linespacing=1.0)
    ax.set_xscale('log')
    ax.set_xlim(floor, 3e-1)
    ax.set_xticks([1e-16, 1e-12, 1e-8, 1e-4])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(-.6, len(rows) - .35)
    ax.set_xlabel('Max relative difference of a recomputed value')
    ax.grid(axis='x', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    save(fig, 'fidelity')


# ---------------------------------------------------------------------------
def fig_lifecycle(records_path, lanes, name='lifecycle'):
    """One executed lifecycle: which runtime accepted cells when, and every routed write.

    `lanes` names the three runtimes (source, middle, return) for the reader."""
    path = Path(records_path)
    PROVENANCE[str(path.relative_to(EXP))] = hashlib.sha256(path.read_bytes()).hexdigest()
    recs = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r['name']] = r
    h1, h2a, h2b = recs['hop1'], recs['hop2a'], recs['hop2b']
    t0 = h1['events'][0]['at']
    T = lambda t: t - t0  # noqa: E731

    def first(h, phase):
        return next(T(e['at']) for e in h['events'] if e['phase'] == phase)

    def last(h, phase):
        return [T(e['at']) for e in h['events'] if e['phase'] == phase][-1]

    writes = recs['final']['writes']['entries']
    end = max(T(w.get('t_done') or w['t_sent']) for w in writes) + .15
    c1, c2 = T(h1['committed_at']), T(h2b['committed_at'])
    closed = [(0, first(h1, 'ADMISSION_CLOSED'), c1), (1, first(h2a, 'ADMISSION_CLOSED'), first(h2a, 'ABORTED')),
              (1, first(h2b, 'ADMISSION_CLOSED'), c2)]
    authority = [(0, -.25, c1), (1, c1, c2), (2, c2, end)]
    fig, ax = plt.subplots(figsize=(COL_W, 1.75))
    fig.subplots_adjust(left=.25, right=.99, bottom=.22, top=.83)
    ys = {0: 2, 1: 1, 2: 0}
    hgt = .56
    for lane, xa, xb in authority:
        ax.add_patch(Rectangle((xa, ys[lane] - hgt / 2), xb - xa, hgt, facecolor=GREEN_PALE, edgecolor=GREEN_DARK,
                               lw=.6, zorder=2))
    for lane, xa, xb in closed:
        ax.add_patch(Rectangle((xa, ys[lane] - hgt / 2), xb - xa, hgt, facecolor='white', edgecolor=GREEN_DARK,
                               hatch='//////', lw=.6, zorder=3))
    role = {'cpu-source': 0, 'gpu-runtime': 1, 'cpu-destination': 2}
    n_ack = n_ref = 0
    for w in writes:
        if w['outcome'] == 'ack':
            n_ack += 1
            y = ys[role[w['runtime_role']]]
            x = T(w['t_done'])
            ax.plot([x, x], [y - hgt / 2 + .05, y + hgt / 2 - .05], color=GREEN_DARK, lw=.45, zorder=4)
        elif w['outcome'] == 'refused':
            n_ref += 1
            x = T(w['t_sent'])
            lane = 0 if w['hop'] == 'hop1' else 1
            ax.plot([x], [ys[lane]], marker='x', ms=1.8, mew=.35, color=INK, zorder=5)
    # Events on the middle runtime: the injected abort, the crash and the resume.
    ab, kill = first(h2a, 'ABORTED'), first(h2b, 'CAPTURED')
    resumed = [T(e['at']) for e in h2b['events'] if e['phase'] == 'REQUESTED'][-1]
    top = ys[1] + hgt / 2
    for x, label, ha in [(ab, 'abort', 'center'), (kill, 'crash ', 'right'), (resumed, ' resume', 'left')]:
        ax.plot([x, x], [top, top + .3], color=INK, lw=.6, zorder=6)
        ax.text(x, top + .34, label, ha=ha, va='bottom', fontsize=6.0)
    ax.set_yticks([ys[k] for k in (0, 1, 2)], lanes)
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(-.55, 2.55)
    ax.set_xlim(-.25, end)
    ax.set_xlabel('Seconds since the first move began')
    ax.spines['left'].set_visible(False)
    # Key above the plot.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=GREEN_PALE, edgecolor=GREEN_DARK, lw=.6, label='authoritative'),
               Patch(facecolor='white', edgecolor=GREEN_DARK, hatch='//////', lw=.6, label='admission closed'),
               Line2D([], [], color=GREEN_DARK, lw=.8, label=f'acknowledged write ({n_ack})'),
               Line2D([], [], color=INK, marker='x', ls='none', ms=3, mew=.6, label=f'refused ({n_ref})')]
    ax.legend(handles=handles, ncol=2, loc='lower left', bbox_to_anchor=(-.36, 1.0), columnspacing=.9,
              handlelength=1.3, handletextpad=.4, fontsize=6.3)
    save(fig, name)
    return n_ack, n_ref


# ---------------------------------------------------------------------------
def movetime_rows(extra=()):
    """(label, serving_s, unavailable_s, after_s) per move, from live records."""
    import statistics as st
    e13 = EXP / 'results/e13/e13_chains.jsonl'
    rows13 = [json.loads(line) for line in e13.read_text().splitlines() if line.strip()]
    het = [r for r in rows13 if r.get('cohort') == 'e13-v2' and r.get('chain') == 'hetero'][0]
    lazy = [h['switch_s'] + h['first_execute_s'] for h in het['hops'] if (h.get('from'), h.get('to')) == ('cpu', 'gpu_t4')]
    e11 = EXP / 'results/e11/e11_runs.jsonl'
    PROVENANCE[str(e11.relative_to(EXP))] = hashlib.sha256(e11.read_bytes()).hexdigest()
    done = [(r['fault'], r['timings']) for r in (json.loads(line) for line in e11.read_text().splitlines() if line.strip())
            if r.get('cohort') == 'e11-r4b' and r.get('phase') == 'DONE']

    def split(t):
        closed = sum(t.get(k, 0) for k in ('drain_s', 'capture_s', 'restore_s', 'validate_s', 'boundary_s', 'commit_s'))
        return t['prepare_s'] + t['preflight_s'], closed, t.get('release_s', 0)
    rows = []
    e15 = EXP / 'results/e15/e15_records.jsonl'
    PROVENANCE[str(e15.relative_to(EXP))] = hashlib.sha256(e15.read_bytes()).hexdigest()
    for r in (json.loads(line) for line in e15.read_text().splitlines() if line.strip()):
        if r.get('cohort') == 'e15-live-6' and r.get('name') == 'hop1' and r['result'].get('phase') == 'DONE':
            rows.append(('CPU \u2192 T4',) + split(r['result']['timings']))
    for i, (fault, t) in enumerate(done, 1):
        rows.append((f'CPU \u2192 CPU, run {i}',) + split(t))
    rows += list(extra)
    return rows


SYSTEM = 'SOMA'


def fig_movetime(extra=()):
    rows = movetime_rows(extra)
    H = 0.62 + 0.30 * len(rows)
    fig, ax = plt.subplots(figsize=(COL_W, H))
    fig.subplots_adjust(left=.27, right=.97, bottom=.36 / H, top=1 - .2 / H)
    for i, (label, serve, closed, after) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, serve, .55, color=GREEN, edgecolor=GREEN_DARK, lw=.6, zorder=3)
        ax.barh(y, closed, .55, left=serve, color='white', edgecolor=GREEN_DARK if serve else SLATE, hatch='//////',
                lw=.6, zorder=3)
        if after:
            ax.barh(y, after, .55, left=serve + closed, color=GREEN_PALE, edgecolor=GREEN_DARK, lw=.6, zorder=3)
        ax.text(serve + closed / 2, y, f'{closed:.0f} s', ha='center', va='center', fontsize=6.3, zorder=5,
                bbox=dict(facecolor='white', edgecolor='none', pad=.6))
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('Seconds from the move request')
    ax.grid(axis='x', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=GREEN, edgecolor=GREEN_DARK, lw=.6, label='source runs'),
                       Patch(facecolor='white', edgecolor=SLATE, hatch='//////', lw=.6, label='unavailable'),
                       Patch(facecolor=GREEN_PALE, edgecolor=GREEN_DARK, lw=.6, label='destination runs')],
              loc='lower left', bbox_to_anchor=(-.02, 1.0), ncol=3, fontsize=6.0, handlelength=1.2,
              columnspacing=.8, handletextpad=.4, borderaxespad=.2)
    save(fig, 'movetime')


# ---------------------------------------------------------------------------
def fig_share(record_path, name='share'):
    """Two cyclic sessions sharing one GPU: where each ran over time, and how many GPU runtimes were alive."""
    path = Path(record_path)
    PROVENANCE[str(path.relative_to(EXP))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rec = json.loads(path.read_text())
    gpu = rec['gpu']
    fig = plt.figure(figsize=(COL_W, 1.72))
    ax = fig.add_axes([0.13, 0.43, 0.85, 0.44])
    ax2 = fig.add_axes([0.13, 0.17, 0.85, 0.18], sharex=ax)
    m = 60.0
    # Stop where the last phase was due to end (the last move's return plus one phase): teardown is not the story.
    stop = max(mv['patch_return_s'] for mv in rec['moves']) + rec['phase_s']
    lanes = {'A': 1, 'B': 0}
    h = 0.62
    for w, y in lanes.items():
        for iv in rec['runtimes'][w]['runtimes']:
            on_gpu = iv['sku'] != 'cpu'
            ax.add_patch(Rectangle((iv['start_s'] / m, y - h / 2), (min(iv['end_s'], stop) - iv['start_s']) / m, h,
                                   facecolor=GREEN if on_gpu else SLATE_LIGHT, edgecolor=GREEN_DARK if on_gpu else SLATE_MID,
                                   lw=.5, zorder=2))
        for mv in rec['moves']:
            if mv['workload'] != w:
                continue
            a, b = mv['patch_start_s'] / m, mv['patch_return_s'] / m
            ax.add_patch(Rectangle((a, y - h / 2), b - a, h, facecolor='white', edgecolor=INK, hatch='//////',
                                   lw=.5, zorder=3))
    ax.set_yticks([1, 0], ['Session A', 'Session B'])
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(-.55, 1.55)
    end = stop / m
    ax.set_xlim(0, end)
    ax.spines['left'].set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    # GPU runtimes alive over time.
    ev = []
    for w in 'AB':
        for iv in rec['runtimes'][w]['runtimes']:
            if iv['sku'] != 'cpu':
                ev.append((iv['start_s'] / m, 1))
                if iv['end_s'] < stop:
                    ev.append((iv['end_s'] / m, -1))
    ev.sort()
    xs, ys, c = [0.0], [0], 0
    for x, d in ev:
        xs += [x, x]; ys += [c, c + d]; c += d
    xs.append(end); ys.append(c)
    ax2.plot(xs, ys, color=GREEN_DARK, lw=1.1)
    ax2.axhline(2, color=SLATE, lw=.9, ls=(0, (3, 1.6)))
    ax2.text(end * .995, 2.12, 'dedicated: 2 GPUs', ha='right', va='bottom', fontsize=6.0)
    ax2.set_ylim(0, 2.6)
    ax2.set_yticks([0, 1, 2])
    ax2.set_ylabel('GPUs', fontsize=6.5)
    ax2.set_xlabel('Minutes')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=SLATE_LIGHT, edgecolor=SLATE_MID, lw=.5, label='CPU: prepares data'),
                       Patch(facecolor=GREEN, edgecolor=GREEN_DARK, lw=.5, label='GPU: trains'),
                       Patch(facecolor='white', edgecolor=INK, hatch='//////', lw=.5, label='moving')],
              loc='lower left', bbox_to_anchor=(-.02, 1.0), ncol=3, fontsize=6.0, handlelength=1.2,
              columnspacing=.9, handletextpad=.4, borderaxespad=.2)
    save(fig, name)


if __name__ == '__main__':
    want = sys.argv[1:] or ['protocol', 'fidelity', 'share', 'cost', 'model_day']  # 'movetime' and 'lifecycle' dropped from the paper
    if 'protocol' in want:
        # Figure 1 has its own script: a measured, overlap-checked swimlane layout.
        subprocess.run([sys.executable, str(Path(__file__).with_name('figure1_protocol.py'))], check=True)
    if 'cost' in want:
        fig_cost()
    if 'model' in want:
        fig_model()
    if 'model_day' in want:
        fig_model_day()
    if 'fidelity' in want:
        fig_fidelity()
    if 'movetime' in want:
        fig_movetime()
    if 'share' in want and (EXP / 'results/e16/e16-micro-2.json').exists():
        fig_share(EXP / 'results/e16/e16-micro-2.json')
    if 'lifecycle' in want:
        print('lifecycle writes (ack, refused):', fig_lifecycle(EXP / 'results/xsub/host_container/e15_records.jsonl',
              ['macOS host\nprocess', 'Linux\ncontainer', 'macOS host\nprocess (return)']))
    prov = {'style': {'ink': INK, 'accent': GREEN, 'accent_dark': GREEN_DARK, 'pale': GREEN_PALE,
                      'font': 'DejaVu Sans', 'formats': ['PDF vector', 'SVG', 'PNG']},
            'sources': PROVENANCE,
            'figures': {'protocol': 'schematic of the handoff protocol; not measured data',
                        'cost': 'e14-v2-main (3 passes), e14-v2-crossover (400, 800 passes); server_list_price estimate'}}
    (OUT / 'provenance.json').write_text(json.dumps(prov, indent=2) + '\n')
    print('generated', want)
