"""Scale the measured two-sessions-one-GPU run to a working day (a model, not a run).

Two sessions alternate GPU training and CPU data preparation, half a cycle
apart, and swap runtimes at every phase boundary. The measured run supplies the
swap overhead (first PATCH start to last PATCH return of each swap round) and
the list prices; the model supplies the phase length and the number of cycles.

Shared: one GPU and one CPU runtime for the whole wall-clock time, including
every swap, charged at both rates (conservative: during a swap a session is
briefly on neither). Dedicated: each session holds its own GPU for its working
time and runs its CPU phases there at no slowdown (the cheapest the baseline
could be). Reads results/e16/e16-micro-2.json; writes figures/out/share_model.json.

    python figures/share_model.py [--phase-min 30] [--hours 8]
    python figures/share_model.py --record /tmp/soma-rerun/e16/e16-rerun.json --out /tmp/soma-rerun/share_model.json
"""
import argparse
import json
import os
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent           # figures/
ROOT = HERE.parent                                # repository root (holds results/)
# server_list_price, the evaluation's primary rate table (e14_lifecycle.RATE_TABLES)
CPU_USD_H = 0.5328
GPU_USD_H = {'gpu_a100_40': 2.1 + 32 * 0.0079992 + 4 * 0.04716, 'gpu_t4': 0.59 + 16 * 0.0079992 + 2 * 0.04716}


def swap_overheads(rec):
    """Time between the last phase of one round ending and the next round starting: both
    moves plus the state checks around them, so everything a swap costs the pair."""
    ph = rec['phases']
    n = min(len(ph['A']), len(ph['B']))
    return [min(ph[w][r + 1]['start_s'] for w in 'AB') - max(ph[w][r]['end_s'] for w in 'AB')
            for r in range(n - 1)]


def measured_window(rec):
    """Measured runtime intervals over the completed rounds (first phase start to last completed phase
    end), at list prices; the dedicated baseline is the same phases each on its own GPU (not run)."""
    g, c = GPU_USD_H[rec['gpu']] / 3600, CPU_USD_H / 3600
    ph = rec['phases']
    t0 = min(ph[w][0]['start_s'] for w in 'AB')
    t1 = max(ph[w][-1]['end_s'] for w in 'AB')
    clip = lambda iv: max(0.0, min(iv['end_s'], t1) - max(iv['start_s'], t0))  # noqa: E731
    ivs = [iv for w in 'AB' for iv in rec['runtimes'][w]['runtimes']]
    gpu_held = sum(clip(iv) for iv in ivs if iv['sku'] != 'cpu')
    cpu_held = sum(clip(iv) for iv in ivs if iv['sku'] == 'cpu')
    train = sum(p['end_s'] - p['start_s'] for w in 'AB' for p in ph[w] if p['kind'] == 'gpu')
    work = sum(p['end_s'] - p['start_s'] for w in 'AB' for p in ph[w])
    shared, dedicated = gpu_held * g + cpu_held * c, work * g
    gpu_ivs = sorted((iv['start_s'], iv['end_s']) for iv in ivs if iv['sku'] != 'cpu')
    overlap = any(b0 < a1 for (a0, a1), (b0, b1) in zip(gpu_ivs, gpu_ivs[1:]))
    return {'window_s': t1 - t0, 'rounds': min(len(ph['A']), len(ph['B'])), 'gpu_held_s': gpu_held,
            'cpu_held_s': cpu_held, 'gpu_train_s': train, 'shared_usd': shared, 'dedicated_usd': dedicated,
            'saving_pct': 100 * (1 - shared / dedicated), 'gpu_busy_pct_shared': 100 * train / gpu_held,
            'gpu_busy_pct_dedicated': 100 * train / work, 'gpu_runtimes_ever_overlap': overlap,
            'moves': len(rec['moves']), 'moves_state_equal': sum(c['ok'] for c in rec['checks']),
            'move_patch_s': [m['patch_return_s'] - m['patch_start_s'] for m in rec['moves']]}


def model(swap_s, gpu, phase_s, hours):
    g, c = GPU_USD_H[gpu] / 3600, CPU_USD_H / 3600
    phases = int(round(hours * 3600 / phase_s))          # per session; half on the GPU
    swaps = phases - 1
    work_s = phases * phase_s
    wall_s = work_s + swaps * swap_s
    shared = wall_s * (g + c)
    dedicated = 2 * work_s * g
    gpu_train_s = 2 * (phases // 2) * phase_s
    return {'gpu': gpu, 'phase_s': phase_s, 'hours': hours, 'swap_s': swap_s, 'swaps': swaps,
            'shared_usd': shared, 'dedicated_usd': dedicated, 'saving_pct': 100 * (1 - shared / dedicated),
            'gpu_hours_shared': wall_s / 3600, 'gpu_hours_dedicated': 2 * work_s / 3600,
            'gpu_busy_pct_shared': 100 * gpu_train_s / wall_s, 'gpu_busy_pct_dedicated': 100 * gpu_train_s / (2 * work_s),
            'finish_delay_pct': 100 * (wall_s / work_s - 1)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--record', default=str(ROOT / 'results/e16/e16-micro-2.json'))
    ap.add_argument('--phase-min', type=float, default=30)
    ap.add_argument('--hours', type=float, default=8)
    ap.add_argument('--out', default=str(HERE / 'out/share_model.json'))
    a = ap.parse_args()
    rec = json.loads(Path(a.record).read_text())
    sw = swap_overheads(rec)
    swap_s = statistics.median(sw)
    record = Path(os.path.relpath(Path(a.record).resolve(), ROOT)).as_posix()   # repo-relative, no host paths
    out = {'measured': {'record': record, 'swap_overheads_s': sw, 'swap_median_s': swap_s,
                        'summary': rec['summary'], 'window': measured_window(rec)},
           'day': model(swap_s, rec['gpu'], a.phase_min * 60, a.hours),
           'day_t4': model(swap_s, 'gpu_t4', a.phase_min * 60, a.hours),
           'day_2x_swap': model(2 * swap_s, rec['gpu'], a.phase_min * 60, a.hours),
           'note': 'model: measured swap overhead and list prices applied to a longer schedule; not a run'}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))
