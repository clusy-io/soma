# Experiments

This document maps every result in the paper to the experiment that produced it, the command that reproduces or re-analyses it, and the record it rests on.

## Setup

```sh
pip install -e '.[torch,figures,dev]'   # Python 3.11, pinned versions; editable, as the controller reads experiments/*.py at import time
export PYTHONPATH=src:.                 # all commands below run from the repository root
```

Each experiment runs on one of three kinds of infrastructure:

- **offline**: local Python processes only, through `experiments/localapi.py`, or re-analysis of the shipped records.
- **Docker**: the runtimes are local containers. The cross-ISA move also needs amd64 emulation (Docker Desktop, or QEMU with binfmt).
- **cloud**: the runtimes are E2B CPU microVMs and Modal GPU containers (T4, L4, A100-40GB), driven through a research build of the platform API. That API is not part of this artifact (see [Cloud runs](#cloud-runs)). For every cloud experiment the artifact ships the harness, the analysis and the records, so the published numbers can be recomputed offline.

## Map from paper results to evidence

"Rerun" is the command that repeats the measurement. "Re-analyse" recomputes the paper's numbers from the shipped record. Rerun commands write to `/tmp/soma-rerun/` under a new cohort name; see [Rerunning without touching the records](#rerunning-without-touching-the-records).

| Paper result | Experiment | Command | Record | Runs on |
|---|---|---|---|---|
| Table 2, T4 to T4; "Exact continuation": 5 moves, 150 losses, parameter, optimizer, scheduler and loader digests and every generator state (including CUDA) bitwise equal to two uninterrupted controls | E13, chain `same` | Re-analyse: `python experiments/e13_analyse.py --cohort e13-v2 --out /tmp/e13_summary.json`<br>Rerun: `python experiments/e13_chain.py --cohort e13-rerun --out /tmp/soma-rerun/e13`, then `python experiments/e13_analyse.py --in /tmp/soma-rerun/e13/e13_chains.jsonl --cohort e13-rerun` | `results/e13/e13_chains.jsonl` | cloud |
| Table 2, T4, CPU and A100; "Across devices and GPU models": 9 moves, stored bytes exact, tied weights kept, step count kept, CPU and named generators equal 9/9, CUDA generators equal on 3 GPU-to-GPU moves and declared dropped on 3 moves to CPU | E13, chain `hetero` | as above | `results/e13/e13_chains.jsonl` | cloud |
| Figure `fidelity` (7.7e-4 default, 3.2e-6 without TF32, 2.4e-6 T4 to CPU, 1.5e-14 across ISAs) | E13 `hetero` and XSUB-ISA | `python figures/build_figures.py fidelity` | `results/e13/e13_chains.jsonl`, `results/xsub/isa/xsub_isa.json` | offline (from records) |
| "Outgrowing a GPU": GPT-2 needs 16.2 GiB at sequence length 1024; T4 to A100 after 20, 120 and 1080 steps; ViT T4 to L4 to A100; 7 of 10 moves verified, 1 inconclusive, 2 refused | Escalation (OOM-triggered moves) | Re-analyse: `python experiments/rederive.py escalation`<br>Rerun: see [Escalation](#escalation-oom-triggered-moves) | `results/escalation/*.jsonl` | cloud |
| Table 2, macOS process and Linux container; "Concurrent writes, an injected fault, and a crash": 101 checks pass, 63 acknowledged writes present once, 73 refused writes absent (3 while the controller was down), one authoritative runtime, abort reopened the source, crashed controller resumed | XSUB host-container (E15 lifecycle on local substrates) | Re-analyse: `python experiments/e15_analyse.py --records results/xsub/host_container/e15_records.jsonl`<br>Rerun: see [XSUB host-container](#xsub-host-container) | `results/xsub/host_container/e15_records.jsonl` | Docker |
| Table 2, arm64 and x86-64; "Across operating systems and instruction sets": two moves DONE, state equal at both boundaries, all relationships kept, losses bitwise equal to the arm64 control until the first x86-64 step | XSUB-ISA | Rerun: see [XSUB-ISA](#xsub-isa); the summary is in the record | `results/xsub/isa/xsub_isa.json` | Docker (amd64 emulated) |
| First macOS-to-Linux attempt aborted on library-dependent Gaussian draws | XSUB, first attempt | `python -m pytest tests/test_cross_substrate.py` | no record kept | offline |
| Table 2, CPU to T4 with concurrent writes: first move commits with every check passing, 20 of 20 acknowledged writes kept, 20 refused writes absent; return move refused at preflight | E15 live, cohort `e15-live-6` | Re-analyse: `python experiments/e15_analyse.py --records results/e15/e15_records.jsonl --cohort e15-live-6`<br>Rerun: `python experiments/e15_combined.py --cohort e15-rerun --outdir /tmp/soma-rerun/e15`, then `python experiments/e15_analyse.py --records /tmp/soma-rerun/e15/e15_records.jsonl --cohort e15-rerun` | `results/e15/e15_records.jsonl` | cloud |
| Table 2, CPU to CPU with injected faults; "Injected faults leave the source serving": a clean commit and 6 faults handled as designed | E11, cohort `e11-r4b` | Re-analyse: `python experiments/rederive.py e11`<br>Rerun: `python experiments/e11_transactional.py --cohort e11-rerun --outdir /tmp/soma-rerun/e11`, then `python experiments/rederive.py --results /tmp/soma-rerun --cohort e11-rerun e11` | `results/e11/e11_runs.jsonl` | cloud (E2B CPU) |
| Table 2, CPU to CPU with controller crashes; "Crash recovery": 17 of 17 crash points recovered, 8 writes refused while down, 1 orphan reclaimed, median recovery 40 s (0 to 65 s), a duplicate request yields one destination | E12, cohort `e12-r7` | Re-analyse: `python experiments/rederive.py e12`<br>Rerun: `python experiments/e12_crash.py --cohort e12-rerun --outdir /tmp/soma-rerun/e12`, then `python experiments/rederive.py --results /tmp/soma-rerun --cohort e12-rerun e12` | `results/e12/e12_runs.jsonl` | cloud (E2B CPU) |
| "The checks can fail": three reinstated defects each fail the combined verdict; regression tests | E15 local sabotage; unit tests | `python experiments/e15_combined.py --local --sabotage inplace_validation` (also `ungated_route`, `unshare_views`)<br>`python -m pytest` | none | offline |
| "How long a session is unavailable": CPU to T4 unavailable for 22 s of a 77 s move; CPU moves 16 to 26 s | E15 `e15-live-6` hop 1; E11 `e11-r4b` committed runs | `python figures/build_figures.py movetime` | `results/e15/e15_records.jsonl`, `results/e11/e11_runs.jsonl` | offline (from records) |
| Batched persistence cut a 64-file switch from 302 s to 96 s | E10, switch time against workspace file count | Re-analyse: `python experiments/rederive.py e10` | `results/e10/*.json` | not reproducible (needs the platform server) |
| Figure `cost`: 45% to 70% cheaper than keeping the GPU with a long CPU phase; GPU container prep 384 to 613 s against 131 to 134 s on the microVM | E14, cohort `e14-v2-crossover` | Re-analyse: `python experiments/e14_analyse.py --cohort e14-v2-crossover --out /tmp/e14_summary.json`; `python figures/build_figures.py cost`<br>Rerun: see [E14](#e14-a-cpu-gpu-cpu-job-under-executed-alternatives) | `results/e14/e14_arms.jsonl` | cloud |
| Restart recomputed 57 s; a complete application checkpoint needed 30 lines and cost 2.7 times as much; an incomplete one broke the analysis | E14, cohort `e14-v2-main` | Re-analyse: `python experiments/e14_analyse.py --cohort e14-v2-main --out /tmp/e14_summary.json` | `results/e14/e14_arms.jsonl` | cloud |
| Figure `model_day` and "Hours-long sessions" | cost model (list prices, 120 s per move, 15 min rebuild) | `python figures/build_figures.py model_day` | none; parameters are in the script | offline |
| Figure `share` and "Two sessions, one GPU": 4 of 4 moves equal, never two GPUs alive, swap 104 s, GPU busy 90% of the time held, 31% cheaper (over the two completed rounds; see [E16](#e16-two-cyclic-sessions-share-one-gpu)) | E16, cohort `e16-micro-2` | Re-analyse: `python figures/build_figures.py share`; `python figures/share_model.py`<br>Rerun: `python experiments/e16_cyclic.py --gpu gpu_a100_40 --phase-s 240 --cycles 2 --cohort e16-rerun --out /tmp/soma-rerun/e16`, then `python figures/share_model.py --record /tmp/soma-rerun/e16/e16-rerun.json --out /tmp/soma-rerun/share_model.json` | `results/e16/e16-micro-2.json` | cloud |
| Figure `protocol` | schematic, no data | `python figures/figure1_protocol.py` | none | offline |
| Production statistics in the abstract and in Motivation and Background | production inventory | none | not shipped | not reproducible |

E11 to E16 also have a `--local` dry run that drives the same programs through local processes (below); the escalation harness has none. Without `--substrates`, a dry run tests the harness, not the system, and nothing measured through it is a result. The XSUB experiments use the same local adapter with real host-process and container substrates, and their records are results, although `e15_analyse.py` labels the host-container record a local dry run. A dry run writes to a temporary directory unless given `--out` (`--outdir` for E11, E12 and E15) and refuses to write under `results/`.

### Rerunning without touching the records

Every live harness writes to `results/` by default and appends to the record there. The analysers and `rederive.py` select the paper's rows by cohort name and take the latest run of a cohort, so a rerun under a paper cohort would replace the paper's rows with its own, even if it failed. Give every rerun a new cohort name and a scratch output, as the Rerun commands above do. As a guard, a live run that would write under `results/` refuses a cohort already recorded there, E16 never overwrites an existing record file, the escalation harness refuses to append to its shipped files, and XSUB-ISA refuses any output under `results/`. `rederive.py --results DIR --cohort NAME` re-derives a rerun written under `DIR`.

## The experiments

### E11: injected faults

A session is moved between two E2B CPU runtimes by the transactional controller (`src/handoff/controller.py`), once cleanly and six times with a fault: a module missing at the destination, an out-of-memory during reconstruction, a corrupted capsule, a distribution missing at the destination (caught by the preflight before any capture), an open file handle (excluded from the capsule, the move commits without it), and a session object holding an open log file (excluded without being loaded, then the move is aborted). The check that matters is that after every abort the source is still authoritative and still accepts work with its state unchanged. A concurrent-PATCH probe also checks that two simultaneous requests to the platform switch produce exactly one success.

The cohort's first attempts of the last two cases coincided with a network outage on the client and are kept in the record with `as_expected: false`; both cases were rerun on the same build (`--faults poison_excluded,nested_open_log`). `rederive.py e11` prints every row and judges each case by its latest row.

```sh
python experiments/e11_transactional.py --local   # dry run; the out-of-memory case needs a memory limit and is skipped
```

### E12: controller crashes

The controller is killed at every journaled phase boundary, immediately before and after the journal write, and in the window between creating a destination and journaling its identifier: 17 crash points. A fresh controller with the same migration identifier must converge to exactly one authoritative runtime with the committed state equal to the capture cut. A write routed while the controller is down must be refused exactly when the journal's phase is inside the admission gate. A destination that was created but never journaled must be found by name and reclaimed. A separate probe sends two concurrent requests for the same move. `rederive.py e12` recomputes the counts and the recovery times.

```sh
python experiments/e12_crash.py --local                            # dry run
python experiments/e12_crash.py --local --sabotage reclaim_blind   # the orphan check fails
```

### E13: repeated moves of one training job

One ResNet-18 training job is moved repeatedly through the platform's in-place switch, with a block of training steps between moves. After every move the controller's twelve state oracles run against an expectation recorded on the source just before the move, together with checks on tied weights, tensor views, loss continuity, the optimizer step count, and the full state of every random number generator. The `same` chain alternates two memory tiers of a T4 (each move destroys and reprovisions the GPU container) and is compared bitwise with two uninterrupted T4 controls. The `hetero` chain crosses T4, CPU and A100 runtimes and is compared with the stored state before each move. Forward outputs are recomputed with and without TF32 on identical bytes, which separates arithmetic differences from state differences.

```sh
python experiments/e13_chain.py --local --out /tmp/e13   # dry run
```

### E14: a CPU, GPU, CPU job under executed alternatives

A job prepares data on a CPU, trains on a T4 and analyses on a CPU, and the analysis consumes artefacts the preparation produced. Each way of getting a GPU for the training phase is executed end to end: keeping the GPU for the whole job, never using one, moving with the in-place switch, restarting on each runtime and recomputing, and three application-level checkpoints (complete, incomplete, and incomplete without `map_location`) moved through object storage. Every arm records fingerprints of the training input and output, so the analysis can state which arms trained from identical inputs. Cost is estimated at provider list prices recorded in each row. Cohort `e14-v2-main` runs every arm once; `e14-v2-crossover` sweeps the length of the CPU phase (`--prep-passes`).

```sh
python experiments/e14_lifecycle.py --local --out /tmp/e14   # dry run, about 6 minutes
# the recorded cloud invocations, as reruns (the paper's cohorts were e14-v2-main and e14-v2-crossover)
python experiments/e14_lifecycle.py --cohort e14-rerun-main --out /tmp/soma-rerun/e14
python experiments/e14_lifecycle.py --cohort e14-rerun-crossover --out /tmp/soma-rerun/e14 --arms switch,always,uninterrupted --prep-passes 3   # also 20, 80, 180, 400, 800
python experiments/e14_lifecycle.py --cohort e14-rerun-crossover --out /tmp/soma-rerun/e14 --arms always,switch --prep-passes 400               # repeated for 800
python experiments/e14_analyse.py --in /tmp/soma-rerun/e14/e14_arms.jsonl --cohort e14-rerun-crossover --out /tmp/soma-rerun/e14_summary.json
```

### E15: concurrent writes, an injected fault and a crash in one lifecycle

One session carrying views of a shared base tensor, a Parameter view, AdamW state, a scheduler, a loader generator, NumPy views, dropout-driven generator use and workspace files goes through three moves while a writer thread routes numbered writes through the controller. The first move commits, the second is fed a corrupted capsule and must abort, and the third kills the controller after capture and resumes it with a fresh one. At every commit boundary the controller's own equivalence check runs, and an independent harness probe reads the committed state without writing to it. `e15_analyse.py` turns a record into a PASS, FAIL or SKIP table per boundary and check.

On the cloud path (cohort `e15-live-6`, E2B CPU to Modal T4) the first move passes every check. The return moves were refused at preflight because training on the GPU had imported distributions the CPU image lacks, so the analyser's overall verdict for that run is NOT OK (exit status 1); all of its failing rows belong to the refused return moves, and the paper reports only the first move.

```sh
python experiments/e15_combined.py --local   # dry run: 101 PASS, 0 FAIL, 6 SKIP (CUDA checks skipped) -> OK
python experiments/e15_combined.py --local --sabotage inplace_validation   # NOT OK; likewise ungated_route, unshare_views
```

### XSUB host-container

The E15 lifecycle with two local substrates instead of cloud runtimes: a macOS arm64 host process and a Linux arm64 container, which share no files and exchange state only through the execute channel. The recorded run used the platform's sandbox image, which is not distributed; `experiments/docker/torch-cpu.Dockerfile` builds a public substitute. The distribution sets on the two ends must match, or the preflight refuses the move (`preflight_missing_distribution`), so the host environment is pinned to the container's.

```sh
docker build -t soma-torch-cpu:py311 -f experiments/docker/torch-cpu.Dockerfile experiments/docker
uv venv --python 3.11 hostenv
uv pip install --python hostenv/bin/python torch==2.7.1 numpy==2.1.2 dill==0.3.9 setuptools tqdm
python experiments/e15_combined.py --local --outdir /tmp/hc \
  --substrates "cpu=python:$PWD/hostenv/bin/python,gpu_t4=docker:soma-torch-cpu:py311"
```

The check verdicts reproduce (101 PASS, 0 FAIL, 6 SKIP). The number of acknowledged and refused writes depends on the writer's timing and varies between runs.

### XSUB-ISA

A NumPy SGD session with a view of its parameters, an optimizer-style structure that references them, an alias, a reference cycle, two generators and a log file moves from an arm64 container to an emulated x86-64 container and back. Two uninterrupted controls run the same job on each ISA. Stored state is compared exactly at every boundary by the controller; per-step losses are compared bitwise with both controls, and the record states the first step at which they differ (matrix products use different SIMD kernels on each ISA).

```sh
docker build --platform linux/arm64 -t soma-port:arm64 -f experiments/docker/port.Dockerfile experiments/docker
docker build --platform linux/amd64 -t soma-port:amd64 -f experiments/docker/port.Dockerfile experiments/docker
python experiments/xsub_isa.py --outdir /tmp/isa --arm docker:soma-port:arm64 --x86 docker:soma-port:amd64@linux/amd64
```

With images built from `port.Dockerfile`, the run reproduces the recorded losses, final state and controls bitwise.

### E16: two cyclic sessions share one GPU

Two sessions alternate CPU phases (pool scoring with NumPy) and GPU phases (training a small CNN on an A100), offset by half a cycle. At every phase boundary the GPU holder moves to the CPU runtime and the other session then moves to the GPU through the in-place switch, so at most one GPU runtime is alive. Every move checks the step counters and a digest of the model parameters. The dedicated-GPU baseline is modelled from the measured phase durations, not run; `figures/share_model.py` scales the measured run to a working day.

The recorded run completed two rounds and both swaps (four moves, every state check equal). In the third round the phase of session A did not return, so the record ends with `error: "RuntimeError: phase 2 of A did not return"`, and its end-to-end `summary` (a saving of -29.9%) includes that failed round; it is not used. The reported numbers come from `figures/share_model.py`, which measures only the completed rounds. The earlier attempts `e16-micro` (a move refused with HTTP 409, the source kept) and `e16-micro-3` (a runtime failed to start before any move) are listed in `runs_meta.jsonl`; their records are not shipped.

```sh
python experiments/e16_cyclic.py --local --phase-s 3 --cycles 2 --out /tmp/e16   # dry run
```

### Escalation: OOM-triggered moves

Two training jobs grow their memory demand partway through (GPT-2 small by sequence length, ViT-B/16 by input resolution), genuinely hit a CUDA out-of-memory error on the GPU they no longer fit, discard the failed attempt, and move to a larger GPU through the in-place switch; the first execute afterwards checks the step count and the optimizer state. `calibrate.py` measures the memory thresholds on each GPU with the same workload source. `rederive.py escalation` tallies every move across the four run files (a move is verified when the restore reports `restored`, the optimizer is intact and its step equals the completed steps) and prints the calibration table.

The recorded invocations were not logged; from the record fields they were the following, shown here as reruns into a scratch directory:

```sh
python experiments/escalation/calibrate.py --workload gpt2 --profile gpu_t4 --out /tmp/soma-rerun/escalation/escalation_calibration.jsonl   # also gpu_a100_40; vit on gpu_t4, gpu_l4, gpu_a100_40
python experiments/escalation/run_escalation.py --exp gpt2 --fractions 0.1,0.5,0.9 --reps 1 --total-steps 1200 --out /tmp/soma-rerun/escalation/escalation_runs.jsonl
python experiments/escalation/run_escalation.py --exp gpt2 --fractions 0.5 --total-steps 40 --reps 1 --out /tmp/soma-rerun/escalation/escalation_runs.jsonl
python experiments/escalation/run_escalation.py --exp vit --reps 2 --out /tmp/soma-rerun/escalation/escalation_runs.jsonl
python experiments/rederive.py --results /tmp/soma-rerun escalation
```

`escalation_runs_failed_sweep1.jsonl` and `escalation_runs_timeout_fail.jsonl` are earlier sweeps that ended in harness failures (an expired token, a cell timeout); their moves are counted because each move itself was recorded. `teardown.py` releases escalation projects; in the recorded runs its fallback after a refused delete used an internal endpoint of the platform, and the shipped version calls pause instead.

### E10: switch time against workspace size

A CPU-to-T4 switch was timed for workspaces of 1, 64 and 256 files on two research builds of the platform server, before and after workspace persistence was batched (labelled `server-build-1` and `server-build-2` in the records). The harness depends on server instrumentation and is not included; the records are, and `rederive.py e10` recomputes the medians.

## Cloud runs

The cloud harnesses drive a research build of the platform API, which is not part of this artifact, through `CLUSY_API_URL` (default `http://localhost:8010`) with a bearer key in `CLUSY_HARNESS_API_KEY`. The transactional controller (class `Api` in `src/handoff/controller.py`) needs only four runtime operations and one optional fallback, none of them specific to the platform: create a fresh, empty runtime from a name and an opaque profile label and return its identifier (`POST /projects`); execute code and return the status, the complete stdout and any error (`POST /projects/{id}/sandbox/execute`); delete a runtime, where 200, 204 and 404 all count as released (`DELETE /projects/{id}`); list runtimes by identifier and name, which crash recovery uses to find a destination that was created but not journaled (`GET /projects`); and, only when a delete fails, pause (`POST /projects/{id}/sandbox/pause`). Each runtime must behave like a Jupyter kernel: `__main__` persists between calls, executions run one at a time in arrival order (the drain barrier relies on this), and stdout must carry a base64 capsule, about 1.33 times the capsule size. It needs a writable working directory and `/tmp`, no network access, the same Python minor version and `dill` 0.3.9 on both ends, and every distribution the source imported installed on the destination, or the preflight refuses the move; validation forks the kernel when `os.fork` is allowed. The capsule, journal, lease and admission gate stay on the controller host, and no object store is involved. `experiments/localapi.py` is a complete local implementation of this contract. E13, E14, E16, the escalation harness and E11's concurrent-PATCH probe additionally use the platform's in-place switch (`PATCH /projects/{id}` with `runtimeProfile`), which is specific to the platform. E14's application-checkpoint arms also need `S3_BUCKET`, `S3_REGION`, AWS credentials, and runtimes with internet access. The escalation harness uses `CLUSY_API_URL` (default `http://localhost:8002`) with a user token in `CLUSY_CAMPAIGN_JWT` (or a file named by `CLUSY_CAMPAIGN_JWT_FILE`), and also calls `GET /projects/{id}`, `POST /projects/{id}/sandbox/resume` and `GET /projects/{id}/sandbox/status`.

## What the artifact cannot reproduce

- **Production statistics.** The session counts, checkpoint size distribution and workspace file counts in the abstract and in Motivation and Background come from read-only queries on the production database. Neither the queries' results nor the data are included.
- **E10.** The 302 s and 96 s switch times were measured on two research builds of the platform server, not on the production service, and rerunning them needs that server. The records are shipped for re-analysis only.
- **The recorded host-container substrate.** The recorded run used the platform's sandbox image; the artifact substitutes a public image, which reproduces the check verdicts but not the exact write counts.
- **Cloud measurements.** Timings, costs and GPU behaviour in E11 to E16 and the escalation runs depend on the platform API and on E2B and Modal. They can be re-analysed from the records but rerun only against an API that implements the contract above.
- **Regression tests.** The artifact ships 198 tests (`python -m pytest`). The paper's count of 198 refers to the research suite; this suite reaches the same number with 20 tests written for the artifact that cover the record guards, the local runtime contract, and paper numbers re-derived from the shipped records.

## Records

Every run of E11 to E16 and of the host-container move appended a provenance line to the `runs_meta.jsonl` next to its records: the exact command line, harness file hashes, the server build and the host. Several harness files changed after the runs and again for release (comments and output guards), so a recorded hash identifies the version that ran, not necessarily the shipped file.

Before release the records were edited as follows, and every analysis and figure was checked to give the same result before and after:

- Local paths were replaced by `<repo>`, `<home>` and `<tmp>`, the host name by `<host>`, private server commit identifiers by `server-build-1` to `server-build-3`, private branch names by `private-server-branch` and `private-harness-branch`, local Docker image identifiers by descriptive placeholders, and the price sources in E14 by plain descriptions.
- Runtime (project) identifiers were replaced one to one with random UUIDs, so every equality between them still holds.
- The names of the platform's own kernel helpers, which appear in namespace listings, were replaced one to one by `_platform_helper_01` to `_platform_helper_14`, and the fixture's workspace constant appears under its shipped name, `WORKSPACE_DIR`.
- The platform switch's internal step timeline was reduced to its phase totals: E10 keeps the per-phase times and drops the steps, and E13 and E14 keep three steps (before the source's destroy, the destroy, and after it) with the same sums (to 0.1 ms), so the E14 cost split can still be recomputed.

Otherwise all numeric content is unchanged. Record files hold every cohort a harness wrote, except E16, which writes one file per cohort and ships only `e16-micro-2.json`; the analyses select the paper's cohort by name. Harness commit identifiers refer to the private development history and serve only as run identifiers.
