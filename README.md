# SOMA

SOMA (Session-Oriented Migration Abstraction) moves a live Python session, such as a Jupyter kernel, between heterogeneous runtimes: host processes, containers and microVMs, across operating systems and instruction sets, and between CPUs and GPUs. It captures the supported application state into a portable capsule, reconstructs and verifies it at the destination, and transfers execution authority transactionally, so a move either commits with exactly one authoritative runtime or aborts with the source still serving. This repository holds the research implementation of SOMA and the experiments behind the paper *Moving Live Python Sessions across Heterogeneous Runtimes*.

## Layout

```
src/
  handoff/            transactional, crash-recoverable migration controller
  capsule/            capsule manifest, storage sharing, optimizer reattachment
  clusy_boundary/     boundary-correctness checks on the moved state
  runmeta.py          provenance metadata written with every run
tests/                regression tests (198)
experiments/          harnesses and analyses for E11 to E16, XSUB and escalation
  localapi.py         local implementation of the runtime contract, for dry runs
  rederive.py         recomputes paper numbers that have no dedicated analyser
  docker/             Dockerfiles for the Docker experiments
results/              recorded runs, one directory per experiment
figures/              figure scripts; out/ holds the paper's figures
docs/EXPERIMENTS.md   map from each paper result to its command and record
```

## Quick start

Install (Python 3.11; dependency versions are pinned; all commands run from the repository root):

```sh
pip install -e '.[torch,figures,dev]' && export PYTHONPATH=src:.
```

Run the tests:

```sh
python -m pytest
```

Run one offline experiment, the E15 lifecycle on local processes (expected verdict: 101 PASS, 0 FAIL, 6 SKIP, OK):

```sh
python experiments/e15_combined.py --local
```

Regenerate the figures into `figures/out/`:

```sh
python figures/build_figures.py && python figures/share_model.py
```

## Reproducing the paper

[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) maps every result in the paper to its experiment, command and record.

- **Offline, on one machine:** the tests, the E11 to E16 dry runs, the two Docker experiments, and the re-analysis of the shipped records, which recomputes every reported number except the production statistics (not shipped).
- **Cloud:** rerunning E11 to E16 and the escalation runs needs CPU and GPU runtimes behind an API that implements the [runtime contract](docs/EXPERIMENTS.md#cloud-runs), which is not part of this repository. Reruns write outside `results/` ([how](docs/EXPERIMENTS.md#rerunning-without-touching-the-records)).
