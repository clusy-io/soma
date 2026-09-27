# SOMA

**Session-Oriented Migration Abstraction**

SOMA moves a live Python session, such as a Jupyter kernel, between heterogeneous runtimes: Firecracker microVMs, gVisor and Docker containers, and host processes, across operating systems, instruction sets, providers, and between CPUs and GPUs. It captures the application-visible state into a portable capsule, reconstructs it at the destination, and commits each move transactionally, so no acknowledged work is lost and a failed move leaves the source serving.

This repository accompanies the paper *Moving Live Python Sessions across Heterogeneous Runtimes*. The version of SOMA and the experiments described in the paper will be published here.

## Status

Code and experiment artifacts are being prepared for release.

## Citation

```bibtex
@misc{hasanov2026soma,
  title  = {Moving Live Python Sessions across Heterogeneous Runtimes},
  author = {Eldar Hasanov and Ju Lin and {Mohamed Fouzil Ali Syed Ali}},
  year   = {2026},
  note   = {Under review}
}
```

## Contact

[Clusy](https://clusy.io)
