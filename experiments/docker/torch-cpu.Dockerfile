# Linux container runtime for the host process <-> container lifecycle (XSUB host-container).
# It stands in for the sandbox image of the recorded run, which is not distributed.
#   docker build -t soma-torch-cpu:py311 -f experiments/docker/torch-cpu.Dockerfile experiments/docker
FROM python:3.11-slim
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy==2.1.2 dill==0.3.9 setuptools tqdm
WORKDIR /w
