# NumPy-only runtime for the arm64 <-> x86-64 move (XSUB-ISA).
# Build once per platform:
#   docker build --platform linux/arm64 -t soma-port:arm64 -f experiments/docker/port.Dockerfile experiments/docker
#   docker build --platform linux/amd64 -t soma-port:amd64 -f experiments/docker/port.Dockerfile experiments/docker
FROM python:3.11.16-slim
RUN pip install --no-cache-dir numpy==2.4.6 dill==0.3.9
WORKDIR /w
