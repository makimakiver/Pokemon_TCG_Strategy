# RL image: the engine base (linux/amd64 so cg/libcg.so loads) + numpy/torch.
# Build from the repo root so the cg/ engine and agents/ are in context:
#   docker build --platform=linux/amd64 -f rl/Dockerfile -t cabt-rl .
# Run (mount the repo for data/*.json which .dockerignore excludes from COPY):
#   docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app cabt-rl \
#     python -m rl.smoke.smoke_test
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app
COPY rl/requirements.txt /app/rl/requirements.txt
RUN pip install --no-cache-dir -r /app/rl/requirements.txt
COPY . /app

# Default: run the P0 smoke test. Override with any `python -m rl.<module>`.
ENTRYPOINT ["python"]
CMD ["-m", "rl.smoke.smoke_test"]
