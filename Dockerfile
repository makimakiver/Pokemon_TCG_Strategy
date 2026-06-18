# Linux x86-64 image so cg/libcg.so (ELF x86-64) loads. On Apple Silicon this
# runs under emulation via --platform=linux/amd64.
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app
COPY . /app

ENTRYPOINT ["python", "runner.py"]
