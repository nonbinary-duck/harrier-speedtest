# Harrier OSS Benchmark

A self-contained benchmarking tool for measuring input token throughput of `microsoft/harrier-oss-v1-0.6b` using a pool of 16 arXiv papers.

## Features

- **Self-Contained**: Model and papers are prefetched during the Docker build.
- **HTML-Only**: Downloads content from `export.arxiv.org` without figures.
- **Rate-Limited**: Prefetcher respects a 3-article/sec limit with a 1s rest every 4 downloads.
- **Token Chunking**: Automatically splits long papers into fixed-size token blocks (default: 8192).
- **Batch Sweeping**: Experiment with multiple batch sizes in a single run.
- **Offline Ready**: No internet access required after the initial image build.

---

## Default Paper Pool

The benchmark uses 16 papers by default, including `1706.03762`, `2604.07053`, and 14 other recently added IDs.

---

## Quick Start

### 1. Build
Downloads the model weights and all paper HTML files into the image.
```bash
docker compose build
```

### 2. Run
Runs the benchmark with default settings (8192 token chunks, batch size 1).
```bash
docker compose run --rm -it harrier-speedtest
```

---

## Advanced Usage

### Experiment with Batch Sizes
To test how throughput scales with batch size, provide a list of values:
```bash
docker compose run --rm -it harrier-speedtest \
  python speedtest.py --batch-sizes 1 2 4 8
```

### Adjust Chunk Size
Split papers into smaller or larger token windows:
```bash
docker compose run --rm -it harrier-speedtest \
  python speedtest.py --chunk-tokens 4096
```

### Comprehensive Stress Test
```bash
docker compose run --rm -it harrier-speedtest \
  python speedtest.py \
    --chunk-tokens 8192 \
    --batch-sizes 1 2 4 \
    --warmup-runs 3 \
    --measure-runs 10 \
    --dtype bfloat16
```

---

## Configuration Parameters

### Core Settings
- `--batch-sizes`: Space-separated list of batch sizes to sweep (Default: `1`).
- `--max-length`: Tokenizer truncation limit. `0` disables truncation (Default: `0`).
- `--dtype`: Compute type (`auto`, `float16`, `bfloat16`, `float32`).
- `--device`: Target device (`cuda` or `cpu`).

### Token Chunking
- `--chunk-tokens`: Size of each chunk in tokens (Default: `8192`).
- `--chunk-overlap-tokens`: Number of tokens to overlap between chunks (Default: `0`).
- `--max-chunks-per-paper`: Limit number of chunks processed per paper for speed (Default: `0` / no limit).

### Benchmarking Methodology
- `--warmup-runs`: Number of non-timed passes to stabilize GPU state (Default: `2`).
- `--measure-runs`: Number of timed passes to average (Default: `5`).
- `--repeat-count`: Number of forward passes per measured run to reduce noise (Default: `3`).

---

## Output Metrics

The app provides detailed statistics for every paper and batch size:
- **Environment**: GPU info, CUDA version, and Model Dtype.
- **Paper Stats**: Total token count and number of chunks generated.
- **Throughput**: Measured in **tokens/second** (mean, median, min, max, stddev).
- **Latency**: Time per forward pass group in seconds.
- **VRAM**: Peak GPU memory utilization per paper.

---

## Notes
- **VRAM Limits**: Large `--batch-sizes` combined with large `--chunk-tokens` (e.g., 8192) may exceed the memory of smaller GPUs like the RTX 3070. If you encounter OOM errors, reduce the batch size or chunk length.
- **Aggregation**: Throughput is calculated by summing total tokens processed across all chunks and dividing by the total elapsed time.
