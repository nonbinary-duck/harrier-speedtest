# Harrier OSS Benchmark
  
A self-contained benchmarking tool for measuring input token throughput of `microsoft/harrier-oss-v1-0.6b` (and other HuggingFace models) using a pool of arXiv papers.
  
## Features
  
- **Self-Contained**: Model and papers are prefetched during the Docker build.
- **HTML-Only**: Downloads content from `export.arxiv.org` without figures.
- **Token Chunking**: Automatically splits long papers into fixed-size token blocks to manage VRAM.
- **Batch Sweeping**: Experiment with multiple batch sizes in a single run.
- **Offline Ready**: No internet access required after the initial image build.
  
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
  
### Adjust Chunk Size & Overlap
Split papers into smaller or larger token windows, optionally with overlapping tokens:
```bash
docker compose run --rm -it harrier-speedtest \
  python speedtest.py --chunk-tokens 4096 --chunk-overlap-tokens 128
```
  
### Comprehensive Stress Test
```bash
docker compose run --rm -it harrier-speedtest \
  python speedtest.py \
    --chunk-tokens 8192 \
    --batch-sizes 1 2 4 \
    --warmup-runs 3 \
    --measure-runs 10 \
    --dtype bfloat16 \
    --report-chunk-stats
```
  
---
  
## Configuration Parameters
  
### Core Settings
- `--model-id`: HuggingFace model ID to benchmark (Default: `microsoft/harrier-oss-v1-0.6b`).
- `--batch-sizes`: Space-separated list of batch sizes to sweep (Default: `1`).
- `--max-length`: Tokenizer truncation limit. `0` disables truncation (Default: `0`).
- `--dtype`: Compute type (`auto`, `float16`, `bfloat16`, `float32`).
- `--device`: Target device (`auto`, `cuda`, or `cpu`).
- `--seed`: Random seed for determinism (Default: `42`).
  
### Token Chunking
- `--chunk-tokens`: Size of each chunk in tokens (Default: `8192`).
- `--chunk-overlap-tokens`: Number of tokens to overlap between consecutive chunks (Default: `0`).
- `--max-chunks-per-paper`: Limit number of chunks processed per paper for speed (Default: `0` / no limit).

### Benchmarking Methodology
- `--warmup-runs`: Number of non-timed passes to stabilize GPU state (Default: `2`).
- `--measure-runs`: Number of timed passes to average (Default: `5`).
- `--repeat-count`: Number of forward passes per measured run to reduce noise (Default: `3`).
- `--report-chunk-stats`: Flag to print detailed throughput statistics for every individual chunk.
  
---
  
## Output Metrics
  
The app provides detailed statistics for every paper and batch size:
- **Environment**: Python/PyTorch versions, GPU info, Compute Capability, and Model Dtype.
- **Paper Stats**: Total token count, number of chunks generated, and chunk size configuration.
- **Throughput**: Measured in **tokens/second** aggregated across papers and summarized for the batch sweep.
- **Chunk Statistics**: If `--report-chunk-stats` is used, prints token count and TPS for every processed chunk.
  
---
  
## Notes
- **VRAM Limits**: Large `--batch-sizes` combined with large `--chunk-tokens` (e.g., 8192) may exceed the memory of smaller GPUs. If you encounter OOM (Out of Memory) errors, reduce the batch size or chunk length.
- **Aggregation**: Paper throughput is calculated correctly by summing the total tokens processed across all chunks and dividing by the total estimated time, rather than naively averaging TPS values.
