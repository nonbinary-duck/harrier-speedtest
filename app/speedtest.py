import argparse
import gc
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    DEFAULT_PAPER_IDS,
    batched_input,
    clear_device_cache,
    dtype_to_string,
    extract_article_text,
    format_stats,
    get_environment_info,
    maybe_sync,
    paper_cache_path,
    resolve_dtype,
    set_determinism,
    summary_stats,
    tokenize_and_chunk,
    wall_time_seconds,
)


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark input token throughput on cached arXiv HTML.")
    p.add_argument("--model-id", type=str, default="microsoft/harrier-oss-v1-0.6b")
    p.add_argument("--cache-dir", type=str, default="/opt/cache")
    p.add_argument("--paper-ids", nargs="+", default=DEFAULT_PAPER_IDS)

    p.add_argument(
        "--max-length",
        type=int,
        default=0,
        help="Tokenizer truncation length. 0 disables truncation (recommended with token chunking).",
    )

    p.add_argument("--repeat-count", type=int, default=3)
    p.add_argument("--warmup-runs", type=int, default=2)
    p.add_argument("--measure-runs", type=int, default=5)

    p.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1],
        help="Sweep these batch sizes in one run (e.g. --batch-sizes 1 2 4 8).",
    )

    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)

    # Token chunking (replaces chapter chunking)
    p.add_argument(
        "--chunk-tokens",
        type=int,
        default=8192,
        help="Chunk size in tokens (default: 8192).",
    )
    p.add_argument(
        "--chunk-overlap-tokens",
        type=int,
        default=0,
        help="Overlap between consecutive chunks in tokens (default: 0).",
    )
    p.add_argument(
        "--max-chunks-per-paper",
        type=int,
        default=0,
        help="Limit number of chunks per paper. 0 means no limit.",
    )

    p.add_argument("--report-chunk-stats", action="store_true")
    return p.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(model_id: str, cache_dir: str, device: torch.device, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True,
        trust_remote_code=True,
        dtype=dtype if device.type == "cuda" else torch.float32,
    )
    model.eval()
    model.to(device)
    return model, tokenizer


def make_tensors_from_chunk(tokenizer, chunk_ids: List[int], device: torch.device, batch_size: int):
    # add_special_tokens=False used during initial tokenization; keep it consistent.
    input_ids = torch.tensor([chunk_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    token_count_single = int(input_ids.shape[1])
    input_ids, attention_mask = batched_input(input_ids, attention_mask, batch_size)
    effective_token_count = int(input_ids.numel())
    return input_ids, attention_mask, token_count_single, effective_token_count


@torch.inference_mode()
def run_forward(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    return model(input_ids=input_ids, attention_mask=attention_mask)


def benchmark_input(model, input_ids, attention_mask, effective_token_count, repeat_count, warmup_runs, measure_runs, device):
    for _ in range(warmup_runs):
        for _ in range(repeat_count):
            _ = run_forward(model, input_ids, attention_mask)
        maybe_sync(device)

    throughputs = []
    latencies = []
    for _ in range(measure_runs):
        maybe_sync(device)
        start = wall_time_seconds()

        for _ in range(repeat_count):
            _ = run_forward(model, input_ids, attention_mask)

        maybe_sync(device)
        end = wall_time_seconds()

        elapsed = end - start
        total_tokens = effective_token_count * repeat_count
        throughputs.append(total_tokens / elapsed if elapsed > 0 else 0.0)
        latencies.append(elapsed)

    return {
        "throughput_stats": summary_stats(throughputs),
        "latency_stats": summary_stats(latencies),
        "throughput_values": throughputs,
        "latency_values": latencies,
    }


def print_environment(model_id: str, selected_dtype: torch.dtype, device: torch.device):
    env = get_environment_info()
    print("Environment:")
    print(f"- Python: {env['python']}")
    print(f"- PyTorch: {env['pytorch']}")
    print(f"- CUDA built: {env['cuda_built']}")
    print(f"- CUDA available: {env['cuda_available']}")
    print(f"- GPU: {env['gpu_name']}")
    print(f"- GPU total memory (GB): {env['gpu_total_memory_gb']}")
    print(f"- Compute capability: {env['gpu_compute_capability']}")
    print(f"- Device used: {device}")
    print(f"- Model: {model_id}")
    print(f"- Dtype: {dtype_to_string(selected_dtype)}")
    print("")


def main():
    args = parse_args()
    set_determinism(args.seed)

    if args.max_length < 0:
        raise ValueError("--max-length must be >= 0")
    if args.chunk_tokens <= 0:
        raise ValueError("--chunk-tokens must be > 0")
    if args.chunk_overlap_tokens < 0:
        raise ValueError("--chunk-overlap-tokens must be >= 0")
    if args.max_chunks_per_paper < 0:
        raise ValueError("--max-chunks-per-paper must be >= 0")
    if any(b <= 0 for b in args.batch_sizes):
        raise ValueError("--batch-sizes must all be > 0")

    device = select_device(args.device)
    selected_dtype = resolve_dtype(args.dtype, device)
    cache_dir = Path(args.cache_dir)
    hf_cache = str(cache_dir / "huggingface")

    print_environment(args.model_id, selected_dtype, device)

    model, tokenizer = load_model_and_tokenizer(args.model_id, hf_cache, device, selected_dtype)

    # Benchmark results: per batch size summary
    batchsize_means: Dict[int, List[float]] = {b: [] for b in args.batch_sizes}

    for paper_id in args.paper_ids:
        html_path = paper_cache_path(cache_dir, paper_id)
        if not html_path.exists():
            raise FileNotFoundError(f"Missing cached paper: {html_path}")

        clear_device_cache(device)

        text = extract_article_text(html_path)

        # Optional tokenizer truncation before chunking (generally leave as 0)
        if args.max_length > 0:
            # truncation at token level by re-encoding with truncation
            enc = tokenizer(text, return_attention_mask=False, truncation=True, max_length=args.max_length, add_special_tokens=False)
            ids = enc["input_ids"]
            # Convert back into "chunks" without re-tokenization logic
            total_tokens = len(ids)
            chunks = []
            start = 0
            idx = 0
            step = args.chunk_tokens - args.chunk_overlap_tokens
            while start < total_tokens:
                end = min(start + args.chunk_tokens, total_tokens)
                chunk_ids = ids[start:end]
                if not chunk_ids:
                    break
                chunks.append({"chunk_index": idx, "chunk_id": f"tok_{start}_{end}", "token_count": len(chunk_ids), "input_ids_list": chunk_ids})
                idx += 1
                start += step
        else:
            chunks, total_tokens = tokenize_and_chunk(
                tokenizer=tokenizer,
                text=text,
                chunk_tokens=args.chunk_tokens,
                overlap_tokens=args.chunk_overlap_tokens,
            )

        if args.max_chunks_per_paper > 0:
            chunks = chunks[: args.max_chunks_per_paper]

        print(f"Paper {paper_id}: total_tokens={total_tokens}, chunks={len(chunks)}, chunk_tokens={args.chunk_tokens}, overlap={args.chunk_overlap_tokens}")

        for batch_size in args.batch_sizes:
            paper_tokens_processed = 0
            paper_elapsed = 0.0

            # Benchmark each chunk; aggregate as total_tokens / total_time (more correct than averaging TPS)
            for ch in chunks:
                input_ids, attention_mask, token_count_single, effective_token_count = make_tensors_from_chunk(
                    tokenizer, ch["input_ids_list"], device, batch_size
                )

                result = benchmark_input(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    effective_token_count=effective_token_count,
                    repeat_count=args.repeat_count,
                    warmup_runs=args.warmup_runs,
                    measure_runs=args.measure_runs,
                    device=device,
                )

                # Use mean latency for aggregation (sum of times, sum of tokens)
                mean_latency = result["latency_stats"]["mean"]
                tokens_this_chunk = effective_token_count * args.repeat_count  # per measured run group
                # But latency is per measured run group; throughput stats already computed across measure_runs.
                # For aggregation, approximate using mean throughput across measure runs:
                # tokens/sec for this chunk at this batch size:
                chunk_tps_mean = result["throughput_stats"]["mean"]

                # Convert back to time estimate for one group to combine:
                time_est = (tokens_this_chunk / chunk_tps_mean) if chunk_tps_mean > 0 else mean_latency

                paper_tokens_processed += tokens_this_chunk
                paper_elapsed += time_est

                if args.report_chunk_stats:
                    print(f"  bs={batch_size} chunk={ch['chunk_index']} tokens={token_count_single} "
                          f"tps={result['throughput_stats']['mean']:.2f}")

                # free between chunks
                del input_ids, attention_mask
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            paper_tps = (paper_tokens_processed / paper_elapsed) if paper_elapsed > 0 else 0.0
            batchsize_means[batch_size].append(paper_tps)

            print(f"- batch_size={batch_size}: approx_paper_throughput={paper_tps:.2f} tokens/sec")

        print("")

    print("Batch size sweep summary (mean of per-paper throughputs):")
    for batch_size in args.batch_sizes:
        stats = summary_stats(batchsize_means[batch_size])
        print(f"- batch_size={batch_size}: {format_stats(stats, 'tokens/sec')}")


if __name__ == "__main__":
    main()