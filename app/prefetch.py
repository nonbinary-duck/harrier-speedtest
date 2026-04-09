import argparse
import time
from pathlib import Path

import requests
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

from utils import DEFAULT_PAPER_IDS, download_arxiv_html, ensure_dir, paper_cache_path


def parse_args():
    parser = argparse.ArgumentParser(description="Prefetch model and arXiv HTML assets into local cache.")
    parser.add_argument("--model-id", type=str, default="microsoft/harrier-oss-v1-0.6b")
    parser.add_argument("--cache-dir", type=str, default="/opt/cache")
    parser.add_argument("--paper-ids", nargs="+", default=DEFAULT_PAPER_IDS)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--force-redownload", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    hf_cache = cache_dir / "huggingface"
    arxiv_cache = cache_dir / "arxiv"

    ensure_dir(hf_cache)
    ensure_dir(arxiv_cache)

    print(f"Prefetching model: {args.model_id}")
    snapshot_download(
        repo_id=args.model_id,
        cache_dir=str(hf_cache),
        local_dir=None,
        local_dir_use_symlinks=False,
        resume_download=False,
    )

    AutoConfig.from_pretrained(
        args.model_id,
        cache_dir=str(hf_cache),
        local_files_only=False,
        trust_remote_code=True,
    )
    AutoTokenizer.from_pretrained(
        args.model_id,
        cache_dir=str(hf_cache),
        local_files_only=False,
        trust_remote_code=True,
    )

    session = requests.Session()

    requested = len(args.paper_ids)
    downloaded = 0
    skipped = 0
    cycle_downloads = 0

    start = time.perf_counter()

    for i, paper_id in enumerate(args.paper_ids):
        dest = paper_cache_path(cache_dir, paper_id)
        if dest.exists() and not args.force_redownload:
            skipped += 1
            print(f"[{i+1}/{requested}] cached: {paper_id}")
            continue

        cycle_start = time.perf_counter()
        print(f"[{i+1}/{requested}] downloading: {paper_id} -> {dest}")
        download_arxiv_html(
            paper_id=paper_id,
            dest_path=dest,
            timeout=args.timeout,
            session=session,
        )
        downloaded += 1
        cycle_downloads += 1

        elapsed = time.perf_counter() - cycle_start
        min_interval = 1.0 / 3.0
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        if cycle_downloads == 4:
            print("Resting 1 second after 4 article downloads...")
            time.sleep(1.0)
            cycle_downloads = 0

    total_elapsed = time.perf_counter() - start
    effective_pps = downloaded / total_elapsed if total_elapsed > 0 else 0.0

    print("")
    print("Prefetch summary:")
    print(f"- Requested papers: {requested}")
    print(f"- Downloaded papers: {downloaded}")
    print(f"- Skipped cached papers: {skipped}")
    print(f"- Elapsed download time: {total_elapsed:.2f} sec")
    print(f"- Effective papers/sec: {effective_pps:.2f}")


if __name__ == "__main__":
    main()