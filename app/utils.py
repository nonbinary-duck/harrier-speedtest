import random
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
from bs4 import BeautifulSoup


DEFAULT_PAPER_IDS = [
    "1706.03762",
    "2604.07053",
    "2604.07157",
    "2604.07094",
    "2604.07088",
    "2604.07237",
    "2604.06541",
    "2604.06543",
    "2604.07019",
    "2604.06652",
    "2604.03522",
    "2604.06491",
    "2604.06701",
    "2604.06689",
    "2604.05669",
]

EXPORT_ARXIV_HTML_TEMPLATE = "https://export.arxiv.org/html/{paper_id}"


def set_determinism(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def paper_cache_path(cache_dir: Path, paper_id: str) -> Path:
    return cache_dir / "arxiv" / f"{paper_id}.html"


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def wall_time_seconds() -> float:
    return time.perf_counter()


def summary_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "stddev": 0.0}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def format_stats(stats: Dict[str, float], unit: str = "") -> str:
    suffix = f" {unit}" if unit else ""
    return (
        f"mean={stats['mean']:.2f}{suffix}, "
        f"median={stats['median']:.2f}{suffix}, "
        f"min={stats['min']:.2f}{suffix}, "
        f"max={stats['max']:.2f}{suffix}, "
        f"stddev={stats['stddev']:.2f}{suffix}"
    )


def dtype_to_string(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    return str(dtype)


def resolve_dtype(dtype_str: str, device: torch.device) -> torch.dtype:
    value = dtype_str.lower()
    if value == "auto":
        if device.type == "cuda":
            major, _ = torch.cuda.get_device_capability(device)
            return torch.bfloat16 if major >= 8 else torch.float16
        return torch.float32
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if value not in mapping:
        raise ValueError("Unsupported dtype. Use auto, float32, float16, or bfloat16.")
    return mapping[value]


def get_environment_info() -> Dict[str, str]:
    import sys

    info = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "pytorch": torch.__version__,
        "cuda_built": torch.version.cuda or "N/A",
        "cuda_available": str(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_gb": f"{props.total_memory / (1024**3):.2f}",
                "gpu_compute_capability": f"{props.major}.{props.minor}",
                "device_count": str(torch.cuda.device_count()),
            }
        )
    else:
        info.update(
            {
                "gpu_name": "CPU",
                "gpu_total_memory_gb": "N/A",
                "gpu_compute_capability": "N/A",
                "device_count": "0",
            }
        )
    return info


def download_arxiv_html(
    paper_id: str,
    dest_path: Path,
    timeout: int = 60,
    session: Optional[requests.Session] = None,
) -> Path:
    ensure_dir(dest_path.parent)
    url = EXPORT_ARXIV_HTML_TEMPLATE.format(paper_id=paper_id)
    sess = session or requests.Session()
    headers = {"User-Agent": "harrier-speedtest/1.0 (build-prefetch; rate-limited)"}
    response = sess.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()
    dest_path.write_text(response.text, encoding="utf-8")
    return dest_path


def load_html(path: Path) -> BeautifulSoup:
    if not path.exists():
        raise FileNotFoundError(f"Missing HTML file: {path}")
    html = path.read_text(encoding="utf-8", errors="replace")
    return BeautifulSoup(html, "lxml")


def get_article_node(path: Path):
    soup = load_html(path)
    article = soup.find("article", class_="ltx_document")
    if article is None:
        raise ValueError(f"Could not find <article class='ltx_document'> in {path}")
    return article


def extract_article_text(path: Path) -> str:
    article = get_article_node(path)
    text = article.get_text(separator="\n", strip=True)
    text = "\n".join([ln.strip() for ln in text.splitlines() if ln.strip()])
    if not text:
        raise ValueError(f"Extracted article text is empty for {path}")
    return text


def tokenize_and_chunk(
    tokenizer,
    text: str,
    chunk_tokens: int,
    overlap_tokens: int = 0,
) -> Tuple[List[Dict], int]:
    """
    Returns: (chunks, total_tokens)
    Each chunk contains token IDs (1D list[int]) so we never re-tokenize per chunk.
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be > 0")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0")
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be < chunk_tokens")

    enc = tokenizer(text, return_attention_mask=False, add_special_tokens=False)
    ids: List[int] = enc["input_ids"]
    total = len(ids)

    chunks: List[Dict] = []
    start = 0
    idx = 0
    step = chunk_tokens - overlap_tokens

    while start < total:
        end = min(start + chunk_tokens, total)
        chunk_ids = ids[start:end]
        if len(chunk_ids) == 0:
            break
        chunks.append(
            {
                "chunk_index": idx,
                "chunk_id": f"tok_{start}_{end}",
                "token_start": start,
                "token_end": end,
                "token_count": len(chunk_ids),
                "input_ids_list": chunk_ids,
            }
        )
        idx += 1
        start += step

    return chunks, total


def batched_input(input_ids: torch.Tensor, attention_mask: torch.Tensor, batch_size: int):
    if batch_size == 1:
        return input_ids, attention_mask
    return input_ids.repeat(batch_size, 1), attention_mask.repeat(batch_size, 1)