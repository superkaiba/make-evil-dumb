"""File-based LLM response cache with LRU in-memory layer.

Keyed by (prompt_hash, params_hash). Disk storage is binned by hash prefix
to avoid huge directories. In-memory OrderedDict provides fast repeated lookups
with configurable memory cap and LRU eviction.

Thread-safe via filelock for disk writes.
"""

import json
import logging
import os
import sys
import tempfile
from collections import OrderedDict, deque
from itertools import chain
from pathlib import Path

import filelock

from explore_persona_space.llm.models import LLMCache, LLMParams, LLMResponse, Prompt

logger = logging.getLogger(__name__)


# ── Size estimation ─────────────────────────────────────────────────────────


def _total_size_mb(o) -> float:
    """Approximate in-memory size of an object tree, in MB."""

    def dict_handler(d):
        return chain.from_iterable(d.items())

    handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    seen: set[int] = set()
    default_size = sys.getsizeof(0)

    def sizeof(obj):
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        s = sys.getsizeof(obj, default_size)
        for typ, handler in handlers.items():
            if isinstance(obj, typ):
                for elem in handler(obj):
                    s += sizeof(elem)
                break
        return s

    return sizeof(o) / 1024 / 1024


# ── File I/O helpers ────────────────────────────────────────────────────────


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _save_json(path: Path, data: dict) -> None:
    """Atomic write via temp file + rename."""
    temp_dir = path.parent
    with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, mode="w") as tmp:
        tmp_path = tmp.name
        try:
            json.dump(data, tmp)
        except Exception:
            tmp.close()
            os.remove(tmp_path)
            raise
    os.replace(src=tmp_path, dst=path)


# ── FileCache ───────────────────────────────────────────────────────────────


class FileCache:
    """File-based LLM response cache with an LRU in-memory layer.

    Disk layout: ``cache_dir / <params_hash> / bin<N>.json``
    Each bin file is a JSON dict mapping prompt_hash -> serialized LLMCache.

    The in-memory layer (OrderedDict keyed by bin file path) avoids redundant
    disk reads. LRU eviction keeps memory usage under ``max_mem_mb``.

    Usage::

        cache = FileCache(Path("eval_results/my_run/llm_cache"))

        # Check cache
        cached = cache.load(prompt, params)
        if cached is not None:
            return cached.responses

        # After API call, save
        cache.save(prompt, params, responses)
    """

    def __init__(self, cache_dir: Path, num_bins: int = 20, max_mem_mb: float = 5_000):
        self.cache_dir = Path(cache_dir)
        self.num_bins = num_bins
        self.max_mem_mb = max_mem_mb

        # LRU in-memory cache: bin_path -> {prompt_hash -> json_str}
        self._mem: OrderedDict[Path, dict] = OrderedDict()
        self._sizes: dict[Path, float] = {}
        self._total_mb = 0.0

    # ── Cache key computation ───────────────────────────────────────────

    @staticmethod
    def _bin_number(hash_hex: str, num_bins: int) -> int:
        return int(hash_hex, 16) % num_bins

    def _cache_file(self, prompt: Prompt, params: LLMParams) -> tuple[Path, str]:
        """Return (bin_file_path, prompt_hash) for a given prompt+params."""
        prompt_hash = prompt.model_hash()
        bin_num = self._bin_number(prompt_hash, self.num_bins)
        cache_dir = self.cache_dir / params.model_hash()
        return cache_dir / f"bin{bin_num}.json", prompt_hash

    # ── LRU memory management ──────────────────────────────────────────

    def _evict_lru(self) -> None:
        lru_key = next(iter(self._mem))
        del self._mem[lru_key]
        self._total_mb -= self._sizes.pop(lru_key)

    def _mem_store(self, bin_path: Path, contents: dict) -> bool:
        """Store a bin in memory, evicting LRU entries if needed.

        Returns False if the entry exceeds the entire cache limit.
        """
        size_mb = _total_size_mb(contents)
        if self.max_mem_mb is not None and size_mb > self.max_mem_mb:
            return False

        # Remove old version first
        if bin_path in self._sizes:
            del self._mem[bin_path]
            self._total_mb -= self._sizes.pop(bin_path)

        # Evict until there's room
        if self.max_mem_mb is not None:
            while self._mem and self._total_mb + size_mb > self.max_mem_mb:
                self._evict_lru()

        self._mem[bin_path] = contents
        self._sizes[bin_path] = size_mb
        self._total_mb += size_mb
        return True

    def _mem_touch(self, bin_path: Path) -> None:
        """Mark a bin as recently used."""
        if bin_path in self._mem:
            self._mem.move_to_end(bin_path)

    # ── Public API ──────────────────────────────────────────────────────

    def load(self, prompt: Prompt, params: LLMParams) -> LLMCache | None:
        """Look up a cached response. Returns None on miss."""
        bin_path, prompt_hash = self._cache_file(prompt, params)
        if not bin_path.exists():
            return None

        # Check in-memory first
        if bin_path in self._mem and prompt_hash in self._mem[bin_path]:
            self._mem_touch(bin_path)
            data = self._mem[bin_path][prompt_hash]
            return LLMCache.model_validate_json(data)

        # Load from disk into memory
        with filelock.FileLock(str(bin_path) + ".lock"):
            contents = _load_json(bin_path)
            self._mem_store(bin_path, contents)

        data = contents.get(prompt_hash)
        return None if data is None else LLMCache.model_validate_json(data)

    def save(self, prompt: Prompt, params: LLMParams, responses: list[LLMResponse]) -> None:
        """Save responses to the cache (disk + memory)."""
        bin_path, prompt_hash = self._cache_file(prompt, params)
        bin_path.parent.mkdir(parents=True, exist_ok=True)

        entry = LLMCache(prompt=prompt, params=params, responses=responses)

        with filelock.FileLock(str(bin_path) + ".lock"):
            cache_data = _load_json(bin_path) if bin_path.exists() else {}
            cache_data[prompt_hash] = entry.model_dump_json()
            _save_json(bin_path, cache_data)

    def load_responses(
        self,
        prompt: Prompt,
        params: LLMParams,
        n: int,
        empty_threshold: float = 0.5,
    ) -> list[LLMResponse] | None:
        """Load cached responses with empty-completion validation.

        Returns None if cache miss, wrong count, or too many empty completions.
        """
        cached = self.load(prompt, params)
        if cached is None or cached.responses is None:
            return None

        responses = cached.responses
        if len(responses) != n:
            logger.warning("Cache has %d responses but expected %d", len(responses), n)
            return None

        n_empty = sum(1 for r in responses if r.completion == "")
        if n_empty / len(responses) > empty_threshold:
            logger.warning(
                "%.0f%% of cached responses are empty, treating as miss",
                100 * n_empty / len(responses),
            )
            return None

        return responses
