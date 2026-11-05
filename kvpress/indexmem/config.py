from __future__ import annotations

from dataclasses import dataclass

HEAD_BUDGETS = ("mass", "uniform", "static", "shuffle")
INFERENCE_MODES = ("evict", "mask")


@dataclass
class IndexMemConfig:
    cache_budget: int = 2048
    sink_size: int = 4
    window_size: int = 128
    cmp_slots: int = 64
    head_budget: str = "mass"
    head_budget_table: str = ""
    min_head_budget: int = 512
    cache_budget_ratio: float | None = None
    decode_batch: int = 1
    inference_mode: str = "evict"

    def __post_init__(self):
        if self.head_budget not in HEAD_BUDGETS:
            raise ValueError(f"head_budget must be one of {HEAD_BUDGETS}, got {self.head_budget!r}")
        if self.inference_mode not in INFERENCE_MODES:
            raise ValueError(f"inference_mode must be one of {INFERENCE_MODES}, got {self.inference_mode!r}")
        if self.head_budget == "static" and not self.head_budget_table:
            raise ValueError(
                "head_budget='static' needs head_budget_table, the path written by scripts.fit_head_budgets"
            )
        for name in ("sink_size", "window_size", "cmp_slots", "min_head_budget"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        # The evictable entries are what remains once the centroid slots and the always-attended
        # regions are funded out of the same budget. Without at least one of them the scorer has
        # nothing to rank, and the eviction layout would address negative slots.
        reserved = self.sink_size + self.window_size + self.cmp_slots
        if self.cache_budget_ratio is None and self.cache_budget <= reserved:
            raise ValueError(
                f"cache_budget={self.cache_budget} leaves no evictable entries: "
                f"sink_size + window_size + cmp_slots = {reserved}"
            )
