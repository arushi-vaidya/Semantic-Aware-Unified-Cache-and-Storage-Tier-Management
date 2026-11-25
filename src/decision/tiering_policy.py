from __future__ import annotations

from dataclasses import dataclass

from src.storage.tier import Tier


@dataclass(frozen=True)
class TieringChoice:
    tier: Tier
    reason: str


class IndependentTieringPolicy:
    """Frequency/age tier baseline that does not consume predictions or semantics."""

    def __init__(
        self,
        hot_access_count: int = 10,
        hot_age_operations: int = 20,
        warm_access_count: int = 3,
        archive_age_operations: int = 500,
    ):
        if min(hot_access_count, hot_age_operations, warm_access_count, archive_age_operations) < 0:
            raise ValueError("tiering thresholds must be non-negative")
        self.hot_access_count = hot_access_count
        self.hot_age_operations = hot_age_operations
        self.warm_access_count = warm_access_count
        self.archive_age_operations = archive_age_operations

    def choose(
        self,
        access_count: int,
        recent_access_count: int,
        operations_since_last_access: int,
        feasible_tiers: list[Tier],
    ) -> TieringChoice:
        if min(access_count, recent_access_count, operations_since_last_access) < 0:
            raise ValueError("access statistics must be non-negative")
        if not feasible_tiers:
            raise RuntimeError("no feasible tier for independent tiering policy")
        if recent_access_count >= self.hot_access_count and operations_since_last_access <= self.hot_age_operations:
            desired, reason = Tier.CACHE, "recent frequency meets configured hot threshold"
        elif access_count >= self.warm_access_count and operations_since_last_access < self.archive_age_operations:
            desired, reason = Tier.SSD, "historical frequency meets configured warm threshold"
        elif operations_since_last_access >= self.archive_age_operations:
            desired, reason = Tier.ARCHIVE, "age meets configured archive threshold"
        else:
            desired, reason = Tier.HDD, "frequency and age indicate cold placement"

        ordered = list(Tier)
        desired_index = ordered.index(desired)
        slower = [tier for tier in ordered[desired_index:] if tier in feasible_tiers]
        if slower:
            return TieringChoice(slower[0], reason if slower[0] == desired else f"{reason}; desired tier lacked capacity")
        faster = [tier for tier in reversed(ordered[:desired_index]) if tier in feasible_tiers]
        return TieringChoice(faster[0], f"{reason}; slower feasible tiers lacked capacity")