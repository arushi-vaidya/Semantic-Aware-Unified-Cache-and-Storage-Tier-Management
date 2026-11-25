from __future__ import annotations

from dataclasses import dataclass

from src.storage.storage_manager import StorageManager
from src.storage.tier import Tier
from src.decision.scoring import TierProperties


@dataclass(frozen=True)
class CapacitySnapshot:
    used_bytes: dict[Tier, int]
    capacity_bytes: dict[Tier, int | None]

    def projected_utilization(self, tier: Tier, object_size_bytes: int, current_tier: Tier) -> float:
        capacity = self.capacity_bytes[tier]
        if capacity is None or capacity == 0:
            return 0.0
        projected = self.used_bytes[tier]
        if tier == current_tier:
            return min(1.0, projected / capacity)
        return min(1.0, (projected + object_size_bytes) / capacity)

    def feasible(self, tier: Tier, object_size_bytes: int, current_tier: Tier) -> bool:
        capacity = self.capacity_bytes[tier]
        if capacity is None:
            return True
        if tier == current_tier:
            return self.used_bytes[tier] <= capacity
        return self.used_bytes[tier] + object_size_bytes <= capacity


def capacity_snapshot(
    storage: StorageManager,
    properties: dict[Tier, TierProperties],
) -> CapacitySnapshot:
    used = {
        tier: sum(storage.path(object_id, tier).stat().st_size for object_id in storage.list_objects(tier))
        for tier in Tier
    }
    capacities = {tier: properties[tier].capacity_bytes for tier in Tier}
    return CapacitySnapshot(used, capacities)


def feasible_tiers(snapshot: CapacitySnapshot, object_size_bytes: int, current_tier: Tier) -> list[Tier]:
    return [tier for tier in Tier if snapshot.feasible(tier, object_size_bytes, current_tier)]