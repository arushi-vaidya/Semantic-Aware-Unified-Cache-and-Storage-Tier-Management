from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class Operation:
    operation: str
    object_id: str
    size_bytes: int = 0


class WorkloadGenerator:
    def __init__(self, seed: int | None = None):
        self.random = random.Random(seed)

    def generate(self, workload_type: str, operations: int, object_count: int = 20, read_ratio: float = 0.8, object_size: int = 1024) -> list[Operation]:
        if operations < 0 or object_count <= 0 or not 0 <= read_ratio <= 1 or object_size < 0:
            raise ValueError("Invalid workload parameters")
        objects = [f"object_{index:04d}" for index in range(object_count)]
        if workload_type == "sequential":
            return [Operation("write", objects[index], object_size) if index < object_count else Operation("read", objects[index % object_count]) for index in range(operations)]
        result: list[Operation] = []
        for index in range(operations):
            if index < object_count:
                result.append(Operation("write", objects[index], object_size))
                continue
            if workload_type == "hot":
                object_id = self.random.choice(objects[:max(1, object_count // 5)])
            elif workload_type == "uniform":
                object_id = self.random.choice(objects)
            elif workload_type == "mixed":
                object_id = self.random.choice(objects)
            else:
                raise ValueError(f"Unknown workload type: {workload_type}")
            operation = "read" if workload_type != "mixed" or self.random.random() < read_ratio else "write"
            result.append(Operation(operation, object_id, object_size))
        return result