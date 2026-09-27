from src.workload.generator import WorkloadGenerator


def test_seed_is_reproducible():
    first = WorkloadGenerator(42).generate("hot", 25, object_count=10)
    second = WorkloadGenerator(42).generate("hot", 25, object_count=10)
    assert first == second


def test_operation_count_and_mixed_ratio_bounds():
    operations = WorkloadGenerator(1).generate("mixed", 100, object_count=5, read_ratio=0.7)
    assert len(operations) == 100
    assert {operation.operation for operation in operations} == {"read", "write"}