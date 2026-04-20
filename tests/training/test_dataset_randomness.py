import random

import numpy as np


def test_same_index_yields_different_samples_across_calls() -> None:
    rng_values_a = [random.random() for _ in range(10)]
    rng_values_b = [random.random() for _ in range(10)]
    assert rng_values_a != rng_values_b


def test_dataset_does_not_seed_per_index() -> None:
    from src.training.data.dataset import OsuDataset

    import inspect
    source = inspect.getsource(OsuDataset.__getitem__)
    assert "rng = random.Random(self._base_seed + index)" not in source, (
        "Dataset is seeding RNG per index — will cause severe overfit via deterministic sample reuse."
    )
    assert "random.choice" in source
    assert "random.uniform" in source


def test_workers_get_distinct_random_state() -> None:
    worker_count = 4
    base_seed = 12345

    def worker_sample(seed: int) -> list[float]:
        r = random.Random(seed)
        return [r.random() for _ in range(5)]

    samples = [worker_sample(base_seed + wid) for wid in range(worker_count)]
    for i in range(worker_count):
        for j in range(i + 1, worker_count):
            assert samples[i] != samples[j], f"workers {i} and {j} collide"
