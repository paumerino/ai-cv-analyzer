"""End-to-end ranking validation test across synthetic candidate tiers."""

from evaluation.benchmark import run_benchmark


def test_benchmark_orders_candidate_tiers_monotonically() -> None:
    rows, passed, total = run_benchmark()

    assert len(rows) == 9
    assert passed == total