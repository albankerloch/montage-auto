"""Le suivi coût/latence : accumulation, remise à zéro, tarifs inconnus.

Chaque test construit son propre `UsageTracker` plutôt que de toucher
l'accumulateur global de `src.usage` : les tests ne doivent pas se polluer
entre eux via un état partagé.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.usage import UsageTracker  # noqa: E402


def test_records_accumulate_across_calls():
    t = UsageTracker()
    t.record(agent="AnnotatorAgent", model="claude-haiku-4-5-20251001",
              input_tokens=1000, output_tokens=200, latency_s=1.5)
    t.record(agent="AnnotatorAgent", model="claude-haiku-4-5-20251001",
              input_tokens=500, output_tokens=100, latency_s=0.5)

    agg = t.by_agent()["AnnotatorAgent"]
    assert agg["calls"] == 2
    assert agg["input_tokens"] == 1500
    assert agg["output_tokens"] == 300
    assert agg["latency_s"] == 2.0


def test_agents_are_aggregated_separately():
    t = UsageTracker()
    t.record(agent="AnnotatorAgent", model="claude-haiku-4-5-20251001",
              input_tokens=100, output_tokens=10, latency_s=1.0)
    t.record(agent="ComparatorAgent", model="claude-sonnet-4-6",
              input_tokens=200, output_tokens=20, latency_s=2.0)

    by_agent = t.by_agent()
    assert set(by_agent) == {"AnnotatorAgent", "ComparatorAgent"}
    assert by_agent["AnnotatorAgent"]["calls"] == 1
    assert by_agent["ComparatorAgent"]["calls"] == 1


def test_cost_uses_the_pricing_table():
    t = UsageTracker()
    t.record(agent="AnnotatorAgent", model="claude-haiku-4-5-20251001",
              input_tokens=1_000_000, output_tokens=1_000_000, latency_s=1.0)
    # 1.00 $/Mtok in + 5.00 $/Mtok out — cf. MODEL_PRICING_PER_MTOK, src/config.py
    assert abs(t.total_cost() - 6.00) < 1e-9


def test_unknown_model_costs_zero_not_a_guess():
    """Un tarif absent de la table n'est jamais deviné : $0, pas une extrapolation."""
    t = UsageTracker()
    t.record(agent="X", model="some-future-model", input_tokens=1_000_000,
              output_tokens=1_000_000, latency_s=1.0)
    assert t.total_cost() == 0.0


def test_reset_clears_everything():
    t = UsageTracker()
    t.record(agent="X", model="claude-haiku-4-5-20251001", input_tokens=10,
              output_tokens=10, latency_s=0.1)
    t.reset()
    assert t.calls == []
    assert t.total_cost() == 0.0
    assert "Aucun appel" in t.report()


def test_report_lists_every_agent_and_a_total():
    t = UsageTracker()
    t.record(agent="ComparatorAgent", model="claude-sonnet-4-6", input_tokens=100,
              output_tokens=50, latency_s=1.0)
    t.record(agent="AnnotatorAgent", model="claude-haiku-4-5-20251001", input_tokens=100,
              output_tokens=50, latency_s=1.0)

    report = t.report()
    assert "AnnotatorAgent" in report
    assert "ComparatorAgent" in report
    assert "total" in report


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
