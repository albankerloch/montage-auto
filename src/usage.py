"""Suivi coût/latence des appels API — en dehors du graphe, pas dedans.

Le nombre de tokens et le temps d'un appel ne sont pas des propriétés du
résultat qu'ils produisent : les mêmes annotations, servies depuis le cache,
ne coûtent rien la seconde fois (`src/graph.py`). Le suivi vit donc à côté,
dans un accumulateur global réinitialisé au début de chaque run, plutôt que
dans un artefact — sinon le coût d'un run se retrouverait gelé dans le cache
d'un autre.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from src.config import MODEL_PRICING_PER_MTOK


@dataclass
class ApiCall:
    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float


def _price_per_mtok(model: str) -> tuple[float, float]:
    """(prix input, prix output) par million de tokens. (0.0, 0.0) si le modèle
    n'est pas dans la grille — on ne devine pas un tarif, on l'affiche à 0."""
    return MODEL_PRICING_PER_MTOK.get(model, (0.0, 0.0))


def _cost(call: ApiCall) -> float:
    in_price, out_price = _price_per_mtok(call.model)
    return call.input_tokens / 1_000_000 * in_price + call.output_tokens / 1_000_000 * out_price


class UsageTracker:
    """Accumulateur global, protégé par verrou pour un futur appel concurrent
    (les presets sont indépendants, cf. `src/nodes.py::_candidates`)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: list[ApiCall] = []

    def record(
        self, *, agent: str, model: str, input_tokens: int, output_tokens: int, latency_s: float
    ) -> None:
        with self._lock:
            self._calls.append(ApiCall(agent, model, input_tokens, output_tokens, latency_s))

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()

    @property
    def calls(self) -> list[ApiCall]:
        with self._lock:
            return list(self._calls)

    def total_cost(self) -> float:
        return sum(_cost(c) for c in self.calls)

    def total_latency_s(self) -> float:
        return sum(c.latency_s for c in self.calls)

    def by_agent(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for c in self.calls:
            agg = out.setdefault(
                c.agent,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_s": 0.0, "cost": 0.0},
            )
            agg["calls"] += 1
            agg["input_tokens"] += c.input_tokens
            agg["output_tokens"] += c.output_tokens
            agg["latency_s"] += c.latency_s
            agg["cost"] += _cost(c)
        return out

    def report(self) -> str:
        calls = self.calls
        if not calls:
            return "Aucun appel API (tout servi par le cache, ou --rank manual)."

        lines = []
        for agent, agg in sorted(self.by_agent().items()):
            lines.append(
                f"  {agent:<18} {agg['calls']:>3} appel(s)  "
                f"{agg['input_tokens']:>7} in / {agg['output_tokens']:>6} out tok  "
                f"{agg['latency_s']:>6.1f}s  ${agg['cost']:.4f}"
            )
        lines.append(
            f"  {'total':<18} {len(calls):>3} appel(s)  "
            f"{sum(c.input_tokens for c in calls):>7} in / "
            f"{sum(c.output_tokens for c in calls):>6} out tok  "
            f"{self.total_latency_s():>6.1f}s  ${self.total_cost():.4f}"
        )
        return "\n".join(lines)


# Un seul accumulateur par process : les agents sont recréés à chaque appel de
# nœud (`nodes.py::_annot`, `_ranked`), donc rien de plus stable ne pourrait
# porter cet état.
usage = UsageTracker()
