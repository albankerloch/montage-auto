"""Le graphe : clés stables, cache, invalidation sélective.

Aucune clé API, aucun fichier vidéo — ces tests tournent en CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph import Node, Store, materialize, stale  # noqa: E402


def _counter():
    calls: dict[str, int] = {}

    def bump(name: str):
        calls[name] = calls.get(name, 0) + 1

    return calls, bump


def build(tmp: Path, target: int, scale: int = 2):
    calls, bump = _counter()

    def raw():
        bump("raw")
        return [1, 2, 3]

    def scaled(base, factor):
        bump("scaled")
        return [b * factor for b in base]

    def plan(values, target):
        bump("plan")
        return {"sum": sum(values), "target": target}

    a = Node("raw", raw)
    b = Node("scaled", scaled, {"base": a}, params={"factor": scale})
    c = Node("plan", plan, {"values": b}, params={"target": target})
    return c, calls


def test_cache_hit_avoids_recompute(tmp_path):
    store = Store(tmp_path)
    node, calls = build(tmp_path, target=60)
    assert materialize(node, store) == {"sum": 12, "target": 60}
    assert calls == {"raw": 1, "scaled": 1, "plan": 1}

    node2, calls2 = build(tmp_path, target=60)
    assert materialize(node2, store) == {"sum": 12, "target": 60}
    assert calls2 == {}, "tout devait venir du cache"


def test_param_change_invalidates_only_downstream(tmp_path):
    """Le cœur de l'affaire : changer la durée cible ne réanalyse rien en amont."""
    store = Store(tmp_path)
    node, _ = build(tmp_path, target=60)
    materialize(node, store)

    node2, calls2 = build(tmp_path, target=90)
    materialize(node2, store)
    assert calls2 == {"plan": 1}, "seul le plan devait être recalculé"


def test_upstream_change_invalidates_downstream(tmp_path):
    store = Store(tmp_path)
    node, _ = build(tmp_path, target=60, scale=2)
    materialize(node, store)

    node2, calls2 = build(tmp_path, target=60, scale=3)
    materialize(node2, store)
    assert calls2 == {"scaled": 1, "plan": 1}
    assert "raw" not in calls2


def test_version_bump_invalidates(tmp_path):
    store = Store(tmp_path)
    n1 = Node("f", lambda: 1, version="1")
    materialize(n1, store)
    n2 = Node("f", lambda: 2, version="2")
    assert materialize(n2, store) == 2, "le bump de version doit forcer le recalcul"


def test_stale_reports_what_is_missing(tmp_path):
    store = Store(tmp_path)
    node, _ = build(tmp_path, target=60)
    assert {n.name for n in stale(node, store)} == {"raw", "scaled", "plan"}
    materialize(node, store)
    assert stale(node, store) == []


def test_resume_after_crash(tmp_path):
    """Un plantage en aval ne perd pas le travail amont — la reprise est
    une propriété du cache, pas une feature à écrire."""
    store = Store(tmp_path)
    calls, bump = _counter()

    def heavy():
        bump("heavy")
        return list(range(5))

    def boom(values):
        bump("boom")
        raise RuntimeError("plantage simulé")

    a = Node("heavy", heavy)
    b = Node("boom", boom, {"values": a})
    try:
        materialize(b, store)
    except RuntimeError:
        pass
    assert calls["heavy"] == 1

    def fixed(values):
        bump("fixed")
        return sum(values)

    b2 = Node("boom", fixed, {"values": Node("heavy", heavy)}, version="2")
    assert materialize(b2, store) == 10
    assert calls["heavy"] == 1, "le nœud coûteux ne devait pas être recalculé"


if __name__ == "__main__":
    import tempfile

    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
            print(f"ok  {name}")
