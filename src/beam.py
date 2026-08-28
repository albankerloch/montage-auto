"""Faisceau de candidats + classement par comparaison par paires.

Remplace la boucle CRITIC → REVISION → SCENARIO :

  - la génération est un faisceau de N intentions **indépendantes** (donc
    parallélisables, sans dépendance séquentielle) et entièrement déterministe :
    les intentions sont des `Preset`, pas des prompts, donc générer le faisceau
    coûte zéro token ;
  - le classement est une comparaison par paires (« lequel des deux ? »), pas
    une note absolue. Un LLM est nettement plus stable en comparatif qu'en
    notation 0.0–1.0 — ce qui était le vice de fond du CRITIC et de son seuil
    à 0.70 sorti de nulle part ;
  - la sortie est un classement, pas un verdict : le premier va au monteur, les
    suivants sont les alternates sur une timeline B.

Une boucle ne peut structurellement pas produire un classement : elle converge
vers un artefact unique.
"""
from __future__ import annotations

import hashlib
from typing import Callable, Iterable, Sequence

from pydantic import BaseModel

from src.assemble import PRESETS, Candidate, Preset, solve
from src.models import VideoSegment

# Un comparateur reçoit (a, b) et renvoie "A" ou "B" — plus une raison libre.
Comparator = Callable[[Candidate, Candidate], tuple[str, str]]


class RankedCandidate(BaseModel):
    candidate: Candidate
    rank: int
    wins: int
    notes: list[str] = []


def generate(
    segments: Sequence[VideoSegment],
    preset_names: Iterable[str],
    k_per_preset: int = 2,
    time_limit_s: float = 15.0,
) -> list[Candidate]:
    """Génère le faisceau. Déterministe, aucun appel LLM.

    Les presets sont indépendants : `solve` peut être lancé en parallèle
    (ThreadPool ou nœuds de graphe séparés) sans rien changer au résultat.
    """
    out: list[Candidate] = []
    for name in preset_names:
        preset = PRESETS[name]
        out.extend(solve(segments, preset, k=k_per_preset, time_limit_s=time_limit_s))
    return out


def _jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def dedupe(candidates: Sequence[Candidate], threshold: float = 0.85) -> list[Candidate]:
    """Élague les candidats trop proches : deux presets convergent souvent."""
    kept: list[Candidate] = []
    for c in candidates:
        if any(_jaccard(c.selection, k.selection) >= threshold for k in kept):
            continue
        kept.append(c)
    return kept


def _orientation(a: Candidate, b: Candidate) -> bool:
    """Ordre de présentation d'une paire, stable et sans biais systématique.

    Un LLM favorise légèrement la première option présentée. On ne peut pas se
    payer le double appel (A/B puis B/A), donc on rend l'orientation
    pseudo-aléatoire mais déterministe : le biais ne s'aligne plus sur un preset.
    """
    key = "|".join(sorted([f"{a.preset}#{a.rank_in_preset}", f"{b.preset}#{b.rank_in_preset}"]))
    return hashlib.sha256(key.encode()).digest()[0] % 2 == 0


def rank(
    candidates: Sequence[Candidate],
    comparator: Comparator,
    max_candidates: int = 6,
) -> tuple[list[RankedCandidate], int]:
    """Trie les candidats par tri fusion, comparateur mémoïsé.

    Renvoie (classement, nombre d'appels au comparateur). Le tri fusion donne
    ~n·log2(n) comparaisons — 6 candidats ≈ 10 appels, à comparer aux 3 appels
    CRITIC + 2 REVISION de la boucle, pour un résultat classé et non un verdict.
    """
    pool = list(candidates)[:max_candidates]
    if len(pool) <= 1:
        return [RankedCandidate(candidate=c, rank=i, wins=0) for i, c in enumerate(pool)], 0

    memo: dict[tuple[int, int], tuple[str, str]] = {}
    wins: dict[int, int] = {id(c): 0 for c in pool}
    notes: dict[int, list[str]] = {id(c): [] for c in pool}
    calls = 0

    def better(a: Candidate, b: Candidate) -> bool:
        """True si a passe devant b."""
        nonlocal calls
        ka, kb = id(a), id(b)
        cache_key = (min(ka, kb), max(ka, kb))
        if cache_key not in memo:
            first, second = (a, b) if _orientation(a, b) else (b, a)
            verdict, reason = comparator(first, second)
            calls += 1
            winner = first if verdict.upper() == "A" else second
            memo[cache_key] = ("A" if winner is a else "B", reason)
            wins[id(winner)] += 1
            notes[ka].append(reason)
            notes[kb].append(reason)
        return memo[cache_key][0] == "A"

    def merge_sort(items: list[Candidate]) -> list[Candidate]:
        if len(items) <= 1:
            return items
        mid = len(items) // 2
        left, right = merge_sort(items[:mid]), merge_sort(items[mid:])
        merged: list[Candidate] = []
        i = j = 0
        while i < len(left) and j < len(right):
            if better(left[i], right[j]):
                merged.append(left[i]); i += 1
            else:
                merged.append(right[j]); j += 1
        return merged + left[i:] + right[j:]

    ordered = merge_sort(pool)
    return (
        [
            RankedCandidate(
                candidate=c, rank=r, wins=wins[id(c)], notes=notes[id(c)][:3]
            )
            for r, c in enumerate(ordered)
        ],
        calls,
    )


def objective_prefilter(candidates: Sequence[Candidate], per_preset: int = 1) -> list[Candidate]:
    """Pré-filtre déterministe avant le classement LLM.

    Les objectifs de presets différents ne sont PAS comparables entre eux (ce
    sont des fonctions objectif différentes) : on ne garde donc que les
    meilleurs *au sein* de chaque preset, jamais un tri global par objectif.
    """
    by_preset: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_preset.setdefault(c.preset, []).append(c)
    out: list[Candidate] = []
    for group in by_preset.values():
        group.sort(key=lambda c: -c.objective)
        out.extend(group[:per_preset])
    return out
