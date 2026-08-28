"""Graphe de dépendances adressé par contenu.

Remplace l'ordonnancement impératif de `orchestrator.py` : on ne décrit plus une
suite d'étapes, on déclare des artefacts dérivés. Chaque noeud porte une clé
`sha256(nom + version + params + clés des dépendances)` ; demander un artefact
remonte le graphe et ne matérialise que ce qui manque.

Conséquences :
  - la reprise après crash est une propriété du cache, pas une feature ;
  - ajouter un rush ne réanalyse que ce rush ;
  - changer la durée cible invalide le plan mais pas les annotations vision ;
  - une correction du monteur est une entrée du graphe, pas une étape finale.

Le corps de `fn` n'est volontairement PAS haché : un hash de bytecode invalide
tout le cache à chaque reformatage. C'est `version` qui déclare qu'un calcul a
changé de sémantique — à incrémenter à la main, comme une migration.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from pydantic import BaseModel

Codec = Literal["json", "pydantic", "pydantic_list", "path", "path_list"]

_FINGERPRINT_CHUNK = 4 * 1024 * 1024  # 4 Mo en tête et en queue


def _canon(obj: Any) -> str:
    """Sérialisation canonique et stable d'un dict de paramètres."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint_file(path: str | Path) -> str:
    """Empreinte d'un fichier source.

    Taille + 4 Mo de tête + 4 Mo de queue. On ne hache pas les 40 Go de rushes :
    la probabilité de collision sur des fichiers vidéo distincts est négligeable
    et le coût d'un hash complet ferait perdre l'intérêt du cache.
    """
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())
    with p.open("rb") as f:
        h.update(f.read(_FINGERPRINT_CHUNK))
        if size > 2 * _FINGERPRINT_CHUNK:
            f.seek(-_FINGERPRINT_CHUNK, 2)
            h.update(f.read(_FINGERPRINT_CHUNK))
    return h.hexdigest()[:24]


@dataclass
class Node:
    """Un artefact dérivé.

    `fn` reçoit les valeurs résolues des dépendances en kwargs, plus `**params`.
    """

    name: str
    fn: Callable[..., Any] | None = None
    deps: dict[str, "Node"] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    version: str = "1"
    codec: Codec = "json"
    model: type[BaseModel] | None = None
    label: str | None = None  # affichage seulement, hors clé

    _key: str | None = field(default=None, repr=False, compare=False)

    def key(self) -> str:
        if self._key is None:
            h = hashlib.sha256()
            h.update(self.name.encode())
            h.update(b"\x00")
            h.update(self.version.encode())
            h.update(b"\x00")
            h.update(_canon(self.params).encode())
            for dep_name in sorted(self.deps):
                h.update(b"\x00")
                h.update(dep_name.encode())
                h.update(b"=")
                h.update(self.deps[dep_name].key().encode())
            self._key = h.hexdigest()[:24]
        return self._key

    def walk(self) -> Iterable["Node"]:
        """Post-ordre : dépendances avant le noeud."""
        for dep in self.deps.values():
            yield from dep.walk()
        yield self

    def __str__(self) -> str:
        return f"{self.name}:{self.key()[:8]}"


def source(path: str | Path, name: str = "source") -> Node:
    """Noeud feuille : un fichier d'entrée, identifié par son empreinte.

    `fn` avale les params (`materialize` les passe systématiquement) : ils sont
    là pour la clé, pas pour le calcul.
    """
    p = Path(path).resolve()
    return Node(
        name=name,
        fn=lambda **_: str(p),
        params={"fingerprint": fingerprint_file(p), "suffix": p.suffix},
        codec="json",
        label=p.name,
    )


class Store:
    """Cache disque. Un sous-dossier par nom de noeud, un fichier par clé."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _slot(self, node: Node) -> Path:
        return self.root / node.name / node.key()

    def has(self, node: Node) -> bool:
        slot = self._slot(node)
        if node.codec in ("path", "path_list"):
            return (slot / "manifest.json").exists()
        return slot.with_suffix(".json").exists()

    def get(self, node: Node) -> Any:
        slot = self._slot(node)
        if node.codec in ("path", "path_list"):
            names = json.loads((slot / "manifest.json").read_text())
            paths = [str(slot / n) for n in names]
            return paths if node.codec == "path_list" else paths[0]

        raw = json.loads(slot.with_suffix(".json").read_text(encoding="utf-8"))
        if node.codec == "pydantic":
            assert node.model is not None, f"{node.name}: codec pydantic sans model"
            return node.model.model_validate(raw)
        if node.codec == "pydantic_list":
            assert node.model is not None, f"{node.name}: codec pydantic_list sans model"
            return [node.model.model_validate(r) for r in raw]
        return raw

    def put(self, node: Node, value: Any) -> Any:
        slot = self._slot(node)
        slot.parent.mkdir(parents=True, exist_ok=True)

        if node.codec in ("path", "path_list"):
            slot.mkdir(parents=True, exist_ok=True)
            srcs = value if node.codec == "path_list" else [value]
            names: list[str] = []
            for src in srcs:
                src_p = Path(src)
                dest = slot / src_p.name
                if src_p.resolve() != dest.resolve():
                    shutil.copy2(src_p, dest)
                names.append(src_p.name)
            (slot / "manifest.json").write_text(json.dumps(names))
            out = [str(slot / n) for n in names]
            return out if node.codec == "path_list" else out[0]

        if node.codec == "pydantic":
            raw = value.model_dump(mode="json")
        elif node.codec == "pydantic_list":
            raw = [v.model_dump(mode="json") for v in value]
        else:
            raw = value

        tmp = slot.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(slot.with_suffix(".json"))  # écriture atomique
        return value


def stale(node: Node, store: Store) -> list[Node]:
    """Noeuds à recalculer si on demande `node` maintenant."""
    seen: set[str] = set()
    todo: list[Node] = []
    for n in node.walk():
        if n.key() in seen:
            continue
        seen.add(n.key())
        if not store.has(n):
            todo.append(n)
    return todo


def materialize(node: Node, store: Store, on_event: Callable[..., None] | None = None) -> Any:
    """Matérialise `node`, en ne calculant que les noeuds absents du cache."""
    memo: dict[str, Any] = {}

    def emit(kind: str, n: Node, detail: str = "") -> None:
        if on_event:
            on_event(kind, n, detail)

    for n in node.walk():
        k = n.key()
        if k in memo:
            continue
        if store.has(n):
            memo[k] = store.get(n)
            emit("hit", n)
            continue
        if n.fn is None:
            raise RuntimeError(f"Noeud {n} absent du cache et sans fn")
        emit("miss", n)
        kwargs = {dep_name: memo[dep.key()] for dep_name, dep in n.deps.items()}
        value = n.fn(**kwargs, **n.params)
        memo[k] = store.put(n, value)
        emit("done", n)

    return memo[node.key()]


def render_plan(node: Node, store: Store) -> str:
    """Rendu texte du graphe avec l'état du cache — pour le debug et les slides."""
    lines: list[str] = []
    seen: set[str] = set()

    def rec(n: Node, depth: int) -> None:
        mark = "✓" if store.has(n) else "•"
        dup = " (déjà)" if n.key() in seen else ""
        label = f" [{n.label}]" if n.label else ""
        lines.append(f"{'  ' * depth}{mark} {n.name}:{n.key()[:8]}{label}{dup}")
        if n.key() in seen:
            return
        seen.add(n.key())
        for dep in n.deps.values():
            rec(dep, depth + 1)

    rec(node, 0)
    return "\n".join(lines)
