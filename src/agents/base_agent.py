"""Wrapper d'agent. La logique d'appel vit dans `backends.py`.

`BaseAgent.call()` garde exactement la même signature : les agents existants
n'ont pas à savoir si le modèle tourne chez Anthropic ou sur la machine. Le
suivi coût/latence est descendu dans les backends, pour que le backend local
soit journalisé lui aussi — comparer local et cloud sur la latence est
précisément l'intérêt de `src/bench_annot.py`.
"""
from __future__ import annotations

from pydantic import BaseModel

from src.agents.backends import (  # noqa: F401 — réexports pour compat
    coerce_json_strings as _coerce_json_strings,
    flatten_schema as _flatten_schema,
    make_backend,
)


class BaseAgent:
    def __init__(self, model: str):
        self.model = model
        self.backend = make_backend(model)

    @property
    def client(self):
        """Compat : certains appels historiques touchaient `agent.client`."""
        return getattr(self.backend, "client", None)

    def call(
        self,
        system: str,
        user_content: str | list,
        output_schema: type[BaseModel],
        max_tokens: int = 4096,
        max_retries: int = 4,  # noqa: ARG002 — porté par le backend
    ) -> BaseModel:
        """Appel à sortie structurée, contraint par `output_schema`."""
        try:
            return self.backend.complete(
                system, user_content, output_schema, max_tokens,
                # Le nom d'agent vient d'ici : sans lui, le rapport `usage`
                # afficherait le nom de la classe de backend et perdrait la
                # ventilation par agent.
                agent=self.__class__.__name__,
            )
        except RuntimeError as e:
            raise RuntimeError(f"[{self.__class__.__name__}] {e}") from e
