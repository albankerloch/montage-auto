"""Backends d'inférence. Deux implémentations, un seul contrat.

Le dépôt ne dépend d'aucun runtime local particulier. vLLM, Ollama, llama.cpp
server, LM Studio et SGLang exposent tous une API compatible OpenAI : viser ce
protocole plutôt qu'une bibliothèque évite d'embarquer une pile CUDA dans le
projet, et laisse le problème sm_120 là où il est — dans l'installation du
serveur, pas dans ce code.

Le point délicat n'est pas l'appel, c'est la **sortie structurée**. Toute
l'architecture repose sur des contrats Pydantic qui tiennent. Un VLM local sans
décodage contraint produit du JSON invalide assez souvent pour rendre le
pipeline inutilisable. Trois mécanismes sont donc tentés dans l'ordre :

  1. `response_format: {"type": "json_schema"}` — vLLM, llama.cpp, LM Studio ;
  2. `guided_json` en extra body — vLLM plus ancien ;
  3. schéma dans le prompt + réparation du texte + relance en montrant l'erreur
     de validation au modèle.

Le troisième n'est pas un vrai filet : il est là pour que l'échec soit lisible,
pas pour prétendre que ça marche. Si un serveur n'accepte ni (1) ni (2), la
console le dit.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from src.usage import usage

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def flatten_schema(schema: dict) -> dict:
    """Résout $defs/$ref : les schémas imbriqués sont mal gérés des deux côtés."""
    defs = schema.pop("$defs", {})
    if not defs:
        return schema

    def resolve(obj):
        if isinstance(obj, dict):
            if "$ref" in obj:
                return resolve(defs.get(obj["$ref"].split("/")[-1], {}))
            return {k: resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve(i) for i in obj]
        return obj

    return resolve(schema)


def coerce_json_strings(obj):
    """Un modèle sérialise parfois un tableau en chaîne JSON. On déplie."""
    if isinstance(obj, dict):
        return {k: coerce_json_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [coerce_json_strings(i) for i in obj]
    if isinstance(obj, str):
        stripped = obj.strip()
        if stripped.startswith(("{", "[")):
            try:
                return coerce_json_strings(json.loads(stripped))
            except json.JSONDecodeError:
                pass
    return obj


def strict_schema(schema: dict) -> dict:
    """Durcit un schéma pour le décodage contraint.

    `additionalProperties: false` partout, et `required` listant toutes les
    propriétés. Sans ça, un modèle contraint reste libre d'omettre des champs
    et de renvoyer un objet vide valide au regard du schéma.
    """
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" in schema:
            schema["additionalProperties"] = False
            schema["required"] = list(schema["properties"].keys())
        return {k: strict_schema(v) for k, v in schema.items()}
    if isinstance(schema, list):
        return [strict_schema(i) for i in schema]
    return schema


class Backend(Protocol):
    name: str

    def complete(
        self,
        system: str,
        user_content: str | list,
        output_schema: type[BaseModel],
        max_tokens: int,
        agent: str,
    ) -> BaseModel: ...


# ── Anthropic ───────────────────────────────────────────────────────────────


class AnthropicBackend:
    """Sortie structurée via tool use. Comportement historique, inchangé."""

    name = "anthropic"

    def __init__(self, model: str, max_retries: int = 4):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_retries = max_retries

    def complete(self, system, user_content, output_schema, max_tokens=4096, agent=""):
        schema = flatten_schema(output_schema.model_json_schema())
        schema.pop("title", None)

        last_error = None
        response = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.perf_counter()
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user_content}],
                    tools=[
                        {
                            "name": "structured_output",
                            "description": f"Return the structured result as {output_schema.__name__}",
                            "input_schema": schema,
                        }
                    ],
                    tool_choice={"type": "any"},
                )
                # Une tentative en échec n'est pas facturée, et l'attente du
                # backoff n'est pas de la latence d'appel : on ne journalise
                # que l'appel réussi.
                usage.record(
                    agent=agent or self.__class__.__name__,
                    model=self.model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    latency_s=time.perf_counter() - t0,
                )
                break
            except (self._anthropic.OverloadedError, self._anthropic.RateLimitError) as e:
                last_error = e
                wait = 2**attempt * 5
                print(f"[anthropic] surcharge/quota, nouvelle tentative dans {wait}s "
                      f"({attempt + 1}/{self.max_retries})")
                time.sleep(wait)
        else:
            raise last_error

        for block in response.content:
            if block.type == "tool_use" and block.name == "structured_output":
                data = coerce_json_strings(block.input)
                try:
                    return output_schema.model_validate(data)
                except Exception as e:
                    raise RuntimeError(
                        f"Sortie invalide : {e}\nBrut : {json.dumps(block.input, indent=2)}"
                    )

        raise RuntimeError(
            f"Aucune sortie structurée. Blocs reçus : {[b.type for b in response.content]}"
        )


# ── Endpoint compatible OpenAI (vLLM, Ollama, llama.cpp, LM Studio) ─────────


def _to_openai_content(user_content: str | list) -> list[dict]:
    """Traduit les blocs façon Anthropic vers le format OpenAI.

    Les images passent en data URI : les serveurs locaux ne savent pas aller
    chercher un fichier sur le disque de l'appelant, même en local.
    """
    if isinstance(user_content, str):
        return [{"type": "text", "text": user_content}]

    out: list[dict] = []
    for block in user_content:
        kind = block.get("type")
        if kind == "text":
            out.append({"type": "text", "text": block["text"]})
        elif kind == "image":
            src = block["source"]
            if src.get("type") == "base64":
                media = src.get("media_type", "image/jpeg")
                out.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media};base64,{src['data']}"},
                    }
                )
            elif src.get("type") == "url":
                out.append({"type": "image_url", "image_url": {"url": src["url"]}})
    return out


class OpenAICompatBackend:
    """Client HTTP minimal pour un serveur d'inférence local.

    Volontairement écrit avec `requests` plutôt qu'avec le SDK openai : une
    dépendance de moins, et le corps de la requête reste lisible — ce qui
    compte, puisque les serveurs diffèrent précisément sur les champs
    d'extension qu'on leur envoie.
    """

    name = "openai_compat"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "not-needed",
        timeout: float = 300.0,
        max_retries: int = 3,
        temperature: float = 0.0,
        structured_mode: str = "auto",  # auto | json_schema | guided_json | prompt
    ):
        self._agent = self.__class__.__name__
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.structured_mode = structured_mode
        self._working_mode: str | None = None if structured_mode == "auto" else structured_mode

    # -- construction de la requête ------------------------------------------

    def _payload(self, system, content, schema, name, max_tokens, mode) -> dict:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            # Température nulle : le cache du graphe est adressé par contenu,
            # un nœud non déterministe le rend menteur.
            "temperature": self.temperature,
        }
        if mode == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            }
        elif mode == "guided_json":
            body["guided_json"] = schema
            body["extra_body"] = {"guided_json": schema}
        elif mode == "prompt":
            messages[0]["content"] = (
                f"{system}\n\nRéponds UNIQUEMENT par un objet JSON conforme à ce "
                f"schéma, sans texte autour ni balises de code :\n{json.dumps(schema)}"
            )
        return body

    def _post(self, body: dict) -> dict:
        """POST + journalisation coût/latence.

        Un serveur local ne facture rien, mais il consomme du temps — et la
        latence est justement ce qu'on veut comparer au cloud. Les tokens
        remontés par le champ `usage` sont journalisés quand le serveur le
        fournit (vLLM et llama.cpp le font, pas tous) ; le tarif vaut 0 pour un
        modèle absent de la grille, donc le coût affiché reste juste.
        """
        import requests

        last: Exception | None = None
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=body,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=self.timeout,
                )
            except Exception as e:  # noqa: BLE001 — serveur pas encore levé, réseau
                last = e
                wait = 2**attempt
                print(f"[local] {self.base_url} injoignable ({e}), nouvelle tentative dans {wait}s")
                time.sleep(wait)
                continue

            if r.status_code == 200:
                data = r.json()
                u = data.get("usage") or {}
                usage.record(
                    agent=getattr(self, "_agent", self.__class__.__name__),
                    model=f"{LOCAL_PREFIX}{self.model}",
                    input_tokens=int(u.get("prompt_tokens", 0) or 0),
                    output_tokens=int(u.get("completion_tokens", 0) or 0),
                    latency_s=time.perf_counter() - t0,
                )
                return data
            if r.status_code in (400, 422):
                # Champ refusé : c'est ce qui distingue les serveurs entre eux.
                raise _UnsupportedRequest(f"HTTP {r.status_code}: {r.text[:400]}")
            if r.status_code in (429, 500, 502, 503, 504):
                last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                wait = 2**attempt
                print(f"[local] HTTP {r.status_code}, nouvelle tentative dans {wait}s")
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
        raise RuntimeError(f"Serveur local injoignable après {self.max_retries} tentatives : {last}")

    # -- appel ---------------------------------------------------------------

    def complete(self, system, user_content, output_schema, max_tokens=4096, agent=""):
        schema = strict_schema(flatten_schema(output_schema.model_json_schema()))
        schema.pop("title", None)
        content = _to_openai_content(user_content)
        name = output_schema.__name__
        self._agent = agent or self.__class__.__name__

        modes = [self._working_mode] if self._working_mode else ["json_schema", "guided_json", "prompt"]

        last_err: Exception | None = None
        for mode in modes:
            try:
                data = self._post(self._payload(system, content, schema, name, max_tokens, mode))
            except _UnsupportedRequest as e:
                print(f"[local] mode « {mode} » refusé par le serveur, essai suivant")
                last_err = e
                continue

            text = data["choices"][0]["message"].get("content") or ""
            try:
                parsed = self._parse(text, output_schema)
            except ValidationError as e:
                if mode == "prompt":
                    parsed = self._repair(system, content, schema, name, max_tokens, text, e, output_schema)
                else:
                    raise RuntimeError(
                        f"Le serveur a accepté « {mode} » mais renvoyé une sortie non "
                        f"conforme au schéma : {e}\nBrut : {text[:400]}"
                    )
            if self._working_mode is None:
                self._working_mode = mode
                print(f"[local] sortie structurée via « {mode} »")
            return parsed

        raise RuntimeError(
            "Aucun mode de sortie structurée accepté par "
            f"{self.base_url}. Dernier refus : {last_err}\n"
            "Sans décodage contraint, les contrats Pydantic du pipeline ne "
            "tiennent pas : préférer un serveur qui le supporte (vLLM, "
            "llama.cpp, LM Studio) plutôt que forcer --structured-mode prompt."
        )

    def _parse(self, text: str, output_schema: type[BaseModel]) -> BaseModel:
        cleaned = _FENCE.sub("", text).strip()
        if not cleaned:
            raise ValidationError.from_exception_data(output_schema.__name__, [])
        try:
            raw = json.loads(cleaned)
        except json.JSONDecodeError:
            start = min((i for i in (cleaned.find("{"), cleaned.find("[")) if i >= 0), default=-1)
            end = max(cleaned.rfind("}"), cleaned.rfind("]"))
            if start < 0 or end <= start:
                raise
            raw = json.loads(cleaned[start : end + 1])
        return output_schema.model_validate(coerce_json_strings(raw))

    def _repair(self, system, content, schema, name, max_tokens, bad, error, output_schema):
        """Une seule relance, en montrant au modèle l'erreur de validation."""
        print(f"[local] sortie non conforme, une relance avec l'erreur : {str(error)[:120]}")
        retry_content = list(content) + [
            {
                "type": "text",
                "text": (
                    f"Ta réponse précédente était invalide.\nRéponse : {bad[:1500]}\n"
                    f"Erreur de validation : {error}\n"
                    "Renvoie UNIQUEMENT le JSON corrigé."
                ),
            }
        ]
        data = self._post(self._payload(system, retry_content, schema, name, max_tokens, "prompt"))
        return self._parse(data["choices"][0]["message"].get("content") or "", output_schema)


class _UnsupportedRequest(RuntimeError):
    """Le serveur a rejeté un champ d'extension : on essaie un autre mode."""


# ── Sélection ───────────────────────────────────────────────────────────────

LOCAL_PREFIX = "local/"


def make_backend(model: str) -> Backend:
    """Choisit le backend d'après le nom du modèle.

    `local/<nom>` route vers le serveur compatible OpenAI ; tout le reste va
    chez Anthropic. Le préfixe fait partie du nom du modèle, donc il fait
    partie de la clé du nœud `annot` : basculer local/cloud invalide les
    annotations et rien d'autre, ce qui est exactement le comportement voulu.
    """
    from src import config

    if model.startswith(LOCAL_PREFIX):
        return OpenAICompatBackend(
            model=model[len(LOCAL_PREFIX):],
            base_url=os.environ.get("LOCAL_VLM_BASE_URL", config.LOCAL_VLM_BASE_URL),
            api_key=os.environ.get("LOCAL_VLM_API_KEY", "not-needed"),
            structured_mode=os.environ.get("LOCAL_VLM_STRUCTURED_MODE", config.LOCAL_VLM_STRUCTURED_MODE),
            timeout=config.LOCAL_VLM_TIMEOUT_S,
        )
    return AnthropicBackend(model)


def is_local(model: str) -> bool:
    return model.startswith(LOCAL_PREFIX)
