"""Le backend local, testé contre un faux serveur compatible OpenAI.

Pas de GPU ici, et c'est le principe : le dépôt parle HTTP, donc ce qui est
testable est exactement ce que le dépôt contrôle — la traduction des blocs, la
négociation du mode de sortie structurée, la réparation, et les messages
d'échec. La pile CUDA reste le problème de l'installation du serveur.

Le faux serveur imite les comportements réellement rencontrés :
  - un serveur qui accepte `response_format: json_schema` ;
  - un serveur plus ancien qui le refuse en 400 et n'accepte que `guided_json` ;
  - un serveur sans décodage contraint, qui renvoie du JSON dans des balises de
    code, ou du JSON invalide.
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.backends import (  # noqa: E402
    OpenAICompatBackend,
    _to_openai_content,
    flatten_schema,
    is_local,
    strict_schema,
)


class Shot(BaseModel):
    quality_score: float = Field(ge=0.0, le=1.0)
    emotion: str
    tags: list[str]


GOOD = {"quality_score": 0.7, "emotion": "calm", "tags": ["a", "b"]}


class FakeServer:
    """Serveur compatible OpenAI paramétrable."""

    def __init__(self, behaviour: str):
        self.behaviour = behaviour
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                status, payload = outer.respond(body)
                raw = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def respond(self, body: dict):
        def msg(text):
            return 200, {"choices": [{"message": {"content": text}}]}

        b = self.behaviour
        if b == "json_schema" and "response_format" in body:
            return msg(json.dumps(GOOD))
        if b == "json_schema":
            return 400, {"error": "response_format required"}

        if b == "guided_json_only":
            if "response_format" in body:
                return 400, {"error": "unknown field response_format"}
            if "guided_json" in body:
                return msg(json.dumps(GOOD))
            return msg("je pense que ce plan est plutôt bon")

        if b == "prompt_only":
            if "response_format" in body or "guided_json" in body:
                return 422, {"error": "unsupported"}
            return msg(f"Voici le résultat :\n```json\n{json.dumps(GOOD)}\n```")

        if b == "invalid_then_good":
            if len(self.requests) == 1:
                return msg(json.dumps({"quality_score": 5.0, "emotion": "calm", "tags": []}))
            return msg(json.dumps(GOOD))

        if b == "never_structured":
            return 400, {"error": "unsupported"}

        if b == "flaky":
            if len(self.requests) < 3:
                return 503, {"error": "loading model"}
            return msg(json.dumps(GOOD))

        raise AssertionError(b)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()


def backend(server: FakeServer, **kw) -> OpenAICompatBackend:
    return OpenAICompatBackend(
        model="fake-vlm", base_url=f"http://127.0.0.1:{server.port}/v1",
        max_retries=4, **kw,
    )


# ── Traduction des blocs ────────────────────────────────────────────────────


def test_image_blocks_become_data_uris():
    """Un serveur local ne peut pas lire un chemin sur le disque de l'appelant."""
    content = _to_openai_content([
        {"type": "text", "text": "regarde"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}},
    ])
    assert content[0] == {"type": "text", "text": "regarde"}
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"


def test_plain_string_content_is_wrapped():
    assert _to_openai_content("bonjour") == [{"type": "text", "text": "bonjour"}]


def test_strict_schema_forbids_omitting_fields():
    """Sans `required` ni `additionalProperties: false`, un modèle contraint
    reste libre de renvoyer un objet vide, valide au regard du schéma."""
    s = strict_schema(flatten_schema(Shot.model_json_schema()))
    assert s["additionalProperties"] is False
    assert set(s["required"]) == {"quality_score", "emotion", "tags"}


# ── Négociation du mode ─────────────────────────────────────────────────────


def test_json_schema_server():
    with FakeServer("json_schema") as srv:
        out = backend(srv).complete("sys", "hi", Shot, 256)
    assert out.quality_score == 0.7
    assert "response_format" in srv.requests[0]


def test_falls_back_to_guided_json():
    """vLLM ancien : refuse `response_format`, accepte `guided_json`."""
    with FakeServer("guided_json_only") as srv:
        out = backend(srv).complete("sys", "hi", Shot, 256)
    assert out.emotion == "calm"
    assert "response_format" in srv.requests[0]
    assert "guided_json" in srv.requests[1]


def test_falls_back_to_prompt_and_strips_code_fences():
    with FakeServer("prompt_only") as srv:
        out = backend(srv).complete("sys", "hi", Shot, 256)
    assert out.tags == ["a", "b"]
    assert len(srv.requests) == 3, "les deux modes contraints doivent être essayés d'abord"


def test_mode_is_memoised_across_calls():
    """Renégocier à chaque plan doublerait le nombre d'appels sur 2000 scènes."""
    with FakeServer("guided_json_only") as srv:
        b = backend(srv)
        b.complete("sys", "hi", Shot, 256)
        first = len(srv.requests)
        b.complete("sys", "hi", Shot, 256)
    assert len(srv.requests) == first + 1


def test_forced_mode_skips_negotiation():
    with FakeServer("json_schema") as srv:
        backend(srv, structured_mode="json_schema").complete("sys", "hi", Shot, 256)
    assert len(srv.requests) == 1


# ── Échecs ──────────────────────────────────────────────────────────────────


def test_no_structured_mode_gives_an_actionable_error():
    """Le message doit dire quoi faire, pas seulement que ça a raté."""
    with FakeServer("never_structured") as srv:
        with pytest.raises(RuntimeError) as exc:
            backend(srv).complete("sys", "hi", Shot, 256)
    msg = str(exc.value)
    assert "décodage contraint" in msg
    assert "vLLM" in msg, "le message doit nommer des serveurs qui conviennent"


def test_constrained_server_returning_bad_data_is_not_silently_repaired():
    """Si le serveur prétend contraindre et sort hors schéma, c'est une erreur
    du serveur. La masquer par une relance cacherait une mauvaise config."""
    with FakeServer("invalid_then_good") as srv:
        with pytest.raises(RuntimeError, match="non\n?\\s*conforme|non conforme"):
            backend(srv, structured_mode="json_schema").complete("sys", "hi", Shot, 256)
    assert len(srv.requests) == 1, "aucune relance ne doit masquer le problème"


def test_transient_5xx_are_retried():
    """Un serveur qui charge encore son modèle renvoie 503 pendant un moment."""
    with FakeServer("flaky") as srv:
        out = backend(srv).complete("sys", "hi", Shot, 256)
    assert out.quality_score == 0.7
    assert len(srv.requests) >= 3


def test_unreachable_server_names_the_url():
    b = OpenAICompatBackend(model="m", base_url="http://127.0.0.1:1/v1", max_retries=1)
    with pytest.raises(RuntimeError, match="injoignable"):
        b.complete("sys", "hi", Shot, 256)


# ── Routage ─────────────────────────────────────────────────────────────────


def test_local_prefix_routes_and_is_part_of_the_cache_key():
    assert is_local("local/Qwen/Qwen3-VL-8B-Instruct")
    assert not is_local("claude-haiku-4-5-20251001")


def test_switching_backend_invalidates_only_annotation(tmp_path):
    from src import nodes
    from src.assemble import PRESETS

    rush = Path("tests/fixtures/rushes/rush_0.mp4")
    if not rush.exists():
        pytest.skip("fixtures vidéo absentes")

    def keys(model, width):
        g = nodes.build(
            [str(rush)], presets=[PRESETS["punchy"]], annot_model=model,
            comparator_model="c", output_dir=str(tmp_path), thumbnail_width=width,
        )
        found = {}
        def walk(n):
            found.setdefault(n.name, n.key())
            for d in n.deps.values():
                walk(d)
        walk(g["candidates"])
        return found

    cloud = keys("claude-haiku-4-5-20251001", 640)
    local = keys("local/Qwen/Qwen3-VL-8B-Instruct", 640)

    assert cloud["scenes"] == local["scenes"], "la détection de plans ne doit pas bouger"
    assert cloud["metrics"] == local["metrics"], "les mesures locales ne doivent pas bouger"
    assert cloud["thumbs"] == local["thumbs"], "à largeur égale, les vignettes non plus"
    assert cloud["annot"] != local["annot"], "l'annotation doit être refaite"
    assert cloud["candidates"] != local["candidates"]

    wide = keys("local/Qwen/Qwen3-VL-8B-Instruct", 1280)
    assert wide["thumbs"] != local["thumbs"], "changer la largeur doit refaire les vignettes"
    assert wide["metrics"] == local["metrics"], "mais pas les mesures pleine résolution"


# ── Suivi coût/latence ──────────────────────────────────────────────────────


def test_local_calls_are_tracked_with_the_agent_name():
    """Régression : en déléguant au backend, le rapport `usage` avait perdu la
    ventilation par agent et affichait le nom de la classe de backend."""
    from src.agents.base_agent import BaseAgent
    from src.usage import usage

    class AnnotatorAgent(BaseAgent):
        pass

    with FakeServer("json_schema") as srv:
        import os

        os.environ["LOCAL_VLM_BASE_URL"] = f"http://127.0.0.1:{srv.port}/v1"
        usage.reset()
        try:
            AnnotatorAgent("local/fake-vlm").call("sys", "hi", Shot, 128)
        finally:
            os.environ.pop("LOCAL_VLM_BASE_URL", None)

    by_agent = usage.by_agent()
    assert "AnnotatorAgent" in by_agent, by_agent
    assert by_agent["AnnotatorAgent"]["calls"] == 1
    assert by_agent["AnnotatorAgent"]["cost"] == 0.0, "un modèle local ne coûte rien"


def test_local_usage_records_tokens_when_the_server_reports_them():
    from src.usage import usage

    class WithUsage(FakeServer):
        def respond(self, body):
            return 200, {
                "choices": [{"message": {"content": json.dumps(GOOD)}}],
                "usage": {"prompt_tokens": 1834, "completion_tokens": 57},
            }

    with WithUsage("json_schema") as srv:
        usage.reset()
        backend(srv).complete("sys", "hi", Shot, 256, agent="AnnotatorAgent")

    agg = usage.by_agent()["AnnotatorAgent"]
    assert agg["input_tokens"] == 1834 and agg["output_tokens"] == 57
    assert agg["latency_s"] > 0, "la latence est ce qu'on veut comparer au cloud"


def test_server_without_usage_field_does_not_crash():
    """Tous les serveurs locaux ne remontent pas `usage`."""
    from src.usage import usage

    with FakeServer("json_schema") as srv:
        usage.reset()
        backend(srv).complete("sys", "hi", Shot, 256, agent="X")

    assert usage.by_agent()["X"]["input_tokens"] == 0
