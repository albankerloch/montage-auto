from __future__ import annotations
import json
import time
import anthropic
from pydantic import BaseModel

from src.usage import usage


def _flatten_schema(schema: dict) -> dict:
    """
    Resolve $defs/$ref in a JSON schema so Claude sees a flat schema
    without nested references — Claude handles these poorly.
    """
    defs = schema.pop("$defs", {})
    if not defs:
        return schema

    def resolve(obj):
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].split("/")[-1]
                resolved = resolve(defs.get(ref_name, {}))
                return resolved
            return {k: resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve(i) for i in obj]
        return obj

    return resolve(schema)


def _coerce_json_strings(obj):
    """
    Claude sometimes serializes arrays/objects as JSON strings inside the tool input.
    Recursively parse any string values that look like JSON arrays or objects.
    """
    if isinstance(obj, dict):
        return {k: _coerce_json_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_json_strings(i) for i in obj]
    if isinstance(obj, str):
        stripped = obj.strip()
        if stripped.startswith(("{", "[")):
            try:
                return _coerce_json_strings(json.loads(stripped))
            except json.JSONDecodeError:
                pass
    return obj


class BaseAgent:
    def __init__(self, model: str):
        self.client = anthropic.Anthropic()
        self.model = model

    def call(
        self,
        system: str,
        user_content: str | list,
        output_schema: type[BaseModel],
        max_tokens: int = 4096,
        max_retries: int = 4,
    ) -> BaseModel:
        """Structured output call using tools as JSON schema enforcer.
        Retries on 529 overload and 529/rate-limit errors with exponential backoff."""
        schema = _flatten_schema(output_schema.model_json_schema())
        schema.pop("title", None)

        last_error = None
        for attempt in range(max_retries):
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
                # Le temps d'attente d'un retry (backoff) n'est pas de la
                # latence d'appel, et une tentative qui échoue n'est pas
                # facturée : on ne chronomètre/journalise que l'appel réussi.
                usage.record(
                    agent=self.__class__.__name__,
                    model=self.model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    latency_s=time.perf_counter() - t0,
                )
                break  # success
            except (anthropic.OverloadedError, anthropic.RateLimitError) as e:
                last_error = e
                wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s
                print(f"[{self.__class__.__name__}] API overloaded/rate-limited, retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
        else:
            raise last_error

        for block in response.content:
            if block.type == "tool_use" and block.name == "structured_output":
                data = _coerce_json_strings(block.input)
                try:
                    return output_schema.model_validate(data)
                except Exception as e:
                    raise RuntimeError(
                        f"Agent {self.__class__.__name__} returned invalid data: {e}\n"
                        f"Raw input: {json.dumps(block.input, indent=2)}"
                    )

        raise RuntimeError(
            f"Agent {self.__class__.__name__} did not return structured output. "
            f"Response: {[b.type for b in response.content]}"
        )
