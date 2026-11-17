"""Batch API judges (Section 4.2). Requires the ``llm`` extra.

:class:`IDPBatch` holds what every judge shares; the subclasses differ in request
format and job handling. Both use :data:`~kg_neighbor.idp.prompts.SYSTEM_PROMPT`
and enforce :data:`~kg_neighbor.idp.prompts.RESPONSE_SCHEMA`.

- :class:`OpenAIBatch` (paper): GPT-5.4, temperature 0, reasoning effort ``none``.
  Key from ``OPENAI_API_KEY``.
- :class:`AnthropicBatch`: Claude Opus 5.5 at the lowest effort (it always thinks,
  so this is the closest to the paper's "no reasoning"). Key from ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import time

from kg_neighbor.idp.prompts import RESPONSE_SCHEMA, SYSTEM_PROMPT


class IDPBatch:
    """Submit questions as batch jobs, wait for them, and parse the answers.

    Subclasses implement :meth:`settings`, :meth:`build_request`, :meth:`submit`,
    :meth:`wait`, :meth:`parse` and :meth:`_new_client`.

    Parameters
    ----------
    model : judge model
    client : API client; created on first use if omitted
    poll_interval : seconds between job status checks
    max_image_dim, image_quality : images are shrunk to this longest side and
        re-encoded as JPEG to stay under the batch size limit
    """

    def __init__(
        self,
        model: str,
        *,
        client=None,
        poll_interval: float = 30.0,
        max_image_dim: int = 768,
        image_quality: int = 85,
    ):
        self.model = model
        self._client = client
        self.poll_interval = poll_interval
        self.max_image_dim = max_image_dim
        self.image_quality = image_quality

    @property
    def client(self):
        if self._client is None:
            self._client = self._new_client()
        return self._client

    def _new_client(self):
        raise NotImplementedError

    def settings(self) -> dict:
        """Judge and its configuration; answers made with other settings are kept apart."""
        raise NotImplementedError

    def _image_b64(self, path: str) -> str:
        """Base64 JPEG of the image, shrunk to ``max_image_dim``."""
        from PIL import Image

        img = Image.open(path).convert("RGB")
        img.thumbnail((self.max_image_dim, self.max_image_dim))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=self.image_quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def request_id(key: str) -> str:
        """Batch request id of a question key."""
        return key

    def build_request(self, question: dict) -> dict:
        """One batch request for a question from :func:`~kg_neighbor.idp.judge.make_questions`."""
        raise NotImplementedError

    def submit(self, requests: list[dict]) -> list[str]:
        """Start the jobs; returns job ids."""
        raise NotImplementedError

    def wait(self, job_ids: list[str]) -> list[dict]:
        """Block until every job ends; returns the result rows."""
        raise NotImplementedError

    @staticmethod
    def parse(rows: list[dict]) -> dict[str, dict]:
        """{request id: {"rationale": str, "valid": bool}}; failed requests are skipped."""
        raise NotImplementedError

    def run(self, questions: list[dict]) -> dict[str, dict]:
        """Submit, wait and parse in one call; answers are keyed by question key."""
        if not questions:
            return {}
        keys = {self.request_id(q["key"]): q["key"] for q in questions}
        answers = self.parse(self.wait(self.submit([self.build_request(q) for q in questions])))
        return {keys[i]: a for i, a in answers.items()}


class OpenAIBatch(IDPBatch):
    """OpenAI Batch API judge (paper configuration).

    Parameters
    ----------
    model, temperature, reasoning_effort : judge configuration
    completion_window : Batch API completion window
    **kwargs : see :class:`IDPBatch`
    """

    MAX_REQUESTS_PER_BATCH = 50_000  # Batch API limit per job

    def __init__(
        self,
        model: str = "gpt-5.4",
        *,
        temperature: float = 0.0,
        reasoning_effort: str | None = "none",
        completion_window: str = "24h",
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.completion_window = completion_window

    def _new_client(self):
        from openai import OpenAI

        return OpenAI()

    def settings(self) -> dict:
        return {"judge": "openai", "model": self.model, "temperature": self.temperature,
                "reasoning_effort": self.reasoning_effort}

    def build_request(self, question: dict) -> dict:
        content = question["prompt"]
        if question.get("image_path"):
            url = "data:image/jpeg;base64," + self._image_b64(question["image_path"])
            content = [
                {"type": "text", "text": question["prompt"]},
                {"type": "image_url", "image_url": {"url": url}},
            ]
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": self.temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "strict": True, "schema": RESPONSE_SCHEMA},
            },
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        return {
            "custom_id": self.request_id(question["key"]),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

    def submit(self, requests: list[dict]) -> list[str]:
        """Upload and start one job per ``MAX_REQUESTS_PER_BATCH`` requests."""
        n = self.MAX_REQUESTS_PER_BATCH
        job_ids = []
        for start in range(0, len(requests), n):
            chunk = requests[start : start + n]
            buf = io.BytesIO("\n".join(json.dumps(r, ensure_ascii=False) for r in chunk).encode())
            buf.name = "batch_input.jsonl"
            upload = self.client.files.create(file=buf, purpose="batch")
            batch = self.client.batches.create(
                input_file_id=upload.id,
                endpoint="/v1/chat/completions",
                completion_window=self.completion_window,
            )
            job_ids.append(batch.id)
        return job_ids

    def wait(self, job_ids: list[str]) -> list[dict]:
        rows = []
        for job_id in job_ids:
            while True:
                batch = self.client.batches.retrieve(job_id)
                if batch.status == "completed":
                    break
                if batch.status in ("failed", "expired", "cancelled"):
                    raise RuntimeError(f"batch {job_id} ended with status {batch.status}")
                time.sleep(self.poll_interval)
            text = self.client.files.content(batch.output_file_id).text
            rows.extend(json.loads(line) for line in text.splitlines() if line.strip())
        return rows

    @staticmethod
    def parse(rows: list[dict]) -> dict[str, dict]:
        out = {}
        for row in rows:
            resp = row.get("response")
            if resp is None or resp.get("status_code") not in (None, 200):
                continue
            out[row["custom_id"]] = json.loads(resp["body"]["choices"][0]["message"]["content"])
        return out


class AnthropicBatch(IDPBatch):
    """Anthropic Message Batches API judge.

    Parameters
    ----------
    model, effort, max_tokens : judge configuration (thinking counts toward ``max_tokens``)
    **kwargs : see :class:`IDPBatch`
    """

    MAX_REQUESTS_PER_BATCH = 100_000  # API limit per batch
    MAX_BYTES_PER_BATCH = 200 * 2**20  # API limit is 256 MB

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        *,
        effort: str | None = "low",
        max_tokens: int = 16000,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.effort = effort
        self.max_tokens = max_tokens

    def _new_client(self):
        import anthropic

        return anthropic.Anthropic()

    def settings(self) -> dict:
        return {"judge": "anthropic", "model": self.model, "effort": self.effort, "max_tokens": self.max_tokens}

    @staticmethod
    def request_id(key: str) -> str:
        """Ids allow only letters, digits, ``_`` and ``-``; question keys carry ``|`` and class names."""
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    def build_request(self, question: dict) -> dict:
        content = [{"type": "text", "text": question["prompt"]}]
        if question.get("image_path"):
            image = {"type": "base64", "media_type": "image/jpeg", "data": self._image_b64(question["image_path"])}
            content.insert(0, {"type": "image", "source": image})
        output_config = {"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}}
        if self.effort:
            output_config["effort"] = self.effort
        return {
            "custom_id": self.request_id(question["key"]),
            "params": {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": content}],
                "output_config": output_config,
            },
        }

    def submit(self, requests: list[dict]) -> list[str]:
        """Start one batch per request-count or size limit."""
        job_ids, chunk, size = [], [], 0
        for r in requests:
            n = len(json.dumps(r, ensure_ascii=False).encode())
            if chunk and (len(chunk) == self.MAX_REQUESTS_PER_BATCH or size + n > self.MAX_BYTES_PER_BATCH):
                job_ids.append(self.client.messages.batches.create(requests=chunk).id)
                chunk, size = [], 0
            chunk.append(r)
            size += n
        if chunk:
            job_ids.append(self.client.messages.batches.create(requests=chunk).id)
        return job_ids

    def wait(self, job_ids: list[str]) -> list[dict]:
        rows = []
        for job_id in job_ids:
            while self.client.messages.batches.retrieve(job_id).processing_status != "ended":
                time.sleep(self.poll_interval)
            rows.extend(r.to_dict(mode="json") for r in self.client.messages.batches.results(job_id))
        return rows

    @staticmethod
    def parse(rows: list[dict]) -> dict[str, dict]:
        """Errored, expired, refused and truncated requests are skipped, so a rerun asks them again."""
        out = {}
        for row in rows:
            result = row.get("result") or {}
            if result.get("type") != "succeeded" or result["message"].get("stop_reason") != "end_turn":
                continue
            text = next((b["text"] for b in result["message"]["content"] if b["type"] == "text"), None)
            if text is not None:
                out[row["custom_id"]] = json.loads(text)
        return out
