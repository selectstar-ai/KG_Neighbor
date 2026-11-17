import re

import pytest

from kg_neighbor.idp import AnthropicBatch, OpenAIBatch
from kg_neighbor.idp.prompts import RESPONSE_SCHEMA, SYSTEM_PROMPT

Q = {"key": "sample 1|colt's foot", "sample_id": "sample 1", "class_name": "colt's foot", "prompt": "is it?"}
ANSWER = '{"rationale": "r", "valid": true}'


def anthropic_row(rid, rtype="succeeded", stop="end_turn"):
    msg = {"stop_reason": stop, "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": ANSWER}]}
    return {"custom_id": rid, "result": {"type": rtype, "message": msg}}


def test_openai_request_keeps_paper_settings():
    r = OpenAIBatch().build_request(Q)
    assert r["custom_id"] == Q["key"] and r["url"] == "/v1/chat/completions"
    b = r["body"]
    assert (b["model"], b["temperature"], b["reasoning_effort"]) == ("gpt-5.4", 0.0, "none")
    assert b["response_format"]["json_schema"]["schema"] == RESPONSE_SCHEMA


def test_anthropic_request_has_schema_and_valid_id():
    r = AnthropicBatch().build_request(Q)
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", r["custom_id"])
    p = r["params"]
    assert p["model"] == "claude-opus-5-5" and p["system"] == SYSTEM_PROMPT
    assert p["output_config"] == {"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}, "effort": "low"}
    assert p["messages"] == [{"role": "user", "content": [{"type": "text", "text": "is it?"}]}]


def test_anthropic_image_goes_before_text(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "x.png"
    Image.new("RGB", (1000, 500)).save(path)
    content = AnthropicBatch().build_request({**Q, "image_path": str(path)})["params"]["messages"][0]["content"]
    assert content[0]["type"] == "image" and content[0]["source"]["media_type"] == "image/jpeg"
    assert content[1] == {"type": "text", "text": "is it?"}


def test_anthropic_parse_keeps_finished_answers_only():
    rows = [anthropic_row("ok"), anthropic_row("err", rtype="errored"),
            anthropic_row("refused", stop="refusal"), anthropic_row("cut", stop="max_tokens")]
    assert AnthropicBatch.parse(rows) == {"ok": {"rationale": "r", "valid": True}}


def test_run_returns_answers_by_question_key():
    class Fake(AnthropicBatch):
        def submit(self, requests):
            self.sent = requests
            return ["job"]

        def wait(self, job_ids):
            return [anthropic_row(r["custom_id"]) for r in self.sent]

    assert Fake().run([Q]) == {Q["key"]: {"rationale": "r", "valid": True}}


def test_settings_name_the_judge_and_its_configuration():
    assert OpenAIBatch().settings() == {
        "judge": "openai", "model": "gpt-5.4", "temperature": 0.0, "reasoning_effort": "none"}
    assert AnthropicBatch("m").settings() == {"judge": "anthropic", "model": "m", "effort": "low", "max_tokens": 16000}
