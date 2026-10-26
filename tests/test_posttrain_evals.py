"""posttrain.evals: transcript prompts, GSM8K formatting/extraction, and a full run against a fake server."""

from __future__ import annotations

import asyncio
import json

import pytest

from kotodama.posttrain import evals


def test_prompt_is_a_user_assistant_transcript() -> None:
    p = evals.prompt_for([("2+2?", "4. The answer is 4.")], "3+3?")
    assert p == "User: 2+2?\n\nAssistant: 4. The answer is 4.\n\nUser: 3+3?\n\nAssistant:"
    assert evals.cut_reply(" six.\n\nUser: thanks") == "six."
    assert evals.cut_reply(" six.\nUser: thanks") == "six."


def test_gsm8k_solution_and_extraction() -> None:
    sol, final = evals.gsm8k_solution("She has 3 + 4 = <<3+4=7>>7 apples.\n#### 1,007")
    assert final == "1007" and sol == "She has 3 + 4 = 7 apples.\nThe answer is 1007."
    assert evals.gsm8k_extract("so 12 then 15. The answer is $1,240.") == "1240"
    assert evals.gsm8k_extract("The answer is 24\nQ: next one. The answer is 800") == "24"
    assert evals.gsm8k_extract("first 3, then 7 left") == "7"
    assert evals.gsm8k_extract("no idea") is None
    assert evals._same_number("18.0", "18") and not evals._same_number(None, "1")


def _fake_transport():
    import httpx

    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        q = body["prompt"].rsplit("User: ", 1)[1]
        text = " The answer is 5.\n\nUser: next" if "five" in q else " I think 9."
        return httpx.Response(200, json={"choices": [{"text": text, "index": 0}]})

    return httpx.MockTransport(handler), seen


def test_gsm8k_run_against_a_fake_server(tmp_path) -> None:
    pytest.importorskip("httpx")
    transport, seen = _fake_transport()
    train = [{"question": f"q{i}", "answer": f"step {i}\n#### {i}"} for i in range(10)]
    test = [{"question": "what is five?", "answer": "#### 5"}, {"question": "what is ten?", "answer": "#### 10"}]
    summary = asyncio.run(evals.run("http://fake", ["gsm8k"], tmp_path, 0, 2, transport=transport,
                                    data={"gsm8k": {"test": test, "train": train}}))
    assert summary["gsm8k_exact_match"] == 0.5 and summary["gsm8k_n"] == 2
    body = seen[0]
    assert body["temperature"] == 0.0 and body["stop"] == ["\n\nUser:"] and body["frequency_penalty"] == 0.0
    assert body["prompt"].count("\n\nAssistant: ") == evals.GSM8K_SHOTS  # eight worked exchanges before the question
    rows = [json.loads(x) for x in (tmp_path / "gsm8k.jsonl").read_text().splitlines()]
    assert rows[0]["reply"] == "The answer is 5." and rows[0]["correct"]
    assert json.loads((tmp_path / "summary.json").read_text())["gsm8k_n"] == 2


def test_ifeval_scoring_against_a_fake_server(tmp_path) -> None:
    pytest.importorskip("lm_eval.tasks.ifeval.utils")
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"text": " hello world, all lowercase here."}]})

    doc = {"key": 1, "prompt": "Reply in all lowercase letters.", "instruction_id_list": ["change_case:english_lowercase"],
           "kwargs": [{}]}
    summary = asyncio.run(evals.run("http://fake", ["ifeval"], tmp_path, 0, 1, transport=httpx.MockTransport(handler),
                                    data={"ifeval": {"docs": [doc]}}))
    assert summary["ifeval_prompt_strict"] == 1.0 and summary["ifeval_inst_loose"] == 1.0
