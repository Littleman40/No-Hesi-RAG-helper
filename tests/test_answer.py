import json

import answer

URL_A = "https://nohesifaq.gitbook.io/nohesifaq-docs/a#step-1"


def test_escalate_topic():
    assert answer.escalate_topic("Can I get a REFUND please") == "refund"
    assert answer.escalate_topic("how do I unlink steam from my account") == "steam_delink"
    assert answer.escalate_topic("game crashes on launch") is None


def test_check_citations():
    allowed = {URL_A}
    assert answer.check_citations(f"Do X.\nSource: {URL_A}", allowed) == f"Do X.\nSource: {URL_A}"
    assert answer.check_citations("Do X.\nSource: https://evil.example/fix", allowed) == "Do X."
    assert answer.check_citations(f"See {URL_A}.", allowed) == f"See {URL_A}."


def test_build_prompt_numbers_blocks():
    prompt = answer.build_prompt("q?", [{"url": URL_A, "text": "hello"}])
    assert f"[1] {URL_A}\nhello" in prompt
    assert prompt.endswith("Question: q?")


def fake_pipeline(monkeypatch, tmp_path, search_results, model_says):
    monkeypatch.setattr(answer, "LOG_FILE", tmp_path / "log.jsonl")
    monkeypatch.setattr(answer.retrieve, "search", lambda q, mode, k: search_results)
    monkeypatch.setattr(answer.retrieve, "get_chunk", lambda cid: {"url": URL_A, "text": "t"})
    monkeypatch.setattr(answer, "generate", lambda q, chunks: model_says)


def test_human_topic_skips_search(monkeypatch, tmp_path):
    fake_pipeline(monkeypatch, tmp_path, None, None)
    result = answer.answer("I want a refund")
    assert result["status"] == "escalated" and result["escalate"]


def test_nothing_above_floor_refuses(monkeypatch, tmp_path):
    fake_pipeline(monkeypatch, tmp_path, [], "should not be used")
    result = answer.answer("capital of France?")
    assert result["status"] == "refused" and result["reply"] == answer.REFUSAL_REPLY


def test_no_answer_refuses(monkeypatch, tmp_path):
    fake_pipeline(monkeypatch, tmp_path, [("c1", 0.03)], "NO_ANSWER")
    assert answer.answer("something odd")["status"] == "refused"


def test_answer_is_logged(monkeypatch, tmp_path):
    fake_pipeline(monkeypatch, tmp_path, [("c1", 0.03)], f"Do X.\nSource: {URL_A}")
    result = answer.answer("how do I X")
    assert result["status"] == "answered" and result["chunk_ids"] == ["c1"]
    assert '"question": "how do I X"' in (tmp_path / "log.jsonl").read_text()


def test_generate_calls_ollama(monkeypatch):
    sent = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({"message": {"content": "<think>hmm</think>\nDo X.\nSource: link"}}).encode()

    def fake_urlopen(request, timeout):
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data)
        return FakeResponse()

    monkeypatch.setattr(answer.urllib.request, "urlopen", fake_urlopen)
    assert answer.generate("q?", [{"url": URL_A, "text": "t"}]) == "Do X.\nSource: link"
    assert sent["url"].endswith("/api/chat")
    assert sent["body"]["options"]["temperature"] <= 0.2
    assert sent["body"]["messages"][0]["content"] == answer.SYSTEM_PROMPT


def test_escalate_topic_matches_words_in_any_order():
    assert answer.escalate_topic("how can i unlink a steam account") == "steam_delink"
    assert answer.escalate_topic("my steam is already hooked upto another account") == "steam_delink"
    assert answer.escalate_topic("i want my money back") == "refund"
    assert answer.escalate_topic("how do i link my steam account") is None
