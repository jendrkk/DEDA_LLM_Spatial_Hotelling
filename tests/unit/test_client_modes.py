from hotelling.llm.client import LLMClient


def test_client_builds_json_mode():
    c = LLMClient(model="gemini/gemma-4-31b-it", instructor_mode="json",
                  reasoning_effort="disable", max_tokens=2048)
    assert c.instructor_mode == "json"
    assert c.reasoning_effort == "disable"
    import importlib.util
    if importlib.util.find_spec("instructor") and importlib.util.find_spec("litellm"):
        assert c._client is not None
