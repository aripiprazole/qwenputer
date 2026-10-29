"""A tiny Qwen3.5 fixture checks logits and recurrent/KV caches after projection changes."""

import copy
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from chipformer.rv import chat as chat_module
from chipformer.rv.chat import FirmwareState, RvChat


@pytest.fixture
def chat():
    cfg = Qwen3_5TextConfig(
        vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"], num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        rope_parameters={"rope_type": "default", "rope_theta": 10000,
                         "partial_rotary_factor": 1.0, "mrope_section": [2, 3, 3],
                         "mrope_interleaved": True}, tie_word_embeddings=False,
    )
    with torch.random.fork_rng():
        torch.manual_seed(3)
        model = Qwen3_5ForCausalLM(cfg).eval()
    result = object.__new__(RvChat)
    result.model, result.lm = model, model.model
    result.v0, result.n_chip, result.dev = 64, 0, "cpu"
    return result


def assert_same_cache(actual, expected):
    for a, b in zip(actual.layers, expected.layers, strict=True):
        for field in ("conv_states", "recurrent_states", "keys", "values"):
            x, y = getattr(a, field, None), getattr(b, field, None)
            if isinstance(x, dict):
                assert x.keys() == y.keys()
                for key in x:
                    torch.testing.assert_close(x[key], y[key], rtol=0, atol=0)
            elif isinstance(x, torch.Tensor):
                torch.testing.assert_close(x, y, rtol=0, atol=0)


def track_head(monkeypatch, chat):
    calls = []
    original = torch.nn.functional.linear
    pointer = chat.model.lm_head.weight.data_ptr()

    def linear(x, weight, bias=None):
        if weight.data_ptr() == pointer:
            calls.append((tuple(x.shape), tuple(weight.shape)))
        return original(x, weight, bias)

    monkeypatch.setattr(torch.nn.functional, "linear", linear)
    return calls


@torch.no_grad()
def test_prefill_projects_only_last_text_position_and_preserves_cache(chat, monkeypatch):
    ids = torch.tensor([[1, 3, 5, 7, 9]])
    baseline = chat.model(input_ids=ids, use_cache=True)
    calls = track_head(monkeypatch, chat)
    cache, logits = chat._prefill(ids)
    assert calls == [((1, 1, 64), (64, 64))]
    torch.testing.assert_close(logits, baseline.logits[0, -1, :chat.v0])
    assert logits.argmax() == baseline.logits[0, -1, :chat.v0].argmax()
    assert_same_cache(cache, baseline.past_key_values)


@pytest.mark.parametrize("full_head", [False, True])
@pytest.mark.parametrize("n_chip", [0, 1])
@torch.no_grad()
def test_text_decode_preserves_logits_and_both_cache_types(chat, monkeypatch, full_head, n_chip):
    chat.n_chip = n_chip
    prompt = torch.tensor([[1, 3, 5, 7]])
    cache = chat.model(input_ids=prompt, use_cache=True).past_key_values
    expected_cache = copy.deepcopy(cache)
    token = torch.tensor([[9]])
    expected = chat.model(input_ids=token, past_key_values=expected_cache, use_cache=True)
    calls = track_head(monkeypatch, chat)
    hc = (chat._layers(token.item(), cache, None)[0] if n_chip
          else chat.lm.embed_tokens(token)[0, 0])
    logits = chat._qwen_step(hc, cache, prompt.shape[1], full_head=full_head)
    width = chat.model.config.vocab_size if full_head else chat.v0
    assert calls == [((1, 1, 64), (width, 64))]
    torch.testing.assert_close(logits, expected.logits[0, -1, :width])
    assert_same_cache(cache, expected.past_key_values)


@torch.no_grad()
def test_circuit_owned_echo_still_updates_language_cache_without_head(chat, monkeypatch):
    chat.n_chip = 1
    ids = torch.tensor([[1, 3, 5, 7]])
    cache = chat.model(input_ids=ids, use_cache=True).past_key_values
    expected_cache = copy.deepcopy(cache)
    token = torch.tensor([[9]])
    chat.model(input_ids=token, past_key_values=expected_cache, use_cache=True)
    calls = track_head(monkeypatch, chat)
    hc, _ = chat._layers(token.item(), cache, None)
    result = chat._qwen_step(hc, cache, ids.shape[1], project=False)
    assert result is None and calls == []
    assert_same_cache(cache, expected_cache)


@torch.no_grad()
def test_firmware_capture_owns_its_state_and_restores_into_a_new_cache(chat):
    ids = torch.tensor([[1, 3, 5, 7]])
    cache = chat.model(input_ids=ids, use_cache=True).past_key_values
    saved = FirmwareState.capture(cache, 1)
    expected = cache.layers[0].recurrent_states[0].clone()
    cache.layers[0].recurrent_states[0].zero_()
    torch.testing.assert_close(saved.heads[0], expected, rtol=0, atol=0)
    target = chat.model(input_ids=torch.tensor([[9]]), use_cache=True).past_key_values
    saved.inject(target)
    torch.testing.assert_close(target.layers[0].recurrent_states[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("check", [False, True])
def test_reference_execution_requires_explicit_check_flag(monkeypatch, check):
    requested = []
    histories = []
    fake = SimpleNamespace(
        T=object(), prompt_ids=lambda *_: [1, 2], text_of=lambda _: "reply",
        boot=lambda _: object(),
    )

    def generate(*args, **kwargs):
        requested.append(kwargs)
        return {"tokens": [3], "text_tokens": 1, "chip_tokens": 0, "seconds": 0.0,
                "head_mismatch": 0, "lock_error": None}

    fake.generate = generate
    monkeypatch.setattr(chat_module, "RvChat", lambda *_args, **_kwargs: fake)
    monkeypatch.setattr(chat_module.elf, "program", lambda _: object())
    monkeypatch.setattr(chat_module.ref.Machine, "from_program", lambda _: object())
    def park(_):
        assert check, "normal chat must boot through the transformer"

    monkeypatch.setattr(chat_module.BR, "park", park)

    def session(*args):
        assert check, "normal chat must not run a shadow reference interpreter"
        return SimpleNamespace(history=histories.append)

    monkeypatch.setattr(chat_module.BR, "Session", session)
    monkeypatch.setattr(chat_module, "replay", lambda *_: {"ok": True})
    monkeypatch.setattr("sys.argv", ["chat", "fixture"] + (["--check"] if check else []))
    chat_module.main()
    assert requested[0]["check_head"] is check
    assert (requested[0]["lock"] is not None) is check
    assert histories == ([[1, 2]] if check else [])
