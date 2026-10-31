"""Talk to a merged rv1 checkpoint: Qwen writes, the RV32 machine runs inside the same model.

    python -m chipformer.rv.chat dist/q4-rv1 --firmware tests/data/rv/c-calc.elf \\
        --ask "What is 123456789 * 987654321?"

Decoding is greedy and transparent to chip tokens (as BatchDecoder's fp + tc): a chip token
runs through the chip layers only, so Qwen's layers, caches and RoPE positions see the text
alone; a text token runs through every layer. At every position the next token is the
circuit's when its output code is set (the owner), else Qwen's argmax over its own vocabulary.
Normal chat boots the firmware through the HF chip layers, parks at its first exit ecall,
and restores that recurrent-state snapshot after each prompt's prefill. ``--check`` enables
the reference bootstrap, lockstep validation and replay.

``--check`` replays the whole stream through the IR evaluator and the reference machine: every
circuit-owned token must equal the IR's prediction and the reference's next token.
"""

import argparse
from dataclasses import dataclass
import json
import time
from pathlib import Path

import numpy as np
import torch

from chipformer.rv import bridge as BR
from chipformer.rv import core, elf, hf, ref
from chipformer.rv import spec as S
from chipformer.rv.decode import CodeDecoder

CALC_TOOL = {"type": "function", "function": {
    "name": "calc",
    "description": "Evaluate an arithmetic expression exactly, with numbers of up to 62 digits: "
                   "+ - * / and parentheses on integers and fractions (exact: the result is an "
                   "integer or a reduced fraction p/q), + - * on decimals, and on integers also "
                   "// (floor division), %, ** and gcd(a,b). Returns the exact value.",
    "parameters": {"type": "object", "properties": {
        "expression": {"type": "string",
                       "description": "e.g. 123456789*987654321-42 or 1.25*-3.5 or 84/18"}},
        "required": ["expression"]}}}


@dataclass(frozen=True)
class FirmwareState:
    """Parked firmware produced by the HF chip layers, reusable across conversations."""

    heads: tuple[torch.Tensor, ...]
    steps: int

    @classmethod
    def capture(cls, cache, layers, steps=0):
        return cls(tuple(cache.layers[i].recurrent_states[0].clone() for i in range(layers)), steps)

    def inject(self, cache):
        for index, state in enumerate(self.heads):
            target = cache.layers[index].recurrent_states[0]
            if target.shape != state.shape:
                raise ValueError("firmware state does not match the model's chip layers")
            target.copy_(state)


@dataclass(frozen=True)
class ExecState:
    """Parked firmware on the executor: every circuit head's state (as FirmwareState holds the
    HF chip layers' recurrent states)."""

    heads: tuple[torch.Tensor, ...]
    steps: int


class ExecRow:
    """One conversation's circuit on the Triton executor (batch 1) with its own CUDA graph.

    Runs of chip tokens go through a device token loop: BLOCK circuit steps captured as one
    graph, each step's predicted token embedded on the device as the next step's input, so the
    host syncs once per block instead of once per token. A block stops being valid at the first
    prediction outside the chip vocabulary (an echoed Qwen token, or no owner): the heads are
    put back to the block's start and only the valid steps are replayed one by one."""

    BLOCK = 256

    def __init__(self, chat: "RvChat") -> None:
        from chipformer.rv.tritonexec import Executor2

        self.chat = chat
        cuda = torch.device(chat.dev).type == "cuda"
        # programmatic dependent launch (griddepcontrol) needs sm_90 or newer
        pdl = cuda and torch.cuda.get_device_capability(chat.dev)[0] >= 9
        self.ex = Executor2(chat.c, 1, chat.dev, fuse=chat.fuse, pdl=pdl, early=pdl,
                            direct=cuda)
        if cuda:
            self._tune()
        self.heads = [st for st in self.ex.steps if st["kind"] == "head"]
        self.xbuf = torch.zeros(1, self.ex.D, device=chat.dev)
        self.kbuf = torch.zeros(len(chat.keep_cols), device=chat.dev)
        self.graph = None
        self.loop = None
        self.fused = cuda  # the device loop's decode + embedding as one Triton kernel

    def _tune(self) -> None:
        """Each step's launch config: tuned once per GPU and map (compiling every candidate
        kernel takes long on a fresh Triton cache), then read back from next to the checkpoint.
        QWENPUTER_TUNE=0 keeps the default configs."""
        import os

        if os.environ.get("QWENPUTER_TUNE") == "0":
            return
        gpu = torch.cuda.get_device_name(self.chat.dev).replace(" ", "_")
        cache = self.chat.path / f"exec_tune.{S.MAP}.{gpu}.json"
        if cache.is_file():
            cfgs = json.loads(cache.read_text())
            if len(cfgs) == len(self.ex.steps):
                for st, cfg in zip(self.ex.steps, cfgs):
                    st["cfg"] = tuple(cfg)
                return
        chosen = self.ex.tune(self.ex.embed([S.tok("GO")]))
        try:
            cache.write_text(json.dumps([list(c) for c in chosen]))
        except OSError:
            pass

    def save(self, steps: int = 0) -> ExecState:
        return ExecState(tuple(st["S"][0].clone() for st in self.heads), steps)

    def load(self, state: ExecState) -> None:
        if len(state.heads) != len(self.heads):
            raise ValueError("firmware state does not match the circuit")
        for st, s in zip(self.heads, state.heads):
            st["S"][0].copy_(s)

    @torch.no_grad()
    def _build_loop(self) -> None:
        from chipformer.dsl.graph import ONE

        chat, dev = self.chat, self.chat.dev
        self.tab = chat.chip_table()  # [chip vocab, features] in ex.feat_ids order
        self.fidx = torch.tensor(self.ex.feat_ids, device=dev)
        self.ncode = len(chat.code_dims)
        self.cur = torch.zeros(1, dtype=torch.long, device=dev)  # the input's chip index
        self.out = torch.zeros(self.BLOCK, dtype=torch.long, device=dev)  # predicted model ids
        self.backup = [(st["S"].clone(), st["hist"].clone()) for st in self.heads]
        last = self.tab.shape[0] - 1

        code_ix = chat.keep_ir[: self.ncode].contiguous()
        tab = self.tab.contiguous()

        def body():
            if self.fused:  # one kernel between steps (Executor2.next_input)
                args = dict(code_ix=code_ix, out=self.out, cur=self.cur, tab=tab,
                            fidx=self.fidx, decoder=chat.decoder, v0=chat.v0, one=ONE)
                self.ex.next_input(self.xbuf, i=0, decode=False, **args)
                for i in range(self.BLOCK):
                    self.ex.step(self.xbuf)
                    self.ex.next_input(self.xbuf, i=i, **args)
                return
            for i in range(self.BLOCK):
                self.xbuf.zero_()
                self.xbuf[0, ONE:ONE + 1].fill_(1.0)  # (a kernel; no host copy in the graph)
                self.xbuf[0].index_copy_(0, self.fidx, torch.index_select(self.tab, 0, self.cur)[0])
                self.ex.step(self.xbuf)
                code = torch.index_select(self.xbuf[0], 0, chat.keep_ir[: self.ncode])
                nxt = chat.decoder(code)
                self.out[i] = nxt
                self.cur.copy_((nxt - chat.v0).clamp(0, last).view(1))

        saved = [(st["S"].clone(), st["hist"].clone()) for st in self.heads]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body()
        torch.cuda.current_stream().wait_stream(s)
        self.loop = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.loop):
            body()
        for st, (S_, h) in zip(self.heads, saved):
            st["S"].copy_(S_)
            st["hist"].copy_(h)

    @torch.no_grad()
    def run_chip(self, first: int, limit: int, on_block=None, stop=None) -> tuple[list, int]:
        """Feed the chip token ``first`` and then the circuit's own predictions while they stay
        chip tokens. Returns (the inputs fed, starting with first; the last prediction): a
        Qwen token or -1 / -2 when the run ended, a chip token when ``limit`` inputs were fed
        or ``stop`` was set. ``on_block(ids)`` sees each block's inputs as it completes."""
        chat = self.chat
        if self.graph is None:
            self._capture()
        if self.loop is None:
            self._build_loop()
        if self.fused:
            return self._run_chip_ahead(first, limit, on_block, stop)
        inputs, head = [], first
        while True:
            self.cur.fill_(head - chat.v0)
            for st, (bs, bh) in zip(self.heads, self.backup):
                bs.copy_(st["S"])
                bh.copy_(st["hist"])
            self.loop.replay()
            preds = self.out.tolist()
            end = next((i for i, t in enumerate(preds) if t < chat.v0), None)
            if end is None:
                block = [head] + preds[:-1]
                inputs += block
                if on_block:
                    on_block(block)
                head = preds[-1]
                if len(inputs) >= limit or (stop is not None and stop.is_set()):
                    return inputs, head
                continue
            # the steps after `end` read a token that is not a chip token: undo the block and
            # run its valid steps again (the same inputs, so the same predictions)
            for st, (bs, bh) in zip(self.heads, self.backup):
                st["S"].copy_(bs)
                st["hist"].copy_(bh)
            block = [head] + preds[:end]
            for t in block:
                keep = self.step(t)
            last = chat.code_token(keep[: self.ncode])
            if (last if last is not None else -1) != preds[end]:
                raise RuntimeError("the replayed block diverged from the device loop")
            inputs += block
            if on_block:
                on_block(block)
            return inputs, preds[end]

    def _keep(self, slot) -> None:
        for st, (bs, bh) in zip(self.heads, self.backups[slot]):
            bs.copy_(st["S"])
            bh.copy_(st["hist"])

    def _restore(self, slot) -> None:
        for st, (bs, bh) in zip(self.heads, self.backups[slot]):
            st["S"].copy_(bs)
            st["hist"].copy_(bh)

    def _launch_block(self, k: int) -> None:
        """Block k on the stream: its start state into backup k % 2, the graph, and its
        predictions on their way to a pinned host buffer (no host sync)."""
        self._keep(k % 2)
        self.loop.replay()
        self.host_out[k % 2].copy_(self.out, non_blocking=True)
        self.done[k % 2].record()

    def _run_chip_ahead(self, first, limit, on_block, stop):
        """run_chip with the next block launched before this one's predictions are read: the
        graph's last kernel leaves the next input (cur) on the device, so block k + 1 runs
        while the host checks block k. A block that ends the run discards the one after it:
        the state goes back to block k's start (its backup) and the valid steps run again."""
        chat = self.chat
        if not hasattr(self, "backups"):
            self.backups = [self.backup, [(S_.clone(), h.clone()) for S_, h in self.backup]]
            self.host_out = [torch.zeros(self.BLOCK, dtype=torch.long).pin_memory()
                             for _ in range(2)]
            self.done = [torch.cuda.Event() for _ in range(2)]
        inputs, head = [], first
        self.cur.fill_(head - chat.v0)
        self._launch_block(0)
        k = 0
        while True:
            ahead = len(inputs) + 2 * self.BLOCK <= limit and not (stop is not None
                                                                    and stop.is_set())
            if ahead:
                self._launch_block(k + 1)
            self.done[k % 2].synchronize()
            preds = self.host_out[k % 2].tolist()
            end = next((i for i, t in enumerate(preds) if t < chat.v0), None)
            if end is None:
                block = [head] + preds[:-1]
                inputs += block
                if on_block:
                    on_block(block)
                head = preds[-1]
                if not ahead:  # the limit or a stop: nothing runs past this block
                    return inputs, head
                k += 1
                continue
            torch.cuda.synchronize()  # the block after it (if launched) is done: discard it
            self._restore(k % 2)
            block = [head] + preds[:end]
            for t in block:
                keep = self.step(t)
            last = chat.code_token(keep[: self.ncode])
            if (last if last is not None else -1) != preds[end]:
                raise RuntimeError("the replayed block diverged from the device loop")
            inputs += block
            if on_block:
                on_block(block)
            return inputs, preds[end]

    def _body(self):
        self.ex.step(self.xbuf)
        self.kbuf.copy_(self.xbuf[0, self.chat.keep_ir])

    @torch.no_grad()
    def _capture(self):
        """Warm-up runs change the state, so it is saved and put back around them."""
        saved = [(st["S"].clone(), st["hist"].clone()) for st in self.heads]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._body()
        for st, (S_, h) in zip(self.heads, saved):
            st["S"].copy_(S_)
            st["hist"].copy_(h)

    @torch.no_grad()
    def start(self, ids: list[int], state: ExecState | None = None) -> None:
        """As the HF path does it: the prompt through the circuit from zero states, then every
        head's state set to its init (the snapshot now in the circuit; hf.inject) or to a
        parked ``state``, the lag windows kept."""
        if self.graph is None:
            self._capture()
        for st in self.heads:
            st["S"].zero_()
            st["hist"].zero_()
        emb = self.chat._exec_emb(ids)
        for t in range(len(ids)):
            self.xbuf.copy_(emb[t:t + 1])
            self.graph.replay()
        if state is None:
            self.ex.load_init()
        else:
            self.load(state)

    @torch.no_grad()
    def step(self, ids: int) -> torch.Tensor:
        """One token through the circuit; returns the kept values (fp32, the code first)."""
        self.xbuf.copy_(self.chat._exec_emb([ids]))
        self.graph.replay()
        return self.kbuf


class RvChat:
    def __init__(self, path, device: str | None = None, fast: bool = True,
                 executor: bool = False, fuse: bool = False) -> None:
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM

        from chipformer.q35.fast import exact_matmuls
        from chipformer.q35.ir import Target

        exact_matmuls()
        self.path = Path(path)
        self.meta = json.loads((self.path / "chipformer.json").read_text())
        if self.meta.get("map", "p2") != S.MAP:
            raise ValueError(f"checkpoint built for address map {self.meta.get('map', 'p2')!r}; "
                             f"set QWENPUTER_RV_MAP to match (now {S.MAP!r})")
        self.fuse = fuse
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Qwen3_5ForCausalLM.from_pretrained(self.path, dtype=torch.bfloat16)
        self.model = self.model.to(self.dev).eval()
        self.lm = self.model.model
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.T = BR.Text(self.path / "tokenizer.json")
        cfg = self.model.config
        self.c = core.build(Target(cfg.linear_num_key_heads, cfg.linear_num_value_heads,
                                   cfg.intermediate_size), self.T)
        self.n_chip, self.v0 = self.meta["chip_layers"], self.meta["chip_token_offset"]
        self.code_dims = torch.tensor(self.meta["code_dims"], device=self.dev)
        self.decoder = CodeDecoder(self.v0, self.T, device=self.dev)
        self.eos = {248046, 248044}
        self.fast = fast and self.dev == "cuda"
        # executor "verify": both chip paths, the HF view's hc checked against the executor's
        self.executor = executor if self.dev == "cuda" else False
        self.verify = self.executor == "verify"
        if self.executor:
            self._build_exec()
        if self.fast and (not self.executor or self.verify):
            self._build_fast()
        self.fast = self.fast and (not self.executor or self.verify)

    # --- the executor path -----------------------------------------------------------------
    def _build_exec(self) -> None:
        """The chip layers as the circuit itself (tritonexec.Executor2: the IR's arithmetic in
        fp32, one CUDA graph per token) instead of their HF modules. After the chip layers the
        merged residual holds Qwen's embedding dims, the kept dims (output code, tau, mu) and
        zeros (the exit clears every other chip dim and the ballast), so a text token's hc is
        built from the embedding row and the executor's kept values; chip tokens need only the
        code."""
        self.keep_ir = torch.tensor(self.c.keep, device=self.dev)
        cols = self.meta["code_dims"] + [self.meta["tau_dim"], self.meta["mu_dim"]]
        assert len(cols) == len(self.c.keep)
        self.keep_cols = torch.tensor(cols, device=self.dev)
        self.d0 = self.meta["chip_dims_offset"]
        self.emb_cache: dict[int, torch.Tensor] = {}
        self.erow = ExecRow(self)

    def _exec_emb(self, ids: list[int]) -> torch.Tensor:
        """Executor input rows for model ids (cached per id; shared by every ExecRow)."""
        new = sorted({i for i in ids if i not in self.emb_cache})
        if new:
            rows = self.erow.ex.embed([i - self.v0 if i >= self.v0 else BR.text_token(i)
                                       for i in new])
            for i, r in zip(new, rows):
                self.emb_cache[i] = r
        return torch.stack([self.emb_cache[i] for i in ids])

    def _exec_start(self, ids: list[int], state: "ExecState | None" = None) -> None:
        self.erow.start(ids, state)

    def chip_table(self) -> torch.Tensor:
        """Every chip token's executor input features (ex.feat_ids order), on the device, for
        the device token loop. The values are the circuit's token features (feature_value),
        computed once and cached next to the checkpoint."""
        ex = self.erow.ex
        cache = self.path / f"chip_features.{S.MAP}.npy"
        if cache.is_file():
            tab = np.load(cache)
        else:
            tab = np.zeros((S.VOCAB, len(ex.feat_ids)), np.float32)
            for j, d in enumerate(ex.feat_ids):
                tab[:, j] = [float(self.c.feature_value(d, t)) for t in range(S.VOCAB)]
            try:
                np.save(cache, tab)
            except OSError:
                pass
        if tab.shape != (S.VOCAB, len(ex.feat_ids)):
            raise ValueError(f"{cache} does not match the circuit; delete it")
        return torch.from_numpy(tab).to(self.dev)

    def _exec_step(self, ids: int) -> torch.Tensor:
        return self.erow.step(ids)

    def _exec_hc(self, ids: int, keep: torch.Tensor) -> torch.Tensor:
        """A text token's residual after the chip layers."""
        hc = torch.zeros(self.lm.embed_tokens.weight.shape[1], dtype=torch.bfloat16,
                         device=self.dev)
        hc[: self.d0] = self.lm.embed_tokens.weight[ids, : self.d0]
        hc[self.keep_cols] = keep.to(torch.bfloat16)
        return hc

    def code_token(self, code) -> int | None:
        """circuit_token from the output code values."""
        token = int(self.decoder(code))
        if token == CodeDecoder.UNOWNED:
            return None
        if token != CodeDecoder.INVALID:
            return token
        # Only malformed codes need their individual bits copied to the host.
        bits = (code > 0).long().tolist()
        nf = core.NFAM
        fid = sum(b << i for i, b in enumerate(bits[:nf]))
        p = sum(b << i for i, b in enumerate(bits[nf:]))
        raise RuntimeError(f"the circuit predicts no token (family {fid}, payload {p})")

    # --- the fast chip path ----------------------------------------------------------------
    def _build_fast(self) -> None:
        """A second view of the chip layers: a text model of the n chip layers (the same module
        objects) plus one zero full-attention layer, its linear maps compacted to their nonzero
        blocks (fast.compact; exact), with its own static cache, run as one CUDA graph per token
        (torch.compile reduce-overhead). Chip tokens run only here; text tokens run here and then
        through Qwen's layers with the main cache."""
        import copy

        from transformers.models.qwen3_5 import modeling_qwen3_5 as q

        from chipformer.q35 import fast as F

        cfg = copy.deepcopy(self.model.config)
        cfg.num_hidden_layers = self.n_chip + 1
        cfg.layer_types = ["linear_attention"] * self.n_chip + ["full_attention"]
        self.ccfg = cfg
        dummy = q.Qwen3_5DecoderLayer(cfg, self.n_chip).to(self.dev, torch.bfloat16)
        for p in dummy.parameters():
            p.data.zero_()
        with torch.device("meta"):
            cm = q.Qwen3_5TextModel(cfg)
        cm.embed_tokens = self.lm.embed_tokens
        cm.layers = torch.nn.ModuleList([*self.lm.layers[: self.n_chip], dummy])
        cm.norm = torch.nn.Identity()
        cm.rotary_emb = q.Qwen3_5TextRotaryEmbedding(config=cfg).to(self.dev)
        self.cm = cm.eval()

        class _Wrap(torch.nn.Module):  # fast.compact's view of it
            def __init__(self, inner):
                super().__init__()
                self.model = inner

        self.compact = F.compact(_Wrap(cm), qwen=False, n_chip=self.n_chip)
        self.graph = None  # captured after the first prefill (the cache buffers are static)

    def _chip_cache(self):
        """The chip view's cache, allocated once: its state buffers keep their addresses (HF
        updates them in place), so one CUDA graph serves every conversation."""
        from transformers import DynamicCache

        if getattr(self, "ccache", None) is None:
            self.ccache = DynamicCache(config=self.ccfg)
        else:
            for L in range(self.n_chip):
                lay = self.ccache.layers[L]
                lay.conv_states[0].zero_()
                lay.recurrent_states[0].zero_()
                lay.has_previous_state[0] = False
        return self.ccache

    @torch.no_grad()
    def _chip_body(self):
        h = self.cm.embed_tokens(self.g_ids)
        for L in range(self.n_chip):
            h = self.cm.layers[L](h, position_embeddings=None, attention_mask=None,
                                  position_ids=None, past_key_values=self.ccache)
        self.g_out.copy_(h)
        self.g_next.copy_(self.decoder(h[0, -1, self.code_dims]))

    @torch.no_grad()
    def _capture(self):
        """One CUDA graph of the chip layers' single-token decode step. Warm-up runs change the
        cache, so the state is saved and put back around them."""
        self.g_ids = torch.zeros((1, 1), dtype=torch.long, device=self.dev)
        self.g_out = torch.zeros((1, 1, self.model.config.hidden_size), dtype=torch.bfloat16,
                                 device=self.dev)
        self.g_next = torch.empty((), dtype=torch.long, device=self.dev)
        lays = [self.ccache.layers[L] for L in range(self.n_chip)]
        saved = [(lay.conv_states[0].clone(), lay.recurrent_states[0].clone()) for lay in lays]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._chip_body()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._chip_body()
        for lay, (cs, rs) in zip(lays, saved):
            lay.conv_states[0].copy_(cs)
            lay.recurrent_states[0].copy_(rs)

    @torch.no_grad()
    def _chip_step(self, ids: int, g: int, cache, compiled: bool = True):
        if not compiled or self.graph is None:
            x = torch.tensor([[ids]], device=self.dev)
            out = self.cm(input_ids=x, past_key_values=cache, use_cache=True,
                          cache_position=torch.tensor([g], device=self.dev))
            return out.last_hidden_state[0, -1].clone()
        self.g_ids.fill_(ids)
        self.graph.replay()
        return self.g_out[0, -1].clone()

    @torch.no_grad()
    def _qwen_step(self, hc, cache, pos: int, *, full_head: bool = True, project: bool = True):
        """Qwen's layers on a text token whose residual after the chip layers is hc."""
        h = hc.view(1, 1, -1)
        p4 = torch.full((4, 1, 1), pos, device=self.dev, dtype=torch.long)
        pe = self.lm.rotary_emb(h, p4[1:])
        for i in range(self.n_chip, len(self.lm.layers)):
            h = self.lm.layers[i](h, position_embeddings=pe, attention_mask=None,
                                  position_ids=p4[0], past_key_values=cache)
        # Echoes still pass through every language layer to update the cache. When
        # the circuit owns the next token, the vocabulary projection is unnecessary.
        return self._project(self.lm.norm(h), full_head=full_head)[0, -1] if project else None

    def _project(self, h, *, full_head: bool):
        """Project only the vocabulary rows decoding will consume."""
        head = self.model.lm_head
        if not full_head and type(head) is torch.nn.Linear:
            bias = None if head.bias is None else head.bias[:self.v0]
            return torch.nn.functional.linear(h, head.weight[:self.v0], bias)
        logits = head(h)
        return logits if full_head else logits[..., :self.v0]

    @torch.no_grad()
    def _prefill(self, x):
        """Cache the whole prompt; project only its last position and text vocabulary."""
        out = self.lm(input_ids=x, use_cache=True)
        logits = self._project(out.last_hidden_state[:, -1:], full_head=False)[0, -1]
        return out.past_key_values, logits

    # --- one position ----------------------------------------------------------------------
    @torch.no_grad()
    def _layers(self, ids, cache, pos: int | None, *, full_head: bool = True):
        """Run one token through the chip layers (pos None) or the whole stack (text at RoPE
        position pos); returns (residual after the chip layers, logits or None)."""
        h = self.lm.embed_tokens(torch.tensor([[ids]], device=self.dev))
        pe = tp = None
        if pos is not None:
            p4 = torch.full((4, 1, 1), pos, device=self.dev, dtype=torch.long)
            tp = p4[0]
            pe = self.lm.rotary_emb(h, p4[1:])
        hc = None
        n = self.n_chip if pos is None else len(self.lm.layers)
        for i in range(n):
            h = self.lm.layers[i](h, position_embeddings=pe, attention_mask=None,
                                  position_ids=tp, past_key_values=cache)
            if i == self.n_chip - 1:
                hc = h
        logits = None if pos is None else self._project(self.lm.norm(h), full_head=full_head)[0, -1]
        return hc[0, -1], logits

    def circuit_token(self, hc) -> int | None:
        """The model id the circuit predicts at a position, None where Qwen predicts."""
        return self.code_token(hc[self.code_dims])

    def _chip_prediction(self, hc) -> int | None:
        if self.fast and self.graph is not None:
            token = int(self.g_next)
            if token == CodeDecoder.UNOWNED:
                return None
            if token != CodeDecoder.INVALID:
                return token
        return self.circuit_token(hc)

    @torch.no_grad()
    def boot(self, program, *, max_steps: int = 60000) -> FirmwareState:
        """Execute firmware startup in the HF circuit and park at its first HALT.

        Machine.from_program only initializes the program's memory/register image;
        no reference instruction is executed. The returned heads can seed new chats.
        """
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if self.executor:
            return self._exec_boot(program, max_steps)
        core.snapshot(self.c, ref.Machine.from_program(program))
        prefix = torch.tensor([[self.T.NL]], device=self.dev)
        if self.fast:
            cache = self._chip_cache()
            self.cm(input_ids=prefix, past_key_values=cache, use_cache=True)
        else:
            cache = self.lm(input_ids=prefix, use_cache=True).past_key_values
        hf.inject(self.c, cache)
        if self.fast and self.graph is None:
            self._capture()
        token, halt = self.v0 + S.tok("GO"), self.v0 + S.tok("HALT")
        fault = self.v0 + S.tok("FLT", 0)
        for step in range(1, max_steps + 1):
            if fault <= token < fault + S.FAMILIES["FLT"][1]:
                raise RuntimeError("firmware faulted during neural boot")
            hc = (self._chip_step(token, step, cache) if self.fast
                  else self._layers(token, cache, None)[0])
            # Consume HALT as well: commit its pending writes and park the bridge.
            if token == halt:
                heads = tuple(cache.layers[i].recurrent_states[0].clone()
                              for i in range(self.n_chip))
                return FirmwareState(heads, step)
            token = self._chip_prediction(hc)
            if token is None:
                raise RuntimeError("the circuit gave up a position during neural boot")
        raise RuntimeError(f"firmware did not halt within {max_steps} neural boot steps")

    @torch.no_grad()
    def _exec_boot(self, program, max_steps: int) -> ExecState:
        """boot on the executor: the same prefix, snapshot and free run to the first HALT."""
        core.snapshot(self.c, ref.Machine.from_program(program))
        self._exec_start([self.T.NL])
        token, halt = self.v0 + S.tok("GO"), self.v0 + S.tok("HALT")
        for step in range(1, max_steps + 1):
            keep = self._exec_step(token)
            if token == halt:
                return self.erow.save(step)
            token = self.code_token(keep[: len(self.code_dims)])
            if token is None or token < self.v0:
                raise RuntimeError("the circuit gave up a position during neural boot")
        raise RuntimeError(f"firmware did not halt within {max_steps} neural boot steps")

    # --- a conversation --------------------------------------------------------------------
    def prompt_ids(self, messages, tools=(CALC_TOOL,)) -> list[int]:
        out = self.tok.apply_chat_template(messages, tools=list(tools) or None,
                                           add_generation_prompt=True, enable_thinking=False,
                                           tokenize=True)
        if not isinstance(out, list):
            out = out["input_ids"]
        return [int(x) for x in out]

    @torch.no_grad()
    def generate(self, ids: list[int], machine: "ref.Machine | FirmwareState", max_new: int = 4000,
                 max_chip: int = 60000, check_head: bool = False, verbose: bool = False,
                 lock: "BR.Session | None" = None, on_token=None, stop=None,
                 force: list[int] | None = None, until_released: bool = False) -> dict:
        """Greedy decoding after the prompt ``ids`` with the parked ``machine`` loaded.
        ``lock`` (a Session on a copy of the machine): stop at the first token that differs from
        the reference's. ``on_token(id)`` is called with every generated model id as it comes.
        ``force``: Qwen tokens written in place of Qwen's own choices at the first positions
        Qwen owns (e.g. a tool call the user typed as keys); the circuit captures them as it
        would Qwen's. ``until_released``: stop when Qwen gets a position back after a run."""
        force = list(force or [])
        booted = isinstance(machine, (FirmwareState, ExecState))
        if booted and isinstance(machine, FirmwareState) == bool(self.executor):
            raise ValueError("this firmware state belongs to the other chip path")
        x = torch.tensor([ids], device=self.dev)
        cache, logits = self._prefill(x)  # (fast: Qwen's layers only)
        if not booted:
            core.snapshot(self.c, machine)
        if self.executor:
            self._exec_start(ids, machine if booted else None)
        if self.fast:
            ccache = self._chip_cache()
            h = self.cm.embed_tokens(x)  # the prompt through the chip layers only
            for L in range(self.n_chip):
                h = self.cm.layers[L](h, position_embeddings=None, attention_mask=None,
                                      position_ids=None, past_key_values=ccache)
            if booted:
                machine.inject(ccache)
            else:
                hf.inject(self.c, ccache)
            if self.graph is None:
                self._capture()
            g = len(ids)  # the chip view's cache position (every token)
        elif not self.executor:
            if booted:
                machine.inject(cache)
            else:
                hf.inject(self.c, cache)
        n_verify_bad = 0
        pos = len(ids)  # the next text token's RoPE position
        gen, n_text, n_chip_tok = [], 0, 0
        # the prompt's last position: Qwen predicts (nothing captured yet)
        nxt = force.pop(0) if force else int(torch.argmax(logits[: self.v0]))
        t0 = time.perf_counter()
        mismatch = 0
        t_chip = t_text = 0.0
        gpu_phases = [] if torch.device(self.dev).type == "cuda" else None
        expect, lock_error = [], None
        owned = False
        while n_text < max_new and n_chip_tok < max_chip:
            if stop is not None and stop.is_set() and not owned:
                break
            gen.append(nxt)
            if on_token is not None:
                on_token(nxt)
            if lock is not None:  # check against the reference as we go
                st = nxt - self.v0 if nxt >= self.v0 else BR.text_token(nxt)
                if expect:
                    e = expect.pop(0)
                    if st != e:
                        lock_error = {"at": len(gen) - 1, "got": S.name(st) if st < S.VOCAB
                                      else self.tok.decode([nxt]),
                                      "want": S.name(e) if e < S.VOCAB
                                      else self.tok.decode([e - S.TEXT_TOKEN]),
                                      "recent": [S.name(t - self.v0) if t >= self.v0 else
                                                 self.tok.decode([t]) for t in gen[-12:]]}
                        break
                elif st < S.VOCAB:
                    lock_error = {"at": len(gen) - 1, "got": S.name(st), "want": "text"}
                    break
                else:
                    k = len(lock.toks)
                    try:
                        lock.qwen([nxt])
                    except RuntimeError as e:  # the reference run did not halt
                        lock_error = {"at": len(gen) - 1, "got": str(e), "want": "a halt"}
                        break
                    expect = list(lock.toks[k + 1:])
            if nxt in self.eos:
                break
            if (self.executor and not self.verify and lock is None and nxt >= self.v0
                    and n_chip_tok + 1 < max_chip):
                # a run of chip tokens: the device token loop
                ts = time.perf_counter()
                if gpu_phases is not None and (not gpu_phases or gpu_phases[-1][0] != "chip"):
                    event = torch.cuda.Event(enable_timing=True)
                    event.record(torch.cuda.current_stream(self.dev))
                    gpu_phases.append(("chip", event))

                first = [True]

                def seen(block):  # nxt itself was reported above
                    if on_token is not None:
                        for t in block[1:] if first[0] else block:
                            on_token(t)
                    first[0] = False

                fed, pred = self.erow.run_chip(nxt, max_chip - n_chip_tok, seen, stop)
                gen += fed[1:]
                n_chip_tok += len(fed)
                t_chip += time.perf_counter() - ts
                if pred < 0:
                    raise RuntimeError("the circuit gave up a chip position")
                nxt, owned = pred, True
                continue
            if verbose and len(gen) % 500 == 0:
                tail = [S.name(t - self.v0) if t >= self.v0 else repr(self.tok.decode([t]))
                        for t in gen[-8:]]
                print(f"  {len(gen)} tokens ({n_text} text, {n_chip_tok} chip) "
                      f"{time.perf_counter() - t0:.0f}s: {' '.join(tail)}", flush=True)
            ts = time.perf_counter()
            chip_in = nxt >= self.v0
            if gpu_phases is not None:
                phase = "chip" if chip_in else "text"
                if not gpu_phases or gpu_phases[-1][0] != phase:
                    event = torch.cuda.Event(enable_timing=True)
                    event.record(torch.cuda.current_stream(self.dev))
                    gpu_phases.append((phase, event))
            hc = keep = None
            if self.executor:
                keep = self._exec_step(nxt)
                if not chip_in:
                    hc = self._exec_hc(nxt, keep)
            if self.fast and (self.verify or not self.executor):
                hf_hc = self._chip_step(nxt, g, ccache)
                g += 1
                if self.verify:
                    same = (bool((hf_hc[self.keep_cols] == keep.to(torch.bfloat16)).all())
                            if chip_in else bool((hf_hc == hc).all()))
                    if not same:
                        n_verify_bad += 1
                hc = hf_hc
            elif not self.executor:
                hc, _ = self._layers(nxt, cache, None)
            if hc is None:
                ct = self.code_token(keep[:len(self.code_dims)])
            else:
                ct = self._chip_prediction(hc)
            owned = ct is not None
            if chip_in:  # a chip token
                n_chip_tok += 1
                nxt = ct
                t_chip += time.perf_counter() - ts
                if nxt is None:
                    raise RuntimeError("the circuit gave up a chip position")
            else:
                n_text += 1
                logits = self._qwen_step(hc, cache, pos, full_head=check_head,
                                         project=ct is None or check_head)
                pos += 1
                if ct is not None and check_head and int(torch.argmax(logits)) != ct:
                    mismatch += 1
                t_text += time.perf_counter() - ts
                if ct is None and force:
                    nxt = force.pop(0)
                elif ct is None and until_released and n_chip_tok:
                    break
                else:
                    nxt = ct if ct is not None else int(torch.argmax(logits[: self.v0]))
        if gpu_phases:
            # Echo steps no longer force a host sync through the head's argmax.
            # Time phase boundaries on the device, then wait once before reporting.
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream(self.dev))
            end.synchronize()
            totals = {"chip": 0.0, "text": 0.0}
            for i, (phase, start) in enumerate(gpu_phases):
                finish = gpu_phases[i + 1][1] if i + 1 < len(gpu_phases) else end
                totals[phase] += start.elapsed_time(finish) / 1000
            t_chip, t_text = totals["chip"], totals["text"]
        dt = time.perf_counter() - t0
        if not self.executor:
            self.last_firmware_state = FirmwareState.capture(
                ccache if self.fast else cache, self.n_chip, n_chip_tok)
        else:
            self.last_firmware_state = self.erow.save(n_chip_tok)
        return {"tokens": gen, "text_tokens": n_text, "chip_tokens": n_chip_tok,
                "seconds": dt, "head_mismatch": mismatch, "lock_error": lock_error,
                "verify_mismatch": n_verify_bad if self.verify else None,
                "chip_seconds": round(t_chip, 2), "text_seconds": round(t_text, 2),
                "circuit_paused": owned}

    def to_stream(self, ids: list[int]) -> list[int]:
        """Model ids -> the circuit's token stream (chip tokens, TEXT_TOKEN | qwen id)."""
        return [i - self.v0 if i >= self.v0 else BR.text_token(i) for i in ids]

    def text_of(self, ids: list[int]) -> str:
        return self.tok.decode([i for i in ids if i < self.v0], skip_special_tokens=False)


def replay(chat: RvChat, prompt: list[int], gen: list[int], firmware) -> dict:
    """Check a generated stream: the reference machine (fed Qwen's own tokens) must produce the
    same chip tokens and echoes, and the IR circuit must predict every circuit-owned token."""
    from chipformer.q35 import ir

    m = ref.Machine.from_program(elf.program(firmware))
    BR.park(m)
    ses = BR.Session(chat.T, m)
    stream = chat.to_stream(prompt + gen)
    # the prompt is prefilled history (its system prompt shows the tool-call format, so it holds
    # <tool_call> .. </tool_call> tokens: the circuit owns those </tool_call> positions, but no
    # run follows); then the reference is fed Qwen's tokens and appends the runs itself
    ses.history(prompt)
    skip = {x for x in range(len(prompt)) if prompt[x] == chat.T.TCC}
    i = 0
    pos = len(prompt)
    while pos < len(stream):
        t = stream[pos]
        if not S.is_text(t):
            raise AssertionError(f"chip token {S.name(t)} at {pos} where the reference has none")
        k = len(ses.toks)
        ses.qwen([t - S.TEXT_TOKEN])
        pos += 1
        added = ses.toks[k + 1:]
        if added:  # a run: the stream must continue with exactly the reference's tokens
            got = stream[pos:pos + len(added)]
            if got != added:
                j = next((x for x in range(min(len(got), len(added))) if got[x] != added[x]),
                         min(len(got), len(added)))
                return {"ok": False, "where": pos + j, "got": got[j:j + 5], "want": added[j:j + 5]}
            pos += len(added)
        i += 1
    m2 = ref.Machine.from_program(elf.program(firmware))
    BR.park(m2)
    core.snapshot(chat.c, m2)
    carry, mats = {}, []
    for a in range(0, len(stream), 2048):
        mats.append(ir.evaluate(chat.c, stream[a:a + 2048], carry=carry))
    mat = np.concatenate(mats)
    pred = core.predictions(chat.c, mat)
    own = np.rint(ir.ev(chat.c.owner, mat)).astype(int)
    bad_own = int(sum(own[x] != int(ses.own[x]) for x in range(len(stream))))
    bad = int(sum(1 for x in range(len(stream) - 1)
                  if ses.own[x] and x not in skip and pred[x] != stream[x + 1]))
    return {"ok": bad == 0 and bad_own == 0, "positions": len(stream),
            "owned": int(sum(ses.own)) - len(skip), "wrong_predictions": bad,
            "owner_mismatch": bad_own, "runs": len(ses.runs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--firmware", default="tests/data/rv/calc-bridge.elf")
    ap.add_argument("--ask", default="Use calc: what is 123456789 * 987654321?")
    ap.add_argument("--max-new", type=int, default=600)
    ap.add_argument("--check", action="store_true",
                    help="validate vocabulary predictions and run reference lockstep and replay")
    ap.add_argument("--slow", action="store_true", help="every layer through the HF modules")
    args = ap.parse_args()
    chat = RvChat(args.model, fast=not args.slow)
    program = elf.program(args.firmware)
    if args.check:
        m = ref.Machine.from_program(program)
        BR.park(m)
    else:
        m = chat.boot(program)
    ids = chat.prompt_ids([{"role": "user", "content": args.ask}])
    lock = None
    if args.check:
        m2 = ref.Machine.from_program(elf.program(args.firmware))
        BR.park(m2)
        lock = BR.Session(chat.T, m2)
        lock.history(ids)
    r = chat.generate(ids, m, args.max_new, check_head=args.check, verbose=True, lock=lock)
    print(chat.text_of(r["tokens"]))
    print(f"[{r['text_tokens']} text + {r['chip_tokens']} chip tokens in {r['seconds']:.1f}s]")
    if args.check:
        print(f"head mismatches {r['head_mismatch']}, lock error {r['lock_error']}")
        print(replay(chat, ids, r["tokens"], args.firmware))


if __name__ == "__main__":
    main()
