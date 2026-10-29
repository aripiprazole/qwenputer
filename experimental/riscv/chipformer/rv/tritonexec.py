"""Executor v2: the circuit's step as fused Triton kernels, one per sublayer, under one CUDA graph.

Same semantics as fastexec.Executor (the IR's arithmetic in fp32, every value exact), with the
weights kept as sparse rows (CSR, padded per sublayer to a power-of-2 width KP so a row is one
2D gather instead of a loop of dependent loads):

  MLP    a program per block of units: a and b are sparse dot products with the residual, then
         coef * a * relu(b) goes (atomically) into the output buffer
  heads  a program per value head: q / k / v / z are sparse dot products over the residual and
         its lag history (the conv taps folded into the weights); then decay (only the rows the
         head's keys reach), the one-row write (key = argmax), the read of the query's rows only
         (at most QHOT of them, found by argmax), the ballast norm, the gate and the output
         columns into the output buffer
  apply  residual += output buffer; after a head sublayer its lag history moves on

    Executor2(c, batch).step(x) -> x after the whole circuit (in place)

With ``fuse``, runs of independent sublayers (the RAM and DROM tables of a large map) run as
one step (see groups). ``tune`` picks each step's launch config by timing (see Executor2.tune).
Weights exact in fp16 are stored at that width (``compact``); with ``pdl`` the kernels use
programmatic dependent launch (constant loads overlap the previous kernel's tail); with
``direct`` a step that reads none of the dims it writes (and, for heads, no lagged values)
adds into the residual itself.
(The circuit's values are exact in fp32, so the order of the additions does not change them.)
"""

import numpy as np
import torch
import triton
import triton.language as tl

from chipformer.dsl.graph import ONE
from chipformer.q35.ir import FFN, HD, Head


def _csr(rows):
    """CSR arrays (row_ptr, idx, w) of a list of {col: weight}."""
    rp, ix, w = [0], [], []
    for r in rows:
        for k, v in r.items():
            if v:
                ix.append(k)
                w.append(v)
        rp.append(len(ix))
    if not ix:
        ix, w = [0], [0.0]
    return np.array(rp, np.int32), np.array(ix, np.int32), np.array(w, np.float32)


def _kp(rp) -> int:
    """The padded row width: the next power of 2 of the longest row (at least 2)."""
    return max(2, 1 << int(max(np.diff(rp).max(initial=0), 1) - 1).bit_length())


@triton.jit
def _ready(PDL: tl.constexpr, EARLY: tl.constexpr):
    """With programmatic dependent launch: wait until the previous kernel in the stream is done
    and its writes are visible (everything after this may read what it wrote); with EARLY, let
    the next kernel start its own constant loads meanwhile."""
    # (the PTX of triton.language.extra.cuda's gdc_wait / gdc_launch_dependents, which do not
    # compile under Triton 3.4)
    if PDL:
        tl.inline_asm_elementwise("griddepcontrol.wait; // $0", "=r", [], dtype=tl.int32,
                                  is_pure=False, pack=1)
        if EARLY:
            tl.inline_asm_elementwise("griddepcontrol.launch_dependents; // $0", "=r", [],
                                      dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _ffn_kernel(x_ptr, y_ptr, D, U, rpa, ixa, wa, rpb, ixb, wb, coef, outd,
                BLOCK: tl.constexpr, KA: tl.constexpr, KB: tl.constexpr,
                PDL: tl.constexpr, EARLY: tl.constexpr):
    pid = tl.program_id(0)
    bat = tl.program_id(1)
    u = pid * BLOCK + tl.arange(0, BLOCK)
    m = u < U
    xb = x_ptr + bat * D
    a0 = tl.load(rpa + u, mask=m, other=0)
    na = tl.load(rpa + u + 1, mask=m, other=0) - a0
    ka = tl.arange(0, KA)
    ma = m[:, None] & (ka[None, :] < na[:, None])
    ia = tl.load(ixa + a0[:, None] + ka[None, :], mask=ma, other=0).to(tl.int32)
    wta = tl.load(wa + a0[:, None] + ka[None, :], mask=ma, other=0.0).to(tl.float32)
    b0 = tl.load(rpb + u, mask=m, other=0)
    nb = tl.load(rpb + u + 1, mask=m, other=0) - b0
    kb = tl.arange(0, KB)
    mb = m[:, None] & (kb[None, :] < nb[:, None])
    ib = tl.load(ixb + b0[:, None] + kb[None, :], mask=mb, other=0).to(tl.int32)
    wtb = tl.load(wb + b0[:, None] + kb[None, :], mask=mb, other=0.0).to(tl.float32)
    cf = tl.load(coef + u, mask=m, other=0.0)
    od = tl.load(outd + u, mask=m, other=0)
    _ready(PDL, EARLY)
    av = tl.sum(wta * tl.load(xb + ia, mask=ma, other=0.0), axis=1)
    bv = tl.sum(wtb * tl.load(xb + ib, mask=mb, other=0.0), axis=1)
    val = cf * av * tl.maximum(bv, 0.0)
    tl.atomic_add(y_ptr + bat * D + od, val, mask=m)


@triton.jit
def _lag_load(xb, hb, i, m, D):
    """The lagged residual: index lag * D + d (lag 0: x, lags 1..3: the history)."""
    lo = i < D
    return tl.where(lo, tl.load(xb + i, mask=m & lo, other=0.0),
                    tl.load(hb + i - D, mask=m & (i >= D), other=0.0))


@triton.jit
def _rows_csr(rp, ix, w, base, ch, KP: tl.constexpr):
    """128 sparse rows (row base + ch): their columns, weights (widened) and mask."""
    r0 = tl.load(rp + base + ch)
    n = tl.load(rp + base + ch + 1) - r0
    kk = tl.arange(0, KP)
    mk = kk[None, :] < n[:, None]
    i = tl.load(ix + r0[:, None] + kk[None, :], mask=mk, other=0).to(tl.int32)
    wt = tl.load(w + r0[:, None] + kk[None, :], mask=mk, other=0.0).to(tl.float32)
    return i, wt, mk


@triton.jit
def _rows_dot(xb, hb, D, i, wt, mk):
    """relu of the rows' dot products with the lagged residual."""
    return tl.maximum(tl.sum(wt * _lag_load(xb, hb, i, mk, D), axis=1), 0.0)


@triton.jit
def _rows(xb, hb, D, rp, ix, w, base, ch, KP: tl.constexpr):
    """relu of 128 sparse rows (row base + ch) over the lagged residual."""
    i, wt, mk = _rows_csr(rp, ix, w, base, ch, KP)
    return _rows_dot(xb, hb, D, i, wt, mk)


@triton.jit
def _scalar_csr(rp, ix, w, vh):
    lane = tl.arange(0, 32)
    s0 = tl.load(rp + vh)
    n = tl.load(rp + vh + 1) - s0
    mk = lane < n
    i = tl.load(ix + s0 + lane, mask=mk, other=0).to(tl.int32)
    wt = tl.load(w + s0 + lane, mask=mk, other=0.0).to(tl.float32)
    return i, wt, mk


@triton.jit
def _scalar(xb, hb, D, rp, ix, w, vh):
    i, wt, mk = _scalar_csr(rp, ix, w, vh)
    return tl.sum(wt * _lag_load(xb, hb, i, mk, D), axis=0)


@triton.jit
def _head_kernel(x_ptr, h_ptr, S_ptr, y_ptr, D, V,
                 qrp, qix, qw, krp, kix, kw, vrp, vix, vw, zrp, zix, zw,
                 arp, aix, aw, brp, bix, bw, bconst, odim, ow, khead, nrows,
                 KQ: tl.constexpr, KK: tl.constexpr, KV: tl.constexpr, KZ: tl.constexpr,
                 QHOT: tl.constexpr, PDL: tl.constexpr, EARLY: tl.constexpr):
    vh = tl.program_id(0)
    bat = tl.program_id(1)
    ch = tl.arange(0, 128)
    xb = x_ptr + bat * D
    hb = h_ptr + bat * 3 * D
    h = tl.load(khead + vh)
    # the constant parts first (with PDL, while the previous kernel finishes)
    ki, kw_, km = _rows_csr(krp, kix, kw, h * 128, ch, KK)
    vi, vw_, vm = _rows_csr(vrp, vix, vw, vh * 128, ch, KV)
    zi, zw_, zm = _rows_csr(zrp, zix, zw, vh * 128, ch, KZ)
    qi_, qw_, qm_ = _rows_csr(qrp, qix, qw, h * 128, ch, KQ)
    bi, bw_, bm = _scalar_csr(brp, bix, bw, vh)
    bc = tl.load(bconst + vh)
    nr = tl.load(nrows + vh)  # > 0: the head decays, in its first nr rows
    wv = tl.load(ow + vh * 128 + ch)
    dd = tl.load(odim + vh * 128 + ch)
    _ready(PDL, EARLY)
    k = _rows_dot(xb, hb, D, ki, kw_, km)
    v = _rows_dot(xb, hb, D, vi, vw_, vm)
    z = _rows_dot(xb, hb, D, zi, zw_, zm)
    q = _rows_dot(xb, hb, D, qi_, qw_, qm_)
    bv = tl.sum(bw_ * _lag_load(xb, hb, bi, bm, D), axis=0) + bc
    Sb = S_ptr + (bat * V + vh) * 16384
    if nr > 0:
        av = _scalar(xb, hb, D, arp, aix, aw, vh)
        fac = tl.where(av == 0.0, 1.0, tl.where(av == 110.0, 0.5, 0.0))
        if fac != 1.0:
            for r in range(nr):
                tl.store(Sb + r * 128 + ch, tl.load(Sb + r * 128 + ch) * fac)
    kmax = tl.max(k, axis=0)
    krow = tl.argmax(k, axis=0)
    if (kmax > 0.0) & (bv == 1.0):
        tl.store(Sb + krow * 128 + ch, v)
    tl.debug_barrier()
    # the read: the sum of the rows whose query is set (multi-hot queries have equal values)
    o = tl.zeros([128], dtype=tl.float32)
    for _ in range(QHOT):
        qm = tl.max(q, axis=0)
        qi = tl.argmax(q, axis=0)
        if qm > 0.0:
            o += tl.load(Sb + qi * 128 + ch)
        q = tl.where(ch == qi, 0.0, q)
    ref = tl.sum(tl.where(ch < 2, o * o, 0.0), axis=0)
    norm = tl.math.sqrt_rn(ref / 128.0)
    out = tl.where(ref > 0.0, tl.math.div_rn(o, tl.where(ref > 0.0, norm, 1.0)), 0.0)
    tl.atomic_add(y_ptr + bat * D + dd, out * z * wv, mask=wv != 0.0)


@triton.jit
def _khead_kernel(x_ptr, h_ptr, S_ptr, y_ptr, D, V,
                  qrp, qix, qw, krp, kix, kw, vrp, vix, vw, zrp, zix, zw,
                  arp, aix, aw, brp, bix, bw, bconst, odim, ow, vh0, nvh, nrows,
                  KQ: tl.constexpr, KK: tl.constexpr, KV: tl.constexpr, KZ: tl.constexpr,
                  QHOT: tl.constexpr, MAXV: tl.constexpr, PDL: tl.constexpr,
                  EARLY: tl.constexpr):
    """_head_kernel with one program per key head: its queries and keys once, then each of its
    value heads. A value head with no key row and no query row set and no decay writes nothing
    and reads zero, so its values, gate and output are skipped (their result is exactly 0)."""
    h = tl.program_id(0)
    bat = tl.program_id(1)
    ch = tl.arange(0, 128)
    xb = x_ptr + bat * D
    hb = h_ptr + bat * 3 * D
    ki, kw_, km = _rows_csr(krp, kix, kw, h * 128, ch, KK)
    qi_, qw_, qm_ = _rows_csr(qrp, qix, qw, h * 128, ch, KQ)
    first = tl.load(vh0 + h)
    count = tl.load(nvh + h)
    _ready(PDL, EARLY)
    k = _rows_dot(xb, hb, D, ki, kw_, km)
    q0 = _rows_dot(xb, hb, D, qi_, qw_, qm_)
    kmax = tl.max(k, axis=0)
    krow = tl.argmax(k, axis=0)
    qmax0 = tl.max(q0, axis=0)
    for j in range(MAXV):
        vh = first + j
        nr = tl.load(nrows + vh, mask=j < count, other=0)
        if (j < count) & ((kmax > 0.0) | (qmax0 > 0.0) | (nr > 0)):
            Sb = S_ptr + (bat * V + vh) * 16384
            if nr > 0:
                av = _scalar(xb, hb, D, arp, aix, aw, vh)
                fac = tl.where(av == 0.0, 1.0, tl.where(av == 110.0, 0.5, 0.0))
                if fac != 1.0:
                    for r in range(nr):
                        tl.store(Sb + r * 128 + ch, tl.load(Sb + r * 128 + ch) * fac)
            bv = _scalar(xb, hb, D, brp, bix, bw, vh) + tl.load(bconst + vh)
            if (kmax > 0.0) & (bv == 1.0):
                v = _rows(xb, hb, D, vrp, vix, vw, vh * 128, ch, KV)
                tl.store(Sb + krow * 128 + ch, v)
            tl.debug_barrier()
            if qmax0 > 0.0:
                q = q0
                o = tl.zeros([128], dtype=tl.float32)
                for _ in range(QHOT):
                    qm = tl.max(q, axis=0)
                    qi = tl.argmax(q, axis=0)
                    if qm > 0.0:
                        o += tl.load(Sb + qi * 128 + ch)
                    q = tl.where(ch == qi, 0.0, q)
                ref = tl.sum(tl.where(ch < 2, o * o, 0.0), axis=0)
                if ref > 0.0:
                    z = _rows(xb, hb, D, zrp, zix, zw, vh * 128, ch, KZ)
                    out = tl.math.div_rn(o, tl.math.sqrt_rn(ref / 128.0))
                    wv = tl.load(ow + vh * 128 + ch)
                    dd = tl.load(odim + vh * 128 + ch)
                    tl.atomic_add(y_ptr + bat * D + dd, out * z * wv, mask=wv != 0.0)


@triton.jit
def _apply_kernel(x_ptr, y_ptr, h_ptr, D, HIST: tl.constexpr, BLOCK: tl.constexpr,
                  PDL: tl.constexpr, EARLY: tl.constexpr):
    """x += y, y = 0; with HIST, the lag history (h: [3, D] per row, h[0] = lag 1) moves on
    first, the input x becoming lag 1."""
    pid = tl.program_id(0)
    bat = tl.program_id(1)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < D
    _ready(PDL, EARLY)
    xo = x_ptr + bat * D + i
    x = tl.load(xo, mask=m, other=0.0)
    if HIST:
        hb = h_ptr + bat * 3 * D + i
        h1 = tl.load(hb + D, mask=m, other=0.0)
        h0 = tl.load(hb, mask=m, other=0.0)
        tl.store(hb + 2 * D, h1, mask=m)
        tl.store(hb + D, h0, mask=m)
        tl.store(hb, x, mask=m)
    yo = y_ptr + bat * D + i
    tl.store(xo, x + tl.load(yo, mask=m, other=0.0), mask=m)
    tl.store(yo, tl.zeros([BLOCK], dtype=tl.float32), mask=m)


@triton.jit
def _next_kernel(x_ptr, D, ONE, code_ix, out_ptr, i, cur_ptr, tab_ptr, fidx_ptr, NF,
                 bases, limits, payloads, printable, NPRINT, v0, last,
                 NCODE: tl.constexpr, NCB: tl.constexpr, NFAM: tl.constexpr,
                 FAMQ: tl.constexpr, FB: tl.constexpr, BLOCK: tl.constexpr,
                 DECODE: tl.constexpr, PDL: tl.constexpr, EARLY: tl.constexpr):
    """Between two circuit steps of the device loop (one program): with DECODE, the step's
    output code -> its model id (CodeDecoder's arithmetic) into out[i], and the chip index of
    that id (clamped) into cur; then the next input: zeros, ONE = 1, and cur's row of the
    feature table at the feature dims (the embedding, as index_copy did)."""
    _ready(PDL, EARLY)
    if DECODE:
        j = tl.arange(0, NCB)
        m = j < NCODE
        code = tl.load(x_ptr + tl.load(code_ix + j, mask=m, other=0), mask=m, other=0.0)
        bit = tl.full([NCB], 1, tl.int64) << j.to(tl.int64)
        packed = tl.sum(tl.where(m & (code > 0), bit, 0), axis=0)
        owned = tl.sum(tl.where(m & (code != 0), 1, 0), axis=0) > 0
        family = packed & ((1 << NFAM) - 1)
        payload = packed >> NFAM
        tok = tl.load(bases + family) + payload * tl.load(payloads + family)
        echoed = tl.load(printable + tl.minimum(payload, NPRINT - 1))
        tok = tl.where(family == FAMQ, echoed, tok)
        tok = tl.where(payload < tl.load(limits + family), tok, -2)
        tok = tl.where(owned, tok, -1)
        tl.store(out_ptr + i, tok)
        cur = tl.minimum(tl.maximum(tok - v0, 0), last)
        tl.store(cur_ptr, cur)
    else:
        cur = tl.load(cur_ptr)
    tl.debug_barrier()
    for off in range(0, D, BLOCK):
        k = off + tl.arange(0, BLOCK)
        tl.store(x_ptr + k, tl.zeros([BLOCK], dtype=tl.float32), mask=k < D)
    tl.debug_barrier()
    tl.store(x_ptr + ONE, 1.0)
    tl.debug_barrier()
    f = tl.arange(0, FB)
    mf = f < NF
    tl.store(x_ptr + tl.load(fidx_ptr + f, mask=mf, other=0),
             tl.load(tab_ptr + cur * NF + f, mask=mf, other=0.0), mask=mf)


def _reads(n) -> set:
    if isinstance(n, FFN):
        return {d for _, a, b in n.terms for d in (*a.terms, *b.terms)}
    ds = set()
    for vd in n.vheads():
        for ch in (*n.q.values(), *n.k.values(), *vd.v.values()):
            ds |= set(ch.expr.terms)
        for e in (*vd.z.values(), vd.a, *([] if vd.beta is None else [vd.beta])):
            ds |= set(e.terms)
    return ds


def _writes(n) -> set:
    return {n.out} if isinstance(n, FFN) else {d for vd in n.vheads() for _, d, _ in vd.out}


def groups(c, fuse: bool) -> list:
    """The nodes of each step: one sublayer each, or with ``fuse`` runs of consecutive sublayers
    of one kind that are independent (none reads a dim another writes). Such sublayers are only
    apart because a layer holds 16 key heads (the RAM and DROM tables): every member then sees
    the same values on the dims it reads, and outputs into shared dims add, so a fused step
    computes exactly what the sublayers compute one after another."""
    out, cur = [], None
    for lv in sorted({n.level for n in c.nodes}):
        nodes = [n for n in c.nodes if n.level == lv]
        kind = isinstance(nodes[0], FFN)
        r = set().union(*map(_reads, nodes))
        w = set().union(*map(_writes, nodes))
        if fuse and cur and cur[0] == kind and not r & cur[2] and not w & cur[1]:
            cur[1] |= r
            cur[2] |= w
            cur[3] += nodes
        else:
            cur = [kind, r, w, nodes]
            out.append(cur)
    return [g[3] for g in out]


class Executor2:
    BLOCK = 64

    def __init__(self, c, batch: int = 1, device="cuda", fuse: bool = False,
                 per_key_head: bool | None = None, compact: bool = True, pdl: bool = False,
                 early: bool = False, direct: bool = False):
        self.c, self.B, self.dev = c, batch, device
        self.per_key_head = bool(per_key_head)
        # programmatic dependent launch: each kernel loads its constant CSR data while the one
        # before it finishes, and waits for it (gdc_wait) before reading anything it wrote
        self._pdl = {"PDL": pdl, "EARLY": pdl and early, "launch_pdl": pdl}
        D = len(c.dims)
        self.D = D
        t = lambda a, dt=torch.int32: torch.tensor(np.asarray(a), dtype=dt, device=device)

        def dev3(arr):
            """CSR on the device. Arrays whose weights are all exact in fp16
            and whose columns fit int16 are stored at that width (half the bytes to read; the
            kernel widens each value back, so the arithmetic is unchanged)."""
            rp, ix, w = arr
            with np.errstate(over="ignore"):  # a weight past fp16's range keeps the array fp32
                exact = np.array_equal(w.astype(np.float16).astype(np.float32), w)
            if compact and len(ix) and int(ix.max()) < 2**15 and exact:
                return [t(rp), t(ix, torch.int16), t(w, torch.float16)]
            return [t(rp), t(ix), t(w, torch.float32)]

        self.steps = []
        for nodes in groups(c, fuse):
            if isinstance(nodes[0], FFN):
                units = [(float(cf), a, b, n.out) for n in nodes for cf, a, b in n.terms]
                ra = _csr([{d: float(v) for d, v in a.terms.items()} for _, a, _, _ in units])
                rb = _csr([{d: float(v) for d, v in b.terms.items()} for _, _, b, _ in units])
                self.steps.append({"kind": "ffn", "U": len(units), "a": dev3(ra), "b": dev3(rb),
                                   "KA": _kp(ra[0]), "KB": _kp(rb[0]),
                                   "coef": t([u[0] for u in units], torch.float32),
                                   "out": t([u[3] for u in units]), "cfg": (self.BLOCK, 4),
                                   # no unit reads a dim the step writes: it may add into the
                                   # residual itself (no buffer, no apply kernel)
                                   "direct": direct and not (set().union(*map(_reads, nodes))
                                                             & set().union(*map(_writes, nodes)))})
                continue
            heads = [n for n in nodes if isinstance(n, Head)]
            vh = [(h, vd) for h, n in enumerate(heads) for vd in n.vheads()]
            V = len(vh)

            def lagged(ch, D=D):
                return {lag * D + d: float(cf) * tw for lag, tw in enumerate(ch.taps) if tw
                        for d, cf in ch.expr.terms.items()}

            qrows, krows = [], []
            for n in heads:
                for s in range(HD):
                    qrows.append(lagged(n.q[s]) if s in n.q else {})
                    krows.append(lagged(n.k[s]) if s in n.k else {})
            vrows, zrows, arows, brows, bconst, odim, ow, khead, nrows = ([] for _ in range(9))
            for h, vd in vh:
                for col in range(HD):
                    vrows.append(lagged(vd.v[col]) if col in vd.v else {})
                    zrows.append({d: float(cf) for d, cf in vd.z[col].terms.items()}
                                 if col in vd.z else {})
                arows.append({d: float(cf) for d, cf in vd.a.terms.items()})
                if vd.beta is None:
                    brows.append({})
                    bconst.append(1.0)
                else:
                    brows.append({d: float(cf) for d, cf in vd.beta.terms.items()})
                    bconst.append(0.0)
                od, w_ = np.zeros(HD, np.int32), np.zeros(HD, np.float32)
                for col, d, w in vd.out:
                    od[col], w_[col] = d, w
                odim.append(od)
                ow.append(w_)
                khead.append(h)
                # a decaying head: only the rows its keys (or its initial state) reach
                reach = set(heads[h].k) | {r for x in (heads[h].init or []) for r in x}
                nrows.append(max(reach) + 1 if vd.a.terms and reach else 0)
            assert max(len(r) for r in arows + brows) <= 32
            q, k, v, z = (_csr(r) for r in (qrows, krows, vrows, zrows))
            st = {"kind": "head", "V": V, "heads": heads, "q": dev3(q), "k": dev3(k),
                  "v": dev3(v), "z": dev3(z), "a": dev3(_csr(arows)), "b": dev3(_csr(brows)),
                  "KQ": _kp(q[0]), "KK": _kp(k[0]), "KV": _kp(v[0]), "KZ": _kp(z[0]),
                  "QHOT": max(getattr(n, "qhot", 2) for n in heads),
                  "bconst": t(bconst, torch.float32), "odim": t(np.concatenate(odim)),
                  "ow": t(np.concatenate(ow), torch.float32), "khead": t(khead),
                  "nrows": t(nrows), "S": torch.zeros(batch, V, HD, HD, device=device),
                  "vh0": t(np.searchsorted(khead, np.arange(len(heads)))),
                  "nvh": t(np.bincount(khead, minlength=len(heads))),
                  "MAXV": int(np.bincount(khead).max()),
                  "hist": torch.zeros(batch, 3, D, device=device),  # lags 1..3
                  "cfg": (self.per_key_head, 4),
                  # reads none of the dims it writes, and no lagged value (its history is never
                  # read): it may add into the residual itself, with no apply kernel
                  "direct": direct and max(int(a[1].max()) for a in (q, k, v)) < D
                  and not (set().union(*map(_reads, nodes)) & set().union(*map(_writes, nodes)))}
            self.steps.append(st)
        self.y = torch.zeros(batch, D, device=device)
        self.feat_ids = sorted(c.token_fns)

    def load_init(self, row: int = 0):
        for st in self.steps:
            if st["kind"] != "head":
                continue
            buf = np.zeros(st["S"].shape[1:], np.float32)
            j = 0
            for n in st["heads"]:
                init = n.init or []
                for x in range(len(n.vheads())):
                    for r, vec in (init[x] if x < len(init) else {}).items():
                        buf[j, r] = vec
                    j += 1
            st["S"][row].copy_(torch.from_numpy(buf))

    def reset(self, rows=None):
        for st in self.steps:
            if st["kind"] == "head":
                (st["hist"].zero_() if rows is None else st["hist"][rows].zero_())

    def embed(self, toks):
        out = np.zeros((len(toks), self.D))
        out[:, ONE] = 1.0
        for d in self.feat_ids:
            out[:, d] = [float(self.c.feature_value(d, t)) for t in toks]
        return torch.tensor(out, dtype=torch.float32, device=self.dev)

    def _launch(self, st, x, cfg) -> None:
        """One step's sublayer kernel (its output into self.y) under the launch config cfg."""
        B, D = x.shape[0], self.D
        if st["kind"] == "ffn":
            block, nw = cfg
            _ffn_kernel[(triton.cdiv(st["U"], block), B)](
                x, x if st["direct"] else self.y, D, st["U"], *st["a"], *st["b"], st["coef"],
                st["out"], BLOCK=block, KA=st["KA"], KB=st["KB"], num_warps=nw, **self._pdl)
            return
        per_key_head, nw = cfg
        if per_key_head:
            _khead_kernel[(len(st["heads"]), B)](
                x, st["hist"], st["S"], x if st["direct"] else self.y, D, st["V"],
                *st["q"], *st["k"], *st["v"],
                *st["z"], *st["a"], *st["b"], st["bconst"], st["odim"], st["ow"],
                st["vh0"], st["nvh"], st["nrows"], KQ=st["KQ"], KK=st["KK"], KV=st["KV"],
                KZ=st["KZ"], QHOT=st["QHOT"], MAXV=st["MAXV"], num_warps=nw, **self._pdl)
        else:
            _head_kernel[(st["V"], B)](
                x, st["hist"], st["S"], x if st["direct"] else self.y, D, st["V"],
                *st["q"], *st["k"], *st["v"],
                *st["z"], *st["a"], *st["b"], st["bconst"], st["odim"], st["ow"],
                st["khead"], st["nrows"], KQ=st["KQ"], KK=st["KK"], KV=st["KV"],
                KZ=st["KZ"], QHOT=st["QHOT"], num_warps=nw, **self._pdl)

    def step(self, x):
        """x [B, D] (in place: the embedding in, the residual after every node out)."""
        grid_apply = (triton.cdiv(self.D, 1024), x.shape[0])
        for st in self.steps:
            self._launch(st, x, st["cfg"])
            if st.get("direct"):
                continue
            _apply_kernel[grid_apply](x, self.y, st.get("hist", x), self.D,
                                      HIST=st["kind"] == "head", BLOCK=1024, **self._pdl)
        return x

    def next_input(self, x, *, code_ix, out, i: int, cur, tab, fidx, decoder, v0: int,
                   one: int, decode: bool = True) -> None:
        """The device loop's work between steps in one kernel (_next_kernel): decode x's code
        into out[i] and cur (decode), then x = the embedding of chip index cur. ``decoder``: a
        CodeDecoder (its tables); ``tab``: [chip vocab, len(fidx)] features."""
        from chipformer.rv import core
        nc = code_ix.numel()
        _next_kernel[(1,)](x, self.D, one, code_ix, out, i, cur, tab, fidx, fidx.numel(),
                           decoder.bases, decoder.limits, decoder.payloads, decoder.printable,
                           decoder.printable.numel(), v0, tab.shape[0] - 1,
                           NCODE=nc, NCB=triton.next_power_of_2(nc), NFAM=core.NFAM,
                           FAMQ=core.FAM_ID["Q"], FB=triton.next_power_of_2(fidx.numel()),
                           BLOCK=1024, DECODE=decode, num_warps=4, **self._pdl)

    @torch.no_grad()
    def tune(self, x, reps: int = 200) -> list:
        """Pick each step's fastest launch config (program per value head or per key head,
        warps, MLP block): each candidate's kernel is timed alone under a CUDA graph. The
        candidates do the same arithmetic per value; the tables' state, the history and the
        buffers are put back afterwards. x: a residual to time with (it is not changed)."""
        saved = [(st["S"].clone(), st["hist"].clone()) for st in self.steps
                 if st["kind"] == "head"]
        x, y0 = x.clone(), self.y.clone()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        def timed(st, cfg) -> float:
            try:  # compiles the candidate; a config over the GPU's limits is never picked
                self._launch(st, x, cfg)
            except triton.runtime.errors.OutOfResources:
                return float("inf")
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._launch(st, x, cfg)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._launch(st, x, cfg)
            g.replay()
            a.record()
            for _ in range(reps):
                g.replay()
            b.record()
            torch.cuda.synchronize()
            return a.elapsed_time(b)

        chosen = []
        for st in self.steps:
            if st["kind"] == "ffn":
                cands = [(blk, nw) for blk in (16, 32, 64) for nw in (1, 2, 4, 8)]
            else:
                cands = [(kh, nw) for kh in (False, True) for nw in (4, 8, 16)]
            st["cfg"] = min(cands, key=lambda cfg: timed(st, cfg))
            chosen.append(st["cfg"])
        it = iter(saved)
        for st in self.steps:
            if st["kind"] == "head":
                S_, h = next(it)
                st["S"].copy_(S_)
                st["hist"].copy_(h)
        self.y.copy_(y0)
        return chosen


def check(c, toks, device="cuda", fuse: bool = False) -> int:
    """Executor2 == ir.evaluate on every dim at every position."""
    from chipformer.q35 import ir

    ex = Executor2(c, 1, device, fuse)
    ex.load_init()
    emb = ex.embed(toks)
    want = ir.evaluate(c, toks, check=False)
    for t in range(len(toks)):
        x = emb[t:t + 1].clone()
        got = ex.step(x)[0].double().cpu().numpy()
        bad = np.nonzero(got != want[t])[0]
        assert not len(bad), (t, [(c.dims[d].name, got[d], want[t, d]) for d in bad[:5]])
    return len(toks)
