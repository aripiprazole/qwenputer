"""Small-memory tensor lowering of the neural IR, for offline harness tests only.

All arithmetic below is sparse neural projection, ReGLU, or DeltaNet recurrence.
There is no instruction decoder or guest ISA implementation in this evaluator.
"""

import torch

from chipformer.q35.ir import FFN, HD


class Projection:
    def __init__(self):
        self.rows, self.lookup = [()], {(): 0}

    def row(self, terms):
        key = tuple(sorted((int(d), float(w)) for d, w in terms if w))
        if key not in self.lookup:
            self.lookup[key] = len(self.rows)
            self.rows.append(key)
        return self.lookup[key]

    def finish(self):
        src, dst, weights = [], [], []
        for row, terms in enumerate(self.rows):
            for column, weight in terms:
                src.append(column); dst.append(row); weights.append(weight)
        self.src, self.dst = torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __call__(self, x):
        return torch.zeros(len(self.rows)).index_add_(0, self.dst, x[self.src] * self.weights)


class SparseTensorCircuit:
    def __init__(self, circuit):
        self.c, self.dims, self.stages = circuit, len(circuit.dims), []
        for level in sorted({n.level for n in circuit.nodes}):
            nodes = [n for n in circuit.nodes if n.level == level]
            projection = Projection()
            if isinstance(nodes[0], FFN):
                terms = [(coef, a, b, node.out) for node in nodes for coef, a, b in node.terms]
                a = torch.tensor([projection.row(a.terms.items()) for _, a, _, _ in terms])
                b = torch.tensor([projection.row(b.terms.items()) for _, _, b, _ in terms])
                coefficients = torch.tensor([float(coef) for coef, _, _, _ in terms])
                outputs = torch.tensor([out for _, _, _, out in terms])
                projection.finish()
                self.stages.append(('ffn', projection, a, b, coefficients, outputs))
                continue
            heads = nodes

            def channel(ch):
                return projection.row((lag * self.dims + dim, float(w) * tap)
                                      for lag, tap in enumerate(ch.taps) if tap
                                      for dim, w in ch.expr.terms.items())

            def channels(values):
                return [channel(values[i]) if i in values else 0 for i in range(HD)]

            q = torch.tensor([channels(h.q) for h in heads])
            k = torch.tensor([channels(h.k) for h in heads])
            values = [(hi, vh) for hi, head in enumerate(heads) for vh in head.vheads()]
            v = torch.tensor([channels(vh.v) for _, vh in values])
            z = torch.tensor([[projection.row(vh.z[i].terms.items()) if i in vh.z else 0
                               for i in range(HD)] for _, vh in values])
            a = torch.tensor([projection.row(vh.a.terms.items()) for _, vh in values])
            beta = torch.tensor([projection.row(vh.beta.terms.items()) if vh.beta is not None else 0
                                 for _, vh in values])
            constant = torch.tensor([float(vh.beta is None) for _, vh in values])
            output = [(j, col, dim, float(weight)) for j, (_, vh) in enumerate(values)
                      for col, dim, weight in vh.out]
            state = torch.zeros(len(values), HD, HD)
            for j, (hi, _) in enumerate(values):
                vh_index = sum(1 for old, _ in values[:j] if old == hi)
                initial = heads[hi].init or []
                for row, data in (initial[vh_index] if vh_index < len(initial) else {}).items():
                    state[j, row] = torch.as_tensor(data, dtype=torch.float32)
            projection.finish()
            self.stages.append(('head', projection, {
                'q': q, 'k': k, 'v': v, 'z': z, 'a': a, 'beta': beta, 'constant': constant,
                'kh': torch.tensor([hi for hi, _ in values]), 'state': state,
                'history': torch.zeros(3, self.dims), 'qhot': max(getattr(h, 'qhot', 2) for h in heads),
                'vi': torch.arange(len(values)),
                'ov': torch.tensor([x[0] for x in output]),
                'oc': torch.tensor([x[1] for x in output]),
                'od': torch.tensor([x[2] for x in output]),
                'ow': torch.tensor([x[3] for x in output]),
            }))

    def snapshot(self):
        """Copy recurrent tensors and convolution history at a parked guest boundary."""
        return tuple((stage[2]['state'].clone(), stage[2]['history'].clone())
                     for stage in self.stages if stage[0] == 'head')

    def restore(self, snapshot):
        heads = [stage[2] for stage in self.stages if stage[0] == 'head']
        for head, (state, history) in zip(heads, snapshot, strict=True):
            head['state'].copy_(state)
            head['history'].copy_(history)

    @torch.no_grad()
    def step_token(self, token):
        x = torch.zeros(self.dims)
        x[0] = 1
        for dim in self.c.token_fns:
            x[dim] = float(self.c.feature_value(dim, token))
        for stage in self.stages:
            if stage[0] == 'ffn':
                _, projection, a, b, weights, outputs = stage
                projected = projection(x)
                x = x + torch.zeros_like(x).index_add_(
                    0, outputs, weights * projected[a] * torch.relu(projected[b]))
                continue
            _, projection, st = stage
            xs = torch.cat([x[None], st['history']])
            values = projection(xs.flatten())
            q, k, v, z = [torch.relu(values[st[name]]) for name in ('q', 'k', 'v', 'z')]
            a = values[st['a']]
            state = st['state']
            state *= torch.where(a == 0, 1.0, torch.where(a == 110, 0.5, 0.0))[:, None, None]
            maximum, row = k[st['kh']].max(-1)
            write = (maximum > 0) & (values[st['beta']] + st['constant'] == 1)
            old = state[st['vi'], row]
            state[st['vi'], row] = torch.where(write[:, None], v, old)
            query, rows = q[st['kh']].topk(st['qhot'], dim=-1)
            out = (state[st['vi'][:, None], rows] * (query > 0)[:, :, None]).sum(1)
            ballast = out[:, :2].square().sum(-1)
            norm = torch.where(ballast > 0, torch.sqrt(ballast / HD), torch.ones_like(ballast))
            out = torch.where((ballast > 0)[:, None], out / norm[:, None], torch.zeros_like(out))
            y = torch.zeros_like(x).index_add_(0, st['od'],
                out[st['ov'], st['oc']] * z[st['ov'], st['oc']] * st['ow'])
            st['history'].copy_(xs[:3])
            x = x + y
        return x
