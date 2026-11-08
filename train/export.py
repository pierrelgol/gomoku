#!/usr/bin/env python3
"""Export / quantise trained checkpoints into the files the engine embeds.

  value:  python export.py value  --ckpt value.pt  --data DIR... --tables T --out assets/standard.nnue
          -> GNN3 int8/int16 network (src/engine/nnue.zig) + <out>.check / <out>.check.bin:
             64 validation positions with the integer reference evaluation; the
             engine must reproduce them exactly (`Gomoku --nncheck <out>.check.bin`).
  policy: python export.py policy --ckpt policy.pt --tables T --out assets/standard.pol
          -> GPOL int32 tables (src/engine/policy.zig).

The integer reference below is the engine's evaluation, operation by
operation (wrapping int16 accumulator, shifts, clamps).
"""
import argparse, math, os, struct
import numpy as np
import torch
from common import *
from train_value import Cache, empty_batch, reparam, acc_range, qlog_for, load_checkpoint


class Quantized:
    """Integer network exactly as the engine computes it (see nnue.zig)."""

    def __init__(self, net, feats, maxdev, K):
        dev = feats.dev
        C = net.C
        self.C, self.H1, self.H2, self.K = C, net.H1, net.H2, int(K)
        # keep the scale QAT trained with (if any): the integer net is then the trained model
        self.qlog = int(round(math.log2(net.qscale))) if net.qscale > 0 else qlog_for(maxdev)
        assert maxdev * 2.0 ** self.qlog < 32000, "accumulator range grew past int16 during QAT"
        Q = 2.0 ** self.qlog
        swap, canon = window_maps()
        vocab = feats.vocab.cpu().numpy()
        T2, P2, cb = reparam(net)
        tq = lambda a: np.clip(np.round(a * Q), -32767, 32767).astype(np.int64)
        T = T2.cpu().numpy()
        P = P2.cpu().numpy()
        perm = feats.cperm.cpu().numpy()
        self.T = np.concatenate([tq(T[vocab[canon]]), tq(T[vocab[canon[swap]]])], 1)  # [65536, 2C]
        self.P = np.concatenate([tq(P), tq(P[perm])], 1)                              # [NP, 2C]
        hp = Q * 64  # high-precision accumulator scale
        cbv = cb.cpu().numpy()
        with torch.no_grad():
            e = empty_batch(feats, dev)
            a0 = net.acc(e["v_stm"], e["wcnt"], e["c_stm"], e["ccnt"], e["n_empty"])[0].cpu().numpy()  # includes bias
        self.cb = np.round(np.concatenate([cbv, cbv]) * hp).astype(np.int64)
        # a0 (with 361 cell biases) at the high-precision scale, minus the 361 cb it already contains
        self.a0 = np.round(np.concatenate([a0, a0]) * hp).astype(np.int64) - 361 * self.cb
        self.shift = self.qlog + 6 + int(math.log2(D)) - int(math.log2(ACT))  # A_hp -> u8 activation
        w = lambda t: np.round(t.detach().clamp(-WMAX, WMAX).cpu().numpy() * WS).astype(np.int64)
        b = lambda t: np.round(t.detach().cpu().numpy() * ACT * WS).astype(np.int64)
        self.W1, self.b1, self.cap = w(net.l1.weight), b(net.l1.bias), b(net.cap)
        self.W2, self.b2 = w(net.l2.weight), b(net.l2.bias)
        self.Wo, self.bo = w(net.out.weight)[0], int(b(net.out.bias)[0])
        # the engine's accumulator is relative to its own (quantised) empty-board sum
        ew, ec, ee, _, _ = board_windows(feats, torch.zeros((1, REC), dtype=torch.uint8, device=dev))
        self.base = self._abs_acc(ew[0], ec[0], ee[0])

    def _abs_acc(self, w, cell, empty):
        a = np.zeros(2 * self.C, np.int64)
        for q in range(361):
            if not empty[q]:
                continue
            for d in range(4):
                a += self.T[w[q, d]]
            a += self.P[cell[q]]
        return a

    def evaluate(self, w, cell, empty, side, cap_stm, cap_opp):
        """w [361,4] absolute windows, cell [361] black-perspective cell index."""
        acc = self._abs_acc(w, cell, empty) - self.base
        acc = ((acc + 32768) % 65536) - 32768  # wrapping int16, as in the engine
        n_empty = int(empty.sum())
        A = self.a0 + (acc << 6) + self.cb * n_empty
        C = self.C
        own, opp = (A[:C], A[C:]) if side == 0 else (A[C:], A[:C])
        x = np.clip(np.concatenate([own, opp]) >> self.shift, 0, 127)
        h = np.clip((self.W1 @ x + self.b1 + self.cap[cap_stm] + self.cap[6 + cap_opp]) >> 6, 0, 127)
        g = np.clip((self.W2 @ h + self.b2) >> 6, 0, 127)
        o = int(self.Wo @ g) + self.bo
        return (o * self.K) >> 13

    def write(self, path):
        C, H1, H2 = self.C, self.H1, self.H2
        with open(path, "wb") as fh:
            fh.write(b"GNN3")
            fh.write(struct.pack("<IIIIII", C, H1, H2, self.K, self.qlog, self.shift))
            fh.write(self.T.astype("<i2").tobytes())
            fh.write(self.P.astype("<i2").tobytes())
            fh.write(self.cb.astype("<i4").tobytes())
            fh.write(self.a0.astype("<i4").tobytes())
            fh.write(self.W1.astype("i1").tobytes())          # [H1][2C]
            fh.write(self.b1.astype("<i4").tobytes())
            fh.write(self.cap.astype("<i4").tobytes())         # [12][H1]
            fh.write(self.W2.astype("i1").tobytes())          # [H2][H1]
            fh.write(self.b2.astype("<i4").tobytes())
            fh.write(self.Wo.astype("i1").tobytes())          # [H2]
            fh.write(struct.pack("<i", self.bo))
        print(f"exported {path} ({os.path.getsize(path)} bytes), Q = 2^{self.qlog}, shift {self.shift}")


def board_windows(feats, raw):
    """Absolute windows / cell indices / empty mask of raw records (numpy)."""
    f = feats.absolute(raw)
    B = raw.shape[0]
    return (f["w"].view(B, 361, 4).cpu().numpy(), f["cell"].cpu().numpy(), f["empty"].cpu().numpy(),
            f["side"].cpu().numpy(), f["cap"].cpu().numpy())



def export_value(args, dev):
    net, K = load_checkpoint(args.ckpt, dev)
    if args.qlog is not None:
        net.qscale = 2.0 ** args.qlog
    net.qat = True
    feats = Features(dev, args.tables)
    _, val_paths = split_paths(args.data)
    va = load_records(val_paths)
    va = va[va["flags"] & 1 == 0]
    va_raw = torch.tensor(va.view(np.uint8).reshape(-1, REC), device=dev)
    va_c = Cache(feats, va_raw)
    qn = Quantized(net, feats, acc_range(net, feats, va_c), K)
    qn.write(args.out)
    n = 64
    w, cell, empty, side, cap = board_windows(feats, va_raw[:n])
    ref = [qn.evaluate(w[i], cell[i], empty[i], side[i], min(cap[i][side[i]], 5), min(cap[i][1 - side[i]], 5)) for i in range(n)]
    with torch.no_grad():
        fl = (net(va_c.batch(torch.arange(n, device=dev))) * K).cpu().numpy()
    print("integer vs float (QAT) max |diff|:", float(np.max(np.abs(np.array(ref) - fl))))
    np.savetxt(args.out + ".check", np.stack([np.arange(n), np.array(ref)], 1), fmt="%d")
    va[:n].tofile(args.out + ".check.bin")


def export_policy(args, dev):
    from train_policy import PFeatures, load_policy
    pf = PFeatures(dev, args.tables)
    net = load_policy(args.ckpt, pf, dev)
    S = 256.0
    qz = lambda t: np.round(t.detach().cpu().numpy().astype(np.float64) * S).astype(np.int64)
    Lf = qz(net.L)
    lv = pf.lvocab.cpu().numpy()
    line = np.arange(1 << 16)
    swapb = ((line & 0xFF) << 8) | (line >> 8)          # LineInfo with the colours swapped
    occurs = np.zeros(1 << 16, bool)
    occurs[pf.lvals] = True
    Lb = np.where(occurs, Lf[lv], 0)                   # own = black
    Lw = np.where(occurs[swapb], Lf[lv[swapb]], 0)     # own = white
    with open(args.out, "wb") as fh:
        fh.write(struct.pack("<4sIII", b"GPOL", 1, 0, int(S)))
        fh.write(qz(net.Pc).astype("<i4").tobytes())                   # [2704] own perspective
        fh.write(np.stack([Lb, Lw], 1).astype("<i4").tobytes())        # [65536][2]
        fh.write(qz(net.Co).astype("<i4").tobytes())                   # [4][6]
        fh.write(qz(net.Cx).astype("<i4").tobytes())
    print(f"exported {args.out} ({os.path.getsize(args.out)} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["value", "policy"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", nargs="+", help="value: data directories (validation parts for the checks)")
    ap.add_argument("--qlog", type=int, default=None, help="value: force the table scale 2^qlog (old checkpoints)")
    args = ap.parse_args()
    dev = setup_cuda()
    if args.kind == "value":
        if not args.data:
            ap.error("value export needs --data")
        export_value(args, dev)
    else:
        export_policy(args, dev)


if __name__ == "__main__":
    main()
