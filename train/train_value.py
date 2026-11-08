#!/usr/bin/env python3
"""Train the Gomoku evaluation network from self-play records (src/tools/datagen.zig).
Writes a checkpoint (.pt); train/export.py turns it into the engine file.

Model (mirrored exactly, in integer arithmetic, by src/engine/nnue.zig):

  Feature accumulator, for perspective p (black / white as "own"):
      A_p = sum over empty cells q of ( sum_d T[window_p(q,d)] + P[cell_p(q)] )
    * window: the 8 neighbours of q on line d (2 bits each, colours swapped
      for p = white); T is tied under mirroring, so the model is exactly
      invariant to the 8 board symmetries.
    * cell: rules-aware per-cell state from the engine's own tables:
      (Pattern4 own, Pattern4 opp, own capture here?, opp capture here?,
       forbidden double-three for own?, for opp?)  -> 13*13*2*2*2*2 = 2704 rows.

  Head (arch "nnue", quantisation-friendly):
      x = crelu(A_stm / D) ++ crelu(A_opp / D)            -> u8  (scale 128)
      h1 = crelu(W1 x + Wcap[caps] + b1)                  -> i8 weights, u8 out
      h2 = crelu(W2 h1 + b2)
      out = Wo h2 + bo                                    -> logit of stm win
  With --qat the forward pass simulates the integer pipeline (straight-through
  rounding), so the exported int8/int16 network loses nothing.
  Engine eval = logit * K.

Training reads every position once into a GPU cache of unique window / cell
indices with multiplicities; a batch is a dense count matrix and the
accumulator is a plain fp32 GEMM (forward and backward), in full IEEE fp32.
"""
import argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import *


def unique_counts(x, valid, sentinel):
    """Per row: unique values of x (where valid) and their multiplicities."""
    x = torch.where(valid, x, torch.full_like(x, sentinel))
    s, _ = torch.sort(x, dim=1)
    first = torch.ones_like(s, dtype=torch.bool)
    first[:, 1:] = s[:, 1:] != s[:, :-1]
    gid = torch.cumsum(first.long(), 1) - 1
    cnt = torch.zeros_like(s).scatter_add_(1, gid, torch.ones_like(s))
    val = torch.zeros_like(s).scatter_(1, gid, s)
    k = int(first.sum(1).max().item())
    val, cnt = val[:, :k], cnt[:, :k]
    pad = (val == sentinel) | (cnt == 0)
    return torch.where(pad, torch.zeros_like(val), val), torch.where(pad, torch.zeros_like(cnt), cnt)


class Cache:
    """All positions, precomputed once on the GPU as lists of unique
    (absolute window, count) and (cell index, count).

    The all-empty window (T row 0, fixed at 0) and the plain empty cell state
    (cell index 0, whose count is n_empty minus the listed cells) are not
    stored; every remaining multiplicity fits in a byte. Chunks are built on
    the GPU, parked in host memory, and copied into exact-size tensors (no 2x
    peak from padding + concatenation)."""

    def __init__(self, feats, raw, chunk=32768):
        self.feats = feats
        dev = raw.device
        parts = []
        with torch.no_grad():
            for i in range(0, len(raw), chunk):
                f = feats.absolute(raw[i:i + chunk])
                e4 = f["empty"].repeat_interleave(4, dim=1)
                wv, wc = unique_counts(f["w"], e4 & (f["w"] != 0), 1 << 16)
                cv, cc = unique_counts(f["cell"], f["empty"] & (f["cell"] != 0), NP)
                assert int(wc.max()) < 256 and int(cc.max()) < 256
                meta = torch.stack([f["side"].float(), f["cap"][:, 0].float(), f["cap"][:, 1].float(),
                                    f["result"], f["score"], f["empty"].sum(1).float()], 1)
                parts.append(dict(wu=torch.where(wv >= 32768, wv - 65536, wv).to(torch.int16).cpu(),
                                  wc=wc.to(torch.uint8).cpu(), cu=cv.to(torch.int16).cpu(),
                                  cc=cc.to(torch.uint8).cpu(), meta=meta.cpu()))
        self.n = len(raw)
        for k in ("wu", "wc", "cu", "cc", "meta"):
            width = max(p[k].shape[1] for p in parts)
            out = torch.zeros((self.n, width), dtype=parts[0][k].dtype, device=dev)
            j = 0
            for p in parts:
                x = p.pop(k)
                out[j:j + len(x), : x.shape[1]] = x.to(dev)
                j += len(x)
            setattr(self, k, out)

    def __len__(self):
        return self.n

    def batch(self, idx):
        fe = self.feats
        w = self.wu[idx].long() & 0xFFFF
        cu = self.cu[idx].long()
        m = self.meta[idx]
        side = m[:, 0].long()
        white = (side == 1).view(-1, 1)
        vb, vw = fe.vocab[fe.canon[w]], fe.vocab[fe.canon[fe.swap[w]]]
        pw = fe.cperm[cu]
        cap0, cap1 = m[:, 1].long(), m[:, 2].long()
        ccnt = self.cc[idx].float()
        # append the implicit plain-empty cell state (index 0, same in both perspectives)
        n0 = (m[:, 5] - ccnt.sum(1)).view(-1, 1)
        z = torch.zeros_like(n0, dtype=torch.long)
        cs, co = torch.where(white, pw, cu), torch.where(white, cu, pw)
        return dict(v_stm=torch.where(white, vw, vb), v_opp=torch.where(white, vb, vw), wcnt=self.wc[idx].float(),
                    c_stm=torch.cat([cs, z], 1), c_opp=torch.cat([co, z], 1), ccnt=torch.cat([ccnt, n0], 1),
                    cap_stm=torch.where(side == 1, cap1, cap0), cap_opp=torch.where(side == 1, cap0, cap1),
                    result=m[:, 3], score=m[:, 4], n_empty=m[:, 5])


# ---------------------------------------------------------------- model

def ste_round(x):
    return x + (torch.round(x) - x).detach()


def ste_floor(x):
    return x + (torch.floor(x) - x).detach()


class Net(nn.Module):
    def __init__(self, V, C=32, H1=32, H2=32, qat=False):
        super().__init__()
        self.V, self.C, self.H1, self.H2, self.qat = V, C, H1, H2, qat
        self.qscale = 0.0  # table quantisation scale used by QAT (set from the data)
        self.T = nn.Parameter(torch.randn(V, C) * 0.01)
        self.P = nn.Parameter(torch.zeros(NP, C))
        self.bias = nn.Parameter(torch.full((C,), 0.5 * D))  # feature bias (inside A)
        self.l1 = nn.Linear(2 * C, H1)
        self.cap = nn.Parameter(torch.zeros(12, H1))
        self.l2 = nn.Linear(H1, H2)
        self.out = nn.Linear(H2, 1)

    def tables(self):
        """(T, P - P[0], P[0], feature bias), in the form the engine stores them:
        row 0 of T (all-empty window) is fixed at 0, the plain empty cell P[0]
        becomes a per-empty-cell bias. With QAT every value sits exactly on the
        exported integer grid (tables at Q, biases at Q * 64)."""
        T = self.T.clamp(-TMAX, TMAX)
        T = torch.cat([torch.zeros_like(T[:1]), T[1:]])
        P = self.P.clamp(-TMAX, TMAX)
        p0, bias = P[:1], self.bias
        P = P - p0
        if self.qat and self.qscale > 0:
            Q, hp = self.qscale, self.qscale * 64
            T = ste_round(T * Q) / Q
            P = ste_round(P * Q) / Q
            p0 = ste_round(p0 * hp) / hp
            bias = ste_round(bias * hp) / hp
        return T, P, p0[0], bias

    def acc(self, v, wcnt, c, ccnt, n_empty):
        T, P, p0, bias = self.tables()
        B = v.shape[0]
        cw = torch.zeros((B, self.V), device=v.device).scatter_add_(1, v, wcnt)
        cp = torch.zeros((B, NP), device=v.device).scatter_add_(1, c, ccnt)
        # The GEMMs sum values on the 1/Q grid, which fp32 does exactly; the
        # large per-empty-cell term goes through fp64 so that A lands on the
        # same grid as the engine's integer accumulator.
        rel = cw @ T + cp @ P
        return rel.double() + (bias + n_empty.view(-1, 1) * p0).double()

    def act(self, x):
        x = torch.clamp(x, 0, 127.0 / ACT)
        return ste_floor(x * ACT) / ACT if self.qat else x

    def wq(self, w):
        w = w.clamp(-WMAX, WMAX)
        return ste_round(w * WS) / WS if self.qat else w

    def bq(self, b):
        return ste_round(b * ACT * WS) / (ACT * WS) if self.qat else b

    def forward(self, f):
        a_s = self.acc(f["v_stm"], f["wcnt"], f["c_stm"], f["ccnt"], f["n_empty"])
        a_o = self.acc(f["v_opp"], f["wcnt"], f["c_opp"], f["ccnt"], f["n_empty"])
        x = self.act((torch.cat([a_s, a_o], 1) / D).float())
        capb = self.bq(self.cap)
        h = x @ self.wq(self.l1.weight).t() + self.bq(self.l1.bias) + capb[f["cap_stm"]] + capb[6 + f["cap_opp"]]
        h = self.act(h)
        h = self.act(h @ self.wq(self.l2.weight).t() + self.bq(self.l2.bias))
        return (h @ self.wq(self.out.weight).t()).squeeze(1) + self.bq(self.out.bias)


# ---------------------------------------------------------------- quantisation helpers

def reparam(net):
    """The exported tables: the most frequent rows (all-empty window, plain
    empty cell) are 0 and the empty cell's value is a per-empty-cell bias cb,
    so they cost no int16 rounding ~1000 times per position."""
    T, P, p0, _ = net.tables()
    return T.detach(), P.detach(), p0.detach()


def empty_batch(feats, dev):
    return Cache(feats, torch.zeros((1, REC), dtype=torch.uint8, device=dev)).batch(torch.zeros(1, dtype=torch.long, device=dev))


def acc_range(net, feats, cache):
    """Largest |A'' - A''_empty| (float units) with the reparametrised
    tables; the engine keeps it in wrapping int16."""
    T2, P2, _ = reparam(net)
    dev = cache.wu.device

    def acc(f, side):
        B = f["v_" + side].shape[0]
        cw = torch.zeros((B, net.V), device=dev).scatter_add_(1, f["v_" + side], f["wcnt"])
        cp = torch.zeros((B, NP), device=dev).scatter_add_(1, f["c_" + side], f["ccnt"])
        return cw @ T2 + cp @ P2

    with torch.no_grad():
        e = empty_batch(feats, dev)
        a0 = acc(e, "stm")
        mx = 0.0
        for i in range(0, min(len(cache), 1000000), 16384):
            f = cache.batch(torch.arange(i, min(i + 16384, len(cache)), device=dev))
            for side in ("stm", "opp"):
                mx = max(mx, (acc(f, side) - a0).abs().max().item())
    return mx


def qlog_for(maxdev):
    """Largest power-of-two table scale keeping |A - A_empty| * Q in int16 (with margin)."""
    return int(min(11, math.floor(math.log2(30000.0 / max(maxdev, 1e-3)))))


def fit_k(score, result):
    """Scale K such that sigmoid(score/K) best predicts the game result."""
    m = score.abs() < 20000
    s, r = score[m], (result[m] + 1) / 2
    best = None
    for K in [50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]:
        l = F.binary_cross_entropy(torch.sigmoid(s / K), r).item()
        if best is None or l < best[1]:
            best = (K, l)
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="directories with part-*.bin")
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", default="value.pt", help="checkpoint (export with train/export.py)")
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--H1", type=int, default=32)
    ap.add_argument("--H2", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=0.5, help="weight of the game result vs search score")
    ap.add_argument("--K", type=float, default=800)
    ap.add_argument("--qat", action="store_true", help="quantisation-aware training")
    ap.add_argument("--qat-epochs", type=int, default=0, help="enable QAT for the last N epochs")
    ap.add_argument("--init", default=None, help="start from a checkpoint")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()
    dev = setup_cuda()
    train_paths, val_paths = split_paths(args.data)
    tr, va = load_records(train_paths), load_records(val_paths)
    keep = tr["flags"] & 1 == 0
    tr, va = tr[keep], va[va["flags"] & 1 == 0]
    print(f"train {len(tr)} val {len(va)} positions", flush=True)
    feats = Features(dev, args.tables)
    t0 = time.time()
    tr_c = Cache(feats, torch.tensor(tr.view(np.uint8).reshape(-1, REC), device=dev))
    va_raw = torch.tensor(va.view(np.uint8).reshape(-1, REC), device=dev)
    va_c = Cache(feats, va_raw)
    print(f"cache {time.time() - t0:.0f}s, vocab {feats.V} windows", flush=True)
    K = args.K if args.K > 0 else fit_k(tr_c.meta[:200000, 4], tr_c.meta[:200000, 3])

    net = Net(feats.V, args.C, args.H1, args.H2, qat=args.qat).to(dev)
    if args.init:
        net.load_state_dict(load_checkpoint(args.init, dev)[0].state_dict())
    epochs = args.epochs
    decay = [p for n_, p in net.named_parameters() if n_ in ("l1.weight", "l2.weight", "out.weight")]
    rest = [p for n_, p in net.named_parameters() if n_ not in ("l1.weight", "l2.weight", "out.weight")]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": rest, "weight_decay": 0}],
                            lr=args.lr, fused=True)
    steps = max(epochs * (len(tr_c) // args.batch), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)

    def target(f):  # game result + search score
        return args.lam * (f["result"] + 1) / 2 + (1 - args.lam) * torch.sigmoid(f["score"] / K)

    def evaluate():
        net.eval()
        tot, n, acc = 0.0, 0, 0.0
        with torch.no_grad():
            for i in range(0, len(va_c), 16384):
                f = va_c.batch(torch.arange(i, min(i + 16384, len(va_c)), device=dev))
                out = net(f)
                tot += F.binary_cross_entropy_with_logits(out, target(f), reduction="sum").item()
                dec = f["result"] != 0
                acc += ((out > 0) == (f["result"] > 0))[dec].float().sum().item()
                n += len(out)
        net.train()
        return tot / n, acc / max(n, 1)

    def _loss(idx):
        f = tr_c.batch(idx)
        return F.binary_cross_entropy_with_logits(net(f), target(f))

    compiled = {}

    def loss_fn(idx):
        key = net.qat
        if key not in compiled:
            compiled[key] = _loss if args.no_compile else torch.compile(_loss, dynamic=False)
        return compiled[key](idx)

    step = 0
    for ep in range(epochs):
        if (args.qat_epochs and ep == epochs - args.qat_epochs) or (args.qat and ep == 0):
            net.qat = True
            net.qscale = 2.0 ** qlog_for(acc_range(net, feats, tr_c))
            print(f"QAT on, table scale Q = {net.qscale}", flush=True)
        perm = torch.randperm(len(tr_c), device=dev)
        t0 = time.time()
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            loss = loss_fn(perm[i:i + args.batch])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step < steps:
                sched.step()
            step += 1
        vl, vacc = evaluate()
        print(f"epoch {ep} val {vl:.4f} val_acc {vacc:.3f} qat={net.qat} ({time.time() - t0:.0f}s)", flush=True)
        save_checkpoint(net, K, args.out)

    net.qat = True
    vl, vacc = evaluate()
    print(f"final (quantised forward) val {vl:.4f} val_acc {vacc:.3f}", flush=True)
    if net.qscale == 0:
        net.qscale = 2.0 ** qlog_for(acc_range(net, feats, tr_c))
    save_checkpoint(net, K, args.out)


def save_checkpoint(net, K, path):
    torch.save(dict(state=net.state_dict(), V=net.V, C=net.C, H1=net.H1, H2=net.H2, K=float(K),
                    qscale=float(net.qscale)), path)
    print(f"saved {path}")


def load_checkpoint(path, dev):
    ck = torch.load(path, map_location=dev)
    if "state" not in ck:  # bare state_dict (older runs)
        st = ck
        ck = dict(state=st, V=st["T"].shape[0], C=st["T"].shape[1], H1=st["l1.weight"].shape[0],
                  H2=st["l2.weight"].shape[0], K=800.0, qscale=0.0)
    net = Net(ck["V"], ck["C"], ck["H1"], ck["H2"], qat=True).to(dev)
    net.load_state_dict(ck["state"])
    net.qscale = ck["qscale"]
    return net, ck["K"]


if __name__ == "__main__":
    main()
