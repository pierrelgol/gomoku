#!/usr/bin/env python3
"""Learned move ordering (policy) for the search, trained on self-play records.

For a candidate cell q, with "own" = side to move, the score is a sum of table
lookups on state the engine already has for every cell:

    s(q) = Pc[cell_own(q)]                 rules-aware cell state (2704 rows:
                                           Pattern4 own/opp, capture here,
                                           forbidden double-three)
         + sum_d L[lineinfo_own(q, d)]     joint per-direction line info (own
                                           pattern, opp pattern, capture flags)
         + sum_d W[window_own(q, d)]       raw 8-neighbour window, mirror-tied
                                           (optional, --windows)
         + Co[capn_own(q), captured_own] + Cx[capn_opp(q), captured_opp]

Trained with a softmax over the engine's candidate cells (empty, inside the
5x5 square or at distance 3 on a line of a stone, not forbidden) against the
search's best move. Exported as int32 tables ("GPOL"), read by
src/engine/policy.zig (export with train/export.py policy); it's a linear model, so the integer policy matches the float one up to rounding.

GPU path: the raw records (104 bytes each) live on the GPU; every batch's
features are computed on the fly (a few elementwise kernels), so there is no
cache and no host traffic. Table gradients use bincount (block-local shared
memory histograms) instead of scatter_add atomics, which serialise on the
very frequent rows (empty windows, plain cells).
"""
import argparse, glob, os, struct, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import *
import common as tr

N = 19
NEG = -1.0e4


class Lookup(torch.autograd.Function):
    """table[idx] with a contention-free backward (bincount)."""

    @staticmethod
    def forward(ctx, table, idx):
        ctx.save_for_backward(idx)
        ctx.n = table.shape[0]
        return table[idx]

    @staticmethod
    def backward(ctx, g):
        (idx,) = ctx.saved_tensors
        gt = torch.bincount(idx.reshape(-1), weights=g.reshape(-1).float(), minlength=ctx.n)
        return gt.to(g.dtype), None


lookup = Lookup.apply


def candidate_kernel(dev):
    k = torch.zeros((1, 1, 7, 7), device=dev)
    k[0, 0, 1:6, 1:6] = 1
    for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1), (-1, 0), (0, -1), (-1, -1), (1, -1)):
        k[0, 0, 3 + 3 * dy, 3 + 3 * dx] = 1
    k[0, 0, 3, 3] = 0
    return k


class PFeatures:
    """Per-cell policy features of raw records, all in the side-to-move perspective."""

    def __init__(self, dev, tables, double_three=True):
        self.fe = tr.Features(dev, tables)
        self.double_three = double_three
        self.ker = candidate_kernel(dev)
        # compact ids of the joint LineInfo values (own byte | opp byte << 8)
        line = self.fe.line.cpu().numpy()
        vals = np.unique(line)
        lv = np.zeros(1 << 16, dtype=np.int64)
        lv[vals] = np.arange(len(vals))
        self.lvals = vals
        self.lvocab = torch.tensor(lv, device=dev)
        self.NL = len(vals)
        self.V = self.fe.V

    def __call__(self, raw):
        fe = self.fe
        f = fe.absolute(raw)
        B = raw.shape[0]
        white = (f["side"] == 1).view(B, 1)
        w = f["w"].view(B, N * N, 4)
        ws = torch.where(white.view(B, 1, 1), fe.swap[w], w)       # own-perspective windows
        li = fe.line[ws]                                            # own low byte, opp high byte
        cell = torch.where(white, fe.cperm[f["cell"]], f["cell"])   # own-perspective cell state
        capo = ((li >> 4) & 3).sum(-1).clamp(max=3)
        capx = ((li >> 12) & 3).sum(-1).clamp(max=3)
        cap = f["cap"]
        cown = torch.where(f["side"] == 1, cap[:, 1], cap[:, 0]).view(B, 1)
        copp = torch.where(f["side"] == 1, cap[:, 0], cap[:, 1]).view(B, 1)
        stones = (~f["empty"]).float().view(B, 1, N, N)
        near = F.conv2d(stones, self.ker, padding=3).view(B, N * N) > 0
        cand = near & f["empty"]
        if self.double_three:  # the engine only skips forbidden cells under the ban
            cand = cand & (((cell >> 1) & 1) == 0)
        best = (raw[:, 98].long() | (raw[:, 99].long() << 8)).clamp(max=N * N - 1)
        return dict(cell=cell, lin=self.lvocab[li], win=fe.vocab[fe.canon[ws]],
                    co=capo * 6 + cown, cx=capx * 6 + copp, cand=cand, best=best)


class Policy(nn.Module):
    def __init__(self, NL, V, windows):
        super().__init__()
        self.windows = windows
        self.Pc = nn.Parameter(torch.zeros(tr.NP))
        self.L = nn.Parameter(torch.zeros(NL))
        self.W = nn.Parameter(torch.zeros(V)) if windows else None
        self.Co = nn.Parameter(torch.zeros(24))
        self.Cx = nn.Parameter(torch.zeros(24))

    def forward(self, f):
        s = lookup(self.Pc, f["cell"]) + lookup(self.L, f["lin"]).sum(-1)
        if self.windows:
            s = s + lookup(self.W, f["win"]).sum(-1)
        s = s + lookup(self.Co, f["co"]) + lookup(self.Cx, f["cx"])
        return torch.where(f["cand"], s, torch.full_like(s, NEG))


def load(paths, dev):
    r = tr.load_records(paths)
    return torch.tensor(r.view(np.uint8).reshape(-1, tr.REC), device=dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", default="policy.pt", help="checkpoint (export with train/export.py)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--windows", action="store_true", help="experimental raw-window features (not used by the engine)")
    ap.add_argument("--max", type=int, default=0, help="subsample the training set (0 = all)")
    ap.add_argument("--profile", type=int, default=0, help="time N steps, print the breakdown, exit")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-double-three", action="store_true", help="data played without the double-three ban")
    args = ap.parse_args()
    dev = setup_cuda()
    train_paths, val_paths = split_paths(args.data)
    t0 = time.time()
    tr_r = load(train_paths, dev)
    va_r = load(val_paths, dev)[:400000]
    if args.max and len(tr_r) > args.max:
        tr_r = tr_r[torch.randperm(len(tr_r), device=dev)[: args.max]]
    pf = PFeatures(dev, args.tables, double_three=not args.no_double_three)
    feats = pf if args.no_compile else torch.compile(pf.__call__, dynamic=False)
    print(f"train {len(tr_r)} val {len(va_r)} positions, {pf.NL} line-info values, {pf.V} windows "
          f"({time.time() - t0:.0f}s)", flush=True)

    net = Policy(pf.NL, pf.V, args.windows).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, fused=True)
    nb = len(tr_r) // args.batch
    steps = max(args.epochs * nb, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.03)

    def step_loss(raw):
        f = feats(raw)
        lg = net(f)
        ok = f["cand"].gather(1, f["best"].view(-1, 1)).view(-1)   # best move is a candidate
        # ignore_index instead of boolean indexing: no host sync per step
        return F.cross_entropy(lg, torch.where(ok, f["best"], -100), ignore_index=-100)

    if args.profile:
        perm = torch.randperm(len(tr_r), device=dev)
        ev = lambda: torch.cuda.Event(enable_timing=True)
        tf = tb = 0.0
        for i in range(args.profile + 5):
            raw = tr_r[perm[i * args.batch:(i + 1) * args.batch]]
            a, b_, c = ev(), ev(), ev()
            a.record()
            loss = step_loss(raw)
            b_.record()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            c.record()
            torch.cuda.synchronize()
            if i >= 5:
                tf += a.elapsed_time(b_)
                tb += b_.elapsed_time(c)
        n = args.profile
        print(f"batch {args.batch}: forward+features {tf / n:.2f} ms, backward+step {tb / n:.2f} ms, "
              f"{args.batch * n / (tf + tb) * 1000:.0f} positions/s, peak mem "
              f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        return

    def evaluate():
        tot, n, hit = 0.0, 0, torch.zeros(3, device=dev)
        with torch.no_grad():
            for i in range(0, len(va_r), 8192):
                f = feats(va_r[i:i + 8192])
                lg = net(f)
                ok = f["cand"].gather(1, f["best"].view(-1, 1)).view(-1)
                lg, y = lg[ok], f["best"][ok]
                tot += F.cross_entropy(lg, y, reduction="sum").item()
                rk = lg.topk(5, 1).indices == y.view(-1, 1)
                hit += torch.stack([rk[:, :1].any(1), rk[:, :3].any(1), rk.any(1)]).float().sum(1)
                n += len(y)
        h = (hit / n).tolist()
        return tot / n, h

    for ep in range(args.epochs):
        t0 = time.time()
        perm = torch.randperm(len(tr_r), device=dev)
        run = torch.zeros((), device=dev)
        for i in range(nb):
            loss = step_loss(tr_r[perm[i * args.batch:(i + 1) * args.batch]])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.detach()
        l, h = evaluate()
        print(f"epoch {ep} train ce {run.item() / nb:.4f} val ce {l:.4f} top1 {h[0]:.3f} top3 {h[1]:.3f} "
              f"top5 {h[2]:.3f} ({time.time() - t0:.0f}s)", flush=True)

    save_policy(net, args.out)


def save_policy(net, path):
    torch.save(dict(state=net.state_dict(), windows=net.windows), path)
    print(f"saved {path}")


def load_policy(path, pf, dev):
    ck = torch.load(path, map_location=dev)
    net = Policy(pf.NL, pf.V, ck["windows"]).to(dev)
    net.load_state_dict(ck["state"])
    if net.windows:
        raise SystemExit("window features are not supported by the engine")
    return net


if __name__ == "__main__":
    main()
