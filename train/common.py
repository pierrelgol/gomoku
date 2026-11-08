"""Shared pieces of the training pipeline: record format, the engine's
pattern tables, window vocabulary, rules-aware cell index and GPU feature
extraction. Mirrors src/engine/{board,nnue,policy}.zig exactly."""
import glob, os
import numpy as np
import torch

REC = 104
N = 19
D = 32.0         # accumulator divisor before the feature activation
TMAX = 15.9      # bound on table values (int16 range at the export scale)
NP = 13 * 13 * 16
ACT = 128.0      # activation scale (u8, max 127)
WS = 64.0        # weight scale (int8, |w| <= 127/64)
WMAX = 127.0 / WS

rec_dtype = np.dtype([
    ("cells", np.uint8, 91), ("side", np.uint8), ("captured", np.uint8, 2),
    ("result", np.int8), ("flags", np.uint8), ("score", "<i2"), ("best", "<u2"),
    ("ply", "<u2"), ("pad", np.uint8, 2)])
assert rec_dtype.itemsize == REC



def load_tables(path):
    blob = np.fromfile(path, dtype=np.uint8)
    line = blob[: 2 << 16].view("<u2").astype(np.int64)          # [65536] LineInfo b | w<<8
    comb = blob[2 << 16: (2 << 16) + (1 << 16)].astype(np.int64)  # [65536] CellInfo
    return line, comb


def window_maps():
    w = np.arange(1 << 16, dtype=np.int64)
    swap = ((w & 0x5555) << 1) | ((w >> 1) & 0x5555)
    mirror = np.zeros_like(w)
    for i in range(8):
        mirror |= ((w >> (2 * i)) & 3) << (2 * (7 - i))
    return swap, np.minimum(w, mirror)


def valid_window(w):
    """Walls may only appear as contiguous blocks at the outer ends."""
    for side in (range(3, -1, -1), range(4, 8)):  # fields from the centre outwards
        wall = False
        for i in side:
            c = (w >> (2 * i)) & 3
            if wall and c != 3:
                return False
            wall = wall or c == 3
    return True


def window_vocab(canon):
    """Compact id for every canonical valid window. Id 0 = all-empty window."""
    vocab = np.zeros(1 << 16, dtype=np.int64)
    ids = {0: 0}
    for w in range(1 << 16):
        c = int(canon[w])
        if not valid_window(c):
            continue
        if c not in ids:
            ids[c] = len(ids)
        vocab[w] = ids[c]
    return vocab, len(ids)


def cell_index(p4o, p4x, capo, capx, fo, fx):
    return ((p4o * 13 + p4x) * 4 + capo * 2 + capx) * 4 + fo * 2 + fx


def cell_perm():
    """Black-perspective cell index -> white-perspective index."""
    i = np.arange(NP)
    f, c, pp = i % 4, (i // 4) % 4, i // 16
    p4b, p4w = pp // 13, pp % 13
    return cell_index(p4w, p4b, c & 1, c >> 1, f & 1, f >> 1)


def load_records(paths):
    arrs = [np.fromfile(p, dtype=rec_dtype) for p in paths]
    return np.concatenate(arrs) if arrs else np.zeros(0, rec_dtype)


class Features:
    """GPU feature extraction from raw records (colour-absolute)."""

    def __init__(self, dev, tables):
        line, comb = load_tables(tables)
        swap, canon = window_maps()
        vocab, self.V = window_vocab(canon)
        t = lambda a: torch.tensor(a, device=dev)
        self.line, self.comb = t(line), t(comb)
        self.swap, self.canon, self.vocab = t(swap), t(canon), t(vocab)
        self.cperm = t(cell_perm())
        self.dev = dev

    def absolute(self, raw):
        B = raw.shape[0]
        dev = self.dev
        cells = raw[:, :91].long()
        codes = ((cells.unsqueeze(-1) >> (torch.arange(4, device=dev) * 2)) & 3).reshape(B, -1)[:, : N * N]
        pad = torch.full((B, N + 8, N + 8), 3, dtype=torch.long, device=dev)
        pad[:, 4:4 + N, 4:4 + N] = codes.view(B, N, N)
        wins = []
        for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1)):
            w = torch.zeros((B, N, N), dtype=torch.long, device=dev)
            for k in (-4, -3, -2, -1, 1, 2, 3, 4):
                sh = 2 * (k + 4) if k < 0 else 2 * (k - 1) + 8
                w |= pad[:, 4 + k * dy: 4 + k * dy + N, 4 + k * dx: 4 + k * dx + N] << sh
            wins.append(w.reshape(B, -1))
        w = torch.stack(wins, -1)                       # [B,361,4] absolute window index
        li = self.line[w]                               # LineInfo black | white << 8
        sh4 = torch.arange(4, device=dev) * 4
        cb = self.comb[((li & 0xF) << sh4).sum(-1)]
        cw = self.comb[(((li >> 8) & 0xF) << sh4).sum(-1)]
        p4b, p4w = cb & 0xF, cw & 0xF
        capb = (((li >> 4) & 3).sum(-1) > 0).long()
        capw = (((li >> 12) & 3).sum(-1) > 0).long()
        # forbidden double-three (Board.isForbidden): flag, no capture, not a five
        fb = (((cb >> 4) & 1) == 1) & (capb == 0) & (p4b != 12)
        fw = (((cw >> 4) & 1) == 1) & (capw == 0) & (p4w != 12)
        cell = cell_index(p4b, p4w, capb, capw, fb.long(), fw.long())
        score = raw[:, 96].long() | (raw[:, 97].long() << 8)
        return dict(w=w.reshape(B, -1), cell=cell, empty=(codes == 0),
                    side=raw[:, 91].long(), cap=raw[:, 92:94].long().clamp(max=5),
                    result=raw[:, 94].view(torch.int8).float(), flags=raw[:, 95].long(),
                    score=torch.where(score >= 32768, score - 65536, score).float())


def split_paths(dirs):
    """Training / validation part files: the last 2 parts of every data
    directory are the validation set."""
    train, val = [], []
    for d in dirs:
        ps = sorted(glob.glob(os.path.join(d, "part-*.bin")))
        train += ps[:-2]
        val += ps[-2:]
    return train, val


def setup_cuda():
    """Full IEEE fp32 everywhere (no TF32)."""
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return torch.device("cuda")
