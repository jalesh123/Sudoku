"""
sudoku_gf_cipher.py
--------------------

This module implements, exactly as specified in the  manuscript:
  1. GF(2^8) element-sequence generation from a degree-8 primitive
     polynomial over GF(2), via the standard LFSR recursion.
  2. Deterministic, seed-driven generation of a valid N x N Sudoku
     matrix (N a perfect square) using the base-pattern + band/stack
     shuffle + symbol-relabelling method.
  3. The full encryption / decryption pipeline: for every pixel, the
     Sudoku matrix selects which GF-derived substitution table
     (permutation) is applied to that pixel's value.
  4. Utility functions for the secret key packaging and key-space
     accounting used in Section 6 of the manuscript.
"""

from __future__ import annotations
import numpy as np
import hashlib
import math
from dataclasses import dataclass, field
from typing import List, Tuple

# --------------------------------------------------------------------------
# 1. Galois Field GF(2^8) generation from a primitive polynomial
# --------------------------------------------------------------------------

# The three base-level degree-8 primitive polynomials used in the paper,
# encoded as the low-order 8 bits a7 a6 ... a0 of  x^8 + a7 x^7 + ... + a0
# (the x^8 term itself is implicit / not stored -- standard convention).
#  
#  --------------------------------------------------------------------
#  The originally submitted manuscript labelled x^8+x^4+x^3+x+1 (0x11B,
#  the AES/Rijndael modulus) and x^8+x^6+x^5+x^3+1 (0x16D) as "primitive
#  polynomials". Both are *irreducible* but NOT primitive: under the
#  LFSR/power-of-x construction used to build the GF sequence (Section
#  3, Stage 1), only a polynomial for which x itself is a primitive
#  root generates the full cycle of 255 non-zero elements. 0x11B and
#  0x16D fail this test (verified below with verify_primitive()) and
#  were replaced with two of the 16 verified primitive degree-8
#  polynomials over GF(2). This addresses Reviewer 1's request to
#  justify the polynomial choice and confirm it is not an
#  unexamined reuse of the AES modulus.
PRIMITIVE_POLYS = {
    "A": 0x1D,  # x^8 + x^4 + x^3 + x^2 + 1   (verified primitive)
    "B": 0x2B,  # x^8 + x^5 + x^3 + x   + 1   (verified primitive)
    "C": 0x63,  # x^8 + x^6 + x^5 + x   + 1   (verified primitive)
}


def verify_primitive(poly: int, n: int = 8) -> bool:
    """Verify a degree-n polynomial (given as its low-n bits, x^n implicit)
    is primitive over GF(2): it must generate all 2^n - 1 non-zero field
    elements before repeating (order = 2^n - 1)."""
    seen = set()
    g = 1
    for _ in range(2 ** n - 1):
        if g in seen:
            return False
        seen.add(g)
        msb = (g >> (n - 1)) & 1
        g = (g << 1) & (2 ** n - 1)
        if msb:
            g ^= poly
    return len(seen) == 2 ** n - 1


def gf_sequence(poly: int, n: int = 8) -> List[int]:
    """Generate the ordered sequence g^0, g^1, ..., g^(2^n-2) of the
    2^n - 1 non-zero elements of GF(2^n) defined by primitive polynomial
    `poly`, using the recursive left-shift-and-reduce rule described in
    Section 3, Stage 1 of the manuscript."""
    seq = [1]  # g^0
    g = 1
    for _ in range(2 ** n - 2):
        msb = (g >> (n - 1)) & 1
        g = (g << 1) & (2 ** n - 1)
        if msb:
            g ^= poly
        seq.append(g)
    return seq


def gf_permutation_table(poly: int, offset: int = 0, n: int = 8) -> np.ndarray:
    """Build a full 256-entry substitution table over byte values
    {0, ..., 255} from a primitive polynomial.

    Byte value 0 is a fixed point (GF(2^n) arithmetic has no additive
    representation for 0 in the multiplicative-order sequence), and
    bytes 1..255 are mapped through the cyclic sequence g^0..g^254,
    starting at a secret rotation `offset` (0..254). This yields a
    bijective S-box T: {0,...,255} -> {0,...,255} for every choice of
    (poly, offset).
    """
    seq = gf_sequence(poly, n)
    table = np.arange(256, dtype=np.uint8)
    rotated = seq[offset:] + seq[:offset]
    # bytes 1..255 map through the rotated cyclic GF sequence
    table[1:256] = np.array(rotated, dtype=np.uint8)
    return table


def inverse_table(table: np.ndarray) -> np.ndarray:
    inv = np.zeros_like(table)
    inv[table] = np.arange(len(table), dtype=table.dtype)
    return inv


# --------------------------------------------------------------------------
# 2. Deterministic, seeded Sudoku-matrix generation
# --------------------------------------------------------------------------

def _base_latin_sudoku(box: int) -> np.ndarray:
    """Canonical base pattern for an N x N Sudoku grid (N = box^2),
    using the standard shifted-band construction:
        f(r, c) = (box * (r % box) + r // box + c) % N
    which always satisfies row, column and box constraints."""
    N = box * box
    idx = np.arange(N)
    r = idx.reshape(-1, 1)
    c = idx.reshape(1, -1)
    grid = (box * (r % box) + r // box + c) % N
    return grid.astype(np.int32)


def generate_sudoku_matrix(box: int, seed: bytes) -> np.ndarray:
    """Deterministically generate a *valid* N x N Sudoku matrix
    (N = box*box) from a secret seed, following the standard
    validity-preserving group of Sudoku symmetries:
        - relabel the N symbols with a secret permutation,
        - permute the `box` rows within each of the `box` bands,
        - permute the `box` bands themselves,
        - permute the `box` columns within each of the `box` stacks,
        - permute the `box` stacks themselves,
        - optionally transpose the grid.
    Every operation in this group preserves the Sudoku row/column/box
    constraints, so the result is guaranteed valid for any seed -- this
    is what makes the matrix both reproducible (same seed -> same
    matrix, needed for decryption) and unpredictable (seed is secret).
    """
    N = box * box
    rng = np.random.default_rng(
        np.frombuffer(hashlib.sha256(seed).digest(), dtype=np.uint32)
    )
    grid = _base_latin_sudoku(box)

    # symbol relabelling
    symbol_perm = rng.permutation(N)
    grid = symbol_perm[grid]

    # row permutation within each band, then band permutation
    band_order = rng.permutation(box)
    row_perm = np.concatenate([
        b * box + rng.permutation(box) for b in band_order
    ])
    grid = grid[row_perm, :]

    # column permutation within each stack, then stack permutation
    stack_order = rng.permutation(box)
    col_perm = np.concatenate([
        s * box + rng.permutation(box) for s in stack_order
    ])
    grid = grid[:, col_perm]

    if rng.integers(0, 2) == 1:
        grid = grid.T.copy()

    return grid.astype(np.int32)


def validate_sudoku(grid: np.ndarray, box: int) -> bool:
    N = box * box
    full = set(range(N))
    for i in range(N):
        if set(grid[i, :].tolist()) != full or set(grid[:, i].tolist()) != full:
            return False
    for br in range(box):
        for bc in range(box):
            block = grid[br * box:(br + 1) * box, bc * box:(bc + 1) * box]
            if set(block.flatten().tolist()) != full:
                return False
    return True


# --------------------------------------------------------------------------
# 3. Secret key container and key-space accounting (Section 6)
# --------------------------------------------------------------------------

@dataclass
class CipherKey:
    poly_names: Tuple[str, ...]      # e.g. ("A","C") -- which base polynomials
    offsets: Tuple[int, ...]         # per-polynomial cyclic rotation, 0..254
    label_perm: Tuple[int, ...]      # assignment of polys to Sudoku label-groups
    sudoku_seed: bytes               # 128-bit secret seed for the Sudoku matrix
    box: int = 4                     # 4 -> 16x16 Sudoku matrix

    @property
    def k(self) -> int:
        return len(self.poly_names)

    def gf_tables(self) -> List[np.ndarray]:
        return [gf_permutation_table(PRIMITIVE_POLYS[p], off)
                for p, off in zip(self.poly_names, self.offsets)]

    def sudoku_matrix(self) -> np.ndarray:
        return generate_sudoku_matrix(self.box, self.sudoku_seed)


def key_space_report(box: int = 4, k: int = 3, sudoku_seed_bits: int = 128,
                     master_bits: int = 256, subset_secret: bool = False) -> dict:
    """Corrected key model (revision, Section 6.1).

    All subkeys are DERIVED from one master secret K (make_key), so they are
    not independent and their sizes must not be multiplied into "secret
    entropy". This function reports (i) the number of reachable derived
    configurations and (ii) the brute-force bound min(lambda, derived) + s.
    NOTE: the earlier version used len(PRIMITIVE_POLYS) == 3 here, which gave
    C(3,1)+C(3,2)+C(3,3) = 7 instead of the count over all 16 primitive
    degree-8 polynomials, C(16,1)+C(16,2)+C(16,3) = 696.
    """
    n_polys_total = 16                       # all primitive degree-8 polynomials over GF(2)
    subsets_all = sum(math.comb(n_polys_total, j) for j in range(1, k + 1))   # 696 for k = 3
    subsets_k = math.comb(n_polys_total, k)                                    # 560 for k = 3
    derived_bits = k * math.log2(255) + math.log2(math.factorial(k)) + sudoku_seed_bits
    s = math.log2(subsets_k) if subset_secret else 0.0
    return {
        "subsets_1_to_k": subsets_all, "subsets_1_to_k_bits": math.log2(subsets_all),
        "subsets_exactly_k": subsets_k, "subsets_exactly_k_bits": math.log2(subsets_k),
        "offset_bits": k * math.log2(255), "label_perm_bits": math.log2(math.factorial(k)),
        "sudoku_seed_bits": sudoku_seed_bits,
        "reachable_derived_configs_bits": derived_bits + math.log2(subsets_k),
        "derived_material_fixed_subset_bits": derived_bits,
        "master_secret_bits": master_bits,
        "brute_force_bound_bits": min(master_bits, derived_bits) + s,
    }


# --------------------------------------------------------------------------
# 4. Encryption / decryption pipeline (Section 3, Stage 3 / Algorithm 1)
# --------------------------------------------------------------------------

def _encrypt_channel(channel: np.ndarray, sudoku: np.ndarray,
                      gf_tables: List[np.ndarray], label_perm) -> np.ndarray:
    H, W = channel.shape
    N = sudoku.shape[0]
    k = len(gf_tables)
    tiled = np.tile(sudoku, (int(np.ceil(H / N)), int(np.ceil(W / N))))[:H, :W]
    group = tiled % k
    group = np.asarray(label_perm)[group]
    out = np.empty_like(channel)
    for idx in range(k):
        mask = group == idx
        out[mask] = gf_tables[idx][channel[mask]]
    return out


def _decrypt_channel(channel: np.ndarray, sudoku: np.ndarray,
                      inv_tables: List[np.ndarray], label_perm) -> np.ndarray:
    H, W = channel.shape
    N = sudoku.shape[0]
    k = len(inv_tables)
    tiled = np.tile(sudoku, (int(np.ceil(H / N)), int(np.ceil(W / N))))[:H, :W]
    group = tiled % k
    group = np.asarray(label_perm)[group]
    out = np.empty_like(channel)
    for idx in range(k):
        mask = group == idx
        out[mask] = inv_tables[idx][channel[mask]]
    return out


def encrypt_image(img: np.ndarray, key: CipherKey) -> np.ndarray:
    """img: HxW (grayscale, uint8) or HxWx3 (RGB, uint8)."""
    sudoku = key.sudoku_matrix()
    gf_tables = key.gf_tables()
    if img.ndim == 2:
        return _encrypt_channel(img, sudoku, gf_tables, key.label_perm)
    else:
        chans = [_encrypt_channel(img[:, :, c], sudoku, gf_tables, key.label_perm)
                  for c in range(img.shape[2])]
        return np.stack(chans, axis=-1)


def decrypt_image(cimg: np.ndarray, key: CipherKey) -> np.ndarray:
    sudoku = key.sudoku_matrix()
    inv_tables = [inverse_table(t) for t in key.gf_tables()]
    if cimg.ndim == 2:
        return _decrypt_channel(cimg, sudoku, inv_tables, key.label_perm)
    else:
        chans = [_decrypt_channel(cimg[:, :, c], sudoku, inv_tables, key.label_perm)
                  for c in range(cimg.shape[2])]
        return np.stack(chans, axis=-1)


def make_key(poly_names: Tuple[str, ...], master_secret: bytes,
             box: int = 4) -> CipherKey:
    """Derive a full CipherKey deterministically from a single master
    secret (as would be shared between sender/receiver), using SHA-256
    based domain separation for each sub-key -- this is what makes the
    whole scheme reproducible from one shared secret while keeping the
    component key space analysed in key_space_report()."""
    k = len(poly_names)
    offsets = []
    for i, p in enumerate(poly_names):
        h = hashlib.sha256(master_secret + f"|offset|{p}|{i}".encode()).digest()
        offsets.append(int.from_bytes(h[:2], "big") % 255)
    h = hashlib.sha256(master_secret + b"|label_perm").digest()
    rng = np.random.default_rng(np.frombuffer(h[:8], dtype=np.uint32))
    label_perm = tuple(rng.permutation(k).tolist())
    sudoku_seed = hashlib.sha256(master_secret + b"|sudoku").digest()[:16]
    return CipherKey(poly_names=tuple(poly_names), offsets=tuple(offsets),
                      label_perm=label_perm, sudoku_seed=sudoku_seed, box=box)


# --------------------------------------------------------------------------
# 5. Optional diffusion layer: Sudoku/key-driven pixel permutation
# --------------------------------------------------------------------------
#  RATIONALE (added during revision, see Results/Discussion):
#  A pure value-substitution cipher (Stage 3 above) is, per pixel, a
#  bijection on {0,...,255}. Shannon entropy is invariant under any
#  bijective relabelling of a random variable's outcomes, so with a
#  SINGLE substitution table applied uniformly (configs A, B, C) the
#  ciphertext entropy is mathematically guaranteed to equal the
#  plaintext entropy -- substitution alone cannot improve it. Mixing
#  k>=2 tables (configs D-G) breaks this invariance because different
#  pixels are no longer subject to the *same* bijection, which is why
#  D-G show materially higher entropy and lower correlation than A-C
#  in the experiments below. However, substitution alone -- regardless
#  of k -- never relocates pixels, so strong local correlation
#  partially survives. This is addressed by adding a lightweight
#  keyed permutation (diffusion) layer, in the spirit of a standard
#  substitution-permutation network, using the same secret material.
def _permutation_from_seed(n_pixels: int, seed: bytes) -> np.ndarray:
    rng = np.random.default_rng(
        np.frombuffer(hashlib.sha256(seed + b"|permute").digest(), dtype=np.uint32)
    )
    return rng.permutation(n_pixels)


def encrypt_image_sp(img: np.ndarray, key: CipherKey) -> np.ndarray:
    """Substitution-Permutation variant: keyed pixel permutation (diffusion)
    followed by the Sudoku/GF substitution (confusion) of Section 3."""
    if img.ndim == 2:
        H, W = img.shape
        perm = _permutation_from_seed(H * W, key.sudoku_seed)
        permuted = img.flatten()[perm].reshape(H, W)
        return encrypt_image(permuted, key)
    else:
        H, W, C = img.shape
        perm = _permutation_from_seed(H * W, key.sudoku_seed)
        chans = []
        for c in range(C):
            permuted = img[:, :, c].flatten()[perm].reshape(H, W)
            chans.append(_encrypt_channel(permuted, key.sudoku_matrix(),
                                           key.gf_tables(), key.label_perm))
        return np.stack(chans, axis=-1)


def decrypt_image_sp(cimg: np.ndarray, key: CipherKey) -> np.ndarray:
    sub = decrypt_image(cimg, key)
    if sub.ndim == 2:
        H, W = sub.shape
        perm = _permutation_from_seed(H * W, key.sudoku_seed)
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(len(perm))
        return sub.flatten()[inv_perm].reshape(H, W)
    else:
        H, W, C = sub.shape
        perm = _permutation_from_seed(H * W, key.sudoku_seed)
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(len(perm))
        chans = [sub[:, :, c].flatten()[inv_perm].reshape(H, W) for c in range(C)]
        return np.stack(chans, axis=-1)


# --------------------------------------------------------------------------
# 6. Diffusion layer: bidirectional ciphertext chaining
# --------------------------------------------------------------------------
#  RATIONALE (added during revision -- see Results, "Differential
#  attack resistance"): experiments showed that the substitution
#  (Section 4) and substitution+permutation (Section 5) schemes both
#  leave NPCR/UACI close to 0% under a single-pixel plaintext change,
#  because every ciphertext pixel is a function of exactly one
#  plaintext pixel (a permutation only relocates that dependency, it
#  does not spread it). This fails the standard differential-attack
#  requirement (NPCR ~99.6%, UACI ~33.46% [25]) and was identified as
#  a genuine weakness of the originally submitted design. It is fixed
#  with a lightweight bidirectional XOR-chaining diffusion pass
#  (forward, then backward, each conditioning every pixel's output on
#  the *previous* pixel's ciphertext) -- a standard, cheap way to make
#  every ciphertext byte depend on the entire plaintext, at O(M^2)
#  cost, no larger than the substitution stage itself.
def _iv_pair(seed: bytes):
    h = hashlib.sha256(seed + b"|iv").digest()
    return h[0], h[1]


_DIFFUSION_ROUNDS = 2  # each round = one forward + one backward chaining pass
#  NOTE: chaining uses addition mod 256, not XOR. An XOR chain was
#  tried first but a constant per-pixel difference introduced by the
#  forward pass turned out to cancel out almost immediately in the
#  backward pass (X^d) ^ (Y^d) = X^Y -- an artifact of XOR being its
#  own inverse. Addition mod 256 does not have this cancellation
#  property (a constant offset compounds instead of vanishing), which
#  is confirmed by the differential-attack results in Section "Results".


def _one_fb_pass(out: np.ndarray, iv1: int, iv2: int) -> None:
    n = out.shape[0]
    prev = int(iv1)
    for i in range(n):
        out[i] = (int(out[i]) + prev) & 0xFF
        prev = int(out[i])
    prev = int(iv2)
    for i in range(n - 1, -1, -1):
        out[i] = (int(out[i]) + prev) & 0xFF
        prev = int(out[i])


def _one_fb_unpass(out: np.ndarray, iv1: int, iv2: int) -> None:
    n = out.shape[0]
    prev = int(iv2)
    for i in range(n - 1, -1, -1):
        cur = int(out[i])
        out[i] = (cur - prev) & 0xFF
        prev = cur
    prev = int(iv1)
    for i in range(n):
        cur = int(out[i])
        out[i] = (cur - prev) & 0xFF
        prev = cur


def _diffuse_forward_backward(flat: np.ndarray, seed: bytes) -> np.ndarray:
    out = flat.copy()
    for r in range(_DIFFUSION_ROUNDS):
        iv1, iv2 = _iv_pair(seed + bytes([r]))
        _one_fb_pass(out, iv1, iv2)
    return out


def _undiffuse_forward_backward(flat: np.ndarray, seed: bytes) -> np.ndarray:
    out = flat.copy()
    for r in range(_DIFFUSION_ROUNDS - 1, -1, -1):
        iv1, iv2 = _iv_pair(seed + bytes([r]))
        _one_fb_unpass(out, iv1, iv2)
    return out


def encrypt_image_spd(img: np.ndarray, key: CipherKey) -> np.ndarray:
    """Full scheme: keyed permutation -> Sudoku/GF substitution ->
    bidirectional diffusion. This is the final, recommended
    configuration reported as the primary result in the revision."""
    sub = encrypt_image_sp(img, key)
    if sub.ndim == 2:
        H, W = sub.shape
        flat = _diffuse_forward_backward(sub.flatten(), key.sudoku_seed)
        return flat.reshape(H, W)
    else:
        H, W, C = sub.shape
        chans = [_diffuse_forward_backward(sub[:, :, c].flatten(),
                                            key.sudoku_seed + bytes([c])).reshape(H, W)
                 for c in range(C)]
        return np.stack(chans, axis=-1)


def decrypt_image_spd(cimg: np.ndarray, key: CipherKey) -> np.ndarray:
    if cimg.ndim == 2:
        H, W = cimg.shape
        flat = _undiffuse_forward_backward(cimg.flatten(), key.sudoku_seed)
        sub = flat.reshape(H, W)
    else:
        H, W, C = cimg.shape
        chans = [_undiffuse_forward_backward(cimg[:, :, c].flatten(),
                                              key.sudoku_seed + bytes([c])).reshape(H, W)
                 for c in range(C)]
        sub = np.stack(chans, axis=-1)
    return decrypt_image_sp(sub, key)


CONFIGS = {
    "A": ("A",), "B": ("B",), "C": ("C",),
    "D": ("A", "B"), "E": ("A", "C"), "F": ("B", "C"),
    "G": ("A", "B", "C"),
}
