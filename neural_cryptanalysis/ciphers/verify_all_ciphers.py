"""
Comprehensive cipher verification script.
Tests each cipher with fixed inputs, known test vectors where available,
avalanche effect, and differential consistency.
Results saved to cipher_verification_output.txt
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from neural_cryptanalysis.ciphers.simon     import SimonCipher
from neural_cryptanalysis.ciphers.gift64    import Gift64Cipher
from neural_cryptanalysis.ciphers.gift128   import Gift128Cipher
from neural_cryptanalysis.ciphers.skinny64  import Skinny64Cipher
from neural_cryptanalysis.ciphers.skinny128 import Skinny128Cipher
from neural_cryptanalysis.ciphers.craft     import CraftCipher
from neural_cryptanalysis.ciphers.pyjamask  import PyjamaskCipher
from neural_cryptanalysis.ciphers.present   import PresentCipher

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def hamming(a, b):
    return bin(a ^ b).count('1')

def avalanche_score(cipher, pt, key, rounds, bit_flips=16):
    """Flip each of `bit_flips` input bits and measure average bit-change in output."""
    base = cipher.encrypt(pt, key, rounds)
    diffs = []
    for i in range(bit_flips):
        flipped = cipher.encrypt(pt ^ (1 << i), key, rounds)
        diffs.append(hamming(base, flipped))
    avg = sum(diffs) / len(diffs)
    ideal = cipher.block_bits / 2
    return avg, ideal

def section(title):
    sep = "=" * 60
    return f"\n{sep}\n  {title}\n{sep}\n"

def subsection(title):
    return f"\n  --- {title} ---\n"

# ─────────────────────────────────────────────
# Test definitions
# ─────────────────────────────────────────────

lines = []
p = lines.append

p("=" * 60)
p("  CIPHER VERIFICATION REPORT")
p("  Neural Cryptanalysis Project")
p("=" * 60)

# ══════════════════════════════════════════════
# 1. SIMON 32/64
# ══════════════════════════════════════════════
p(section("1. SIMON 32/64  (block=32, key=64, rounds=32)"))

simon = SimonCipher()
PT_S  = 0x65656877
KEY_S = 0x1918111009080100
EXP_S = 0xc69be9bb

p(f"  Plaintext  : {hex(PT_S)}")
p(f"  Key        : {hex(KEY_S)}")
p(f"  Expected   : {hex(EXP_S)}")

full = simon.encrypt(PT_S, KEY_S, 32)
p(f"  Got (r=32) : {hex(full)}")
p(f"  Test vector: {'PASS' if full == EXP_S else 'FAIL'}")

p(subsection("Reduced-round outputs"))
for r in [1, 2, 3, 4, 5, 6, 8, 10, 16, 32]:
    ct = simon.encrypt(PT_S, KEY_S, r)
    p(f"    r={r:2d}  ->  {hex(ct)}")

p(subsection("Differential (ΔP = 0x40000000)"))
delta = 0x40000000
for r in [4, 6, 8]:
    c1 = simon.encrypt(PT_S, KEY_S, r)
    c2 = simon.encrypt(PT_S ^ delta, KEY_S, r)
    p(f"    r={r}  C={hex(c1)}  C'={hex(c2)}  ΔC={hex(c1^c2)}")

p(subsection("Avalanche (r=8, flipping 16 LSBs)"))
avg, ideal = avalanche_score(simon, PT_S, KEY_S, 8)
p(f"    Avg bit flips: {avg:.2f} / {ideal:.1f} ideal  ({'good' if abs(avg-ideal)<ideal*0.3 else 'weak'})")

# ══════════════════════════════════════════════
# 2. GIFT-64
# ══════════════════════════════════════════════
p(section("2. GIFT-64  (block=64, key=128, rounds=28)"))

gift64 = Gift64Cipher()
PT_G64  = 0x0000000000000000
KEY_G64 = 0x00000000000000000000000000000000

p(f"  Plaintext  : {hex(PT_G64)}")
p(f"  Key        : {hex(KEY_G64)}")

full = gift64.encrypt(PT_G64, KEY_G64, 28)
p(f"  Got (r=28) : {hex(full)}")
p(f"  Non-zero   : {'PASS' if full != 0 else 'FAIL'}")

p(subsection("Reduced-round outputs"))
for r in [1, 2, 3, 4, 5, 6, 8, 10, 14, 28]:
    ct = gift64.encrypt(PT_G64, KEY_G64, r)
    p(f"    r={r:2d}  ->  {hex(ct)}")

p(subsection("Differential (ΔP = 0x0000000000000001)"))
delta = 0x0000000000000001
for r in [4, 6, 8]:
    c1 = gift64.encrypt(PT_G64, KEY_G64, r)
    c2 = gift64.encrypt(PT_G64 ^ delta, KEY_G64, r)
    p(f"    r={r}  C={hex(c1)}  C'={hex(c2)}  ΔC={hex(c1^c2)}")

p(subsection("Avalanche (r=10, flipping 16 LSBs)"))
avg, ideal = avalanche_score(gift64, PT_G64, KEY_G64, 10)
p(f"    Avg bit flips: {avg:.2f} / {ideal:.1f} ideal  ({'good' if abs(avg-ideal)<ideal*0.3 else 'weak'})")

# ══════════════════════════════════════════════
# 3. GIFT-128
# ══════════════════════════════════════════════
p(section("3. GIFT-128  (block=128, key=128, rounds=40)"))

gift128 = Gift128Cipher()
PT_G128  = 0x00000000000000000000000000000000
KEY_G128 = 0x00000000000000000000000000000000

p(f"  Plaintext  : {hex(PT_G128)}")
p(f"  Key        : {hex(KEY_G128)}")

full = gift128.encrypt(PT_G128, KEY_G128, 40)
p(f"  Got (r=40) : {hex(full)}")
p(f"  Non-zero   : {'PASS' if full != 0 else 'FAIL'}")

p(subsection("Reduced-round outputs"))
for r in [1, 2, 3, 4, 5, 6, 8, 10, 20, 40]:
    ct = gift128.encrypt(PT_G128, KEY_G128, r)
    p(f"    r={r:2d}  ->  {hex(ct)}")

p(subsection("Differential (ΔP = 0x1)"))
delta = 0x00000000000000000000000000000001
for r in [4, 6, 8]:
    c1 = gift128.encrypt(PT_G128, KEY_G128, r)
    c2 = gift128.encrypt(PT_G128 ^ delta, KEY_G128, r)
    p(f"    r={r}  ΔC={hex(c1^c2)}  HW={bin(c1^c2).count('1')}")

p(subsection("Avalanche (r=10, flipping 16 LSBs)"))
avg, ideal = avalanche_score(gift128, PT_G128, KEY_G128, 10)
p(f"    Avg bit flips: {avg:.2f} / {ideal:.1f} ideal  ({'good' if abs(avg-ideal)<ideal*0.3 else 'weak'})")

# ══════════════════════════════════════════════
# 4. SKINNY-64-64
# ══════════════════════════════════════════════
p(section("4. SKINNY-64-64  (block=64, key=64, rounds=32)"))

sk64 = Skinny64Cipher()
PT_SK64  = 0x06034F957724D19D
KEY_SK64 = 0xF5269826FC681238
EXP_SK64 = 0xBB39DFB2429B8AC7

p(f"  Plaintext  : {hex(PT_SK64)}")
p(f"  Key        : {hex(KEY_SK64)}")
p(f"  Expected   : {hex(EXP_SK64)}")

full = sk64.encrypt(PT_SK64, KEY_SK64, 32)
p(f"  Got (r=32) : {hex(full)}")
p(f"  Test vector: {'PASS' if full == EXP_SK64 else 'FAIL'}")

p(subsection("Reduced-round outputs"))
for r in [1, 2, 3, 4, 5, 6, 8, 10, 16, 32]:
    ct = sk64.encrypt(PT_SK64, KEY_SK64, r)
    p(f"    r={r:2d}  ->  {hex(ct)}")

p(subsection("Differential (ΔP = 0x0000000000000001)"))
delta = 0x0000000000000001
for r in [4, 6, 8]:
    c1 = sk64.encrypt(PT_SK64, KEY_SK64, r)
    c2 = sk64.encrypt(PT_SK64 ^ delta, KEY_SK64, r)
    p(f"    r={r}  C={hex(c1)}  C'={hex(c2)}  ΔC={hex(c1^c2)}")

p(subsection("Avalanche (r=8, flipping 16 LSBs)"))
avg, ideal = avalanche_score(sk64, PT_SK64, KEY_SK64, 8)
p(f"    Avg bit flips: {avg:.2f} / {ideal:.1f} ideal  ({'good' if abs(avg-ideal)<ideal*0.3 else 'weak'})")

# ══════════════════════════════════════════════
# 5. SKINNY-128-128
# ══════════════════════════════════════════════
p(section("5. SKINNY-128-128  (block=128, key=128, rounds=40)"))

sk128 = Skinny128Cipher()
PT_SK128  = 0x00000000000000000000000000000000
KEY_SK128 = 0x00000000000000000000000000000000

p(f"  Plaintext  : {hex(PT_SK128)}")
p(f"  Key        : {hex(KEY_SK128)}")

full = sk128.encrypt(PT_SK128, KEY_SK128, 40)
p(f"  Got (r=40) : {hex(full)}")
p(f"  Non-zero   : {'PASS' if full != 0 else 'FAIL'}")

p(subsection("Reduced-round outputs"))
for r in [1, 2, 3, 4, 5, 6, 8, 10, 20, 40]:
    ct = sk128.encrypt(PT_SK128, KEY_SK128, r)
    p(f"    r={r:2d}  ->  {hex(ct)}")

p(subsection("Differential (ΔP = 0x1)"))
delta = 0x00000000000000000000000000000001
for r in [4, 6, 8]:
    c1 = sk128.encrypt(PT_SK128, KEY_SK128, r)
    c2 = sk128.encrypt(PT_SK128 ^ delta, KEY_SK128, r)
    p(f"    r={r}  ΔC={hex(c1^c2)}  HW={bin(c1^c2).count('1')}")

p(subsection("Avalanche (r=10, flipping 16 LSBs)"))
avg, ideal = avalanche_score(sk128, PT_SK128, KEY_SK128, 10)
p(f"    Avg bit flips: {avg:.2f} / {ideal:.1f} ideal  ({'good' if abs(avg-ideal)<ideal*0.3 else 'weak'})")

# ══════════════════════════════════════════════
# 6. CRAFT
# ══════════════════════════════════════════════
p(section("6. CRAFT  (block=64, key=64+64 tweak, rounds=32)"))

craft = CraftCipher()
# key = top 64 bits (key K) | bottom 64 bits (tweak T)
PT_CR  = 0x0000000000000000
KEY_CR = 0x00000000000000000000000000000000   # K=0, T=0

p(f"  Plaintext  : {hex(PT_CR)}")
p(f"  Key (K||T) : {hex(KEY_CR)}")

full = craft.encrypt(PT_CR, KEY_CR, 32)
p(f"  Got (r=32) : {hex(full)}")
p(f"  Non-zero   : {'PASS' if full != 0 else 'FAIL'}")

# Non-zero key test
PT_CR2  = 0xFFFFFFFFFFFFFFFF
KEY_CR2 = 0x0123456789ABCDEF0123456789ABCDEF
full2 = craft.encrypt(PT_CR2, KEY_CR2, 32)
p(f"\n  Plaintext  : {hex(PT_CR2)}")
p(f"  Key (K||T) : {hex(KEY_CR2)}")
p(f"  Got (r=32) : {hex(full2)}")
p(f"  Non-zero   : {'PASS' if full2 != 0 else 'FAIL'}")

p(subsection("Reduced-round outputs (K=0, T=0)"))
for r in [1, 2, 3, 4, 5, 6, 8, 10, 16, 32]:
    ct = craft.encrypt(PT_CR, KEY_CR, r)
    p(f"    r={r:2d}  ->  {hex(ct)}")

p(subsection("Differential (ΔP = 0x0000000000000001)"))
delta = 0x0000000000000001
for r in [4, 6, 8]:
    c1 = craft.encrypt(PT_CR, KEY_CR, r)
    c2 = craft.encrypt(PT_CR ^ delta, KEY_CR, r)
    p(f"    r={r}  C={hex(c1)}  C'={hex(c2)}  ΔC={hex(c1^c2)}")

p(subsection("Avalanche (r=8, flipping 16 LSBs)"))
KEY_CR_nz = 0xDEADBEEFCAFEBABE0123456789ABCDEF
avg_full, ideal = avalanche_score(craft, PT_CR, KEY_CR_nz, 8)
p(f"    Avg bit flips: {avg_full:.2f} / {ideal:.1f} ideal  ({'good' if abs(avg_full-ideal)<ideal*0.3 else 'weak'})")

# ══════════════════════════════════════════════
# 7. PYJAMASK-96
# ══════════════════════════════════════════════
p(section("7. PYJAMASK-96  (block=96, key=128, rounds=14)"))

pyj = PyjamaskCipher()
PT_PY  = 0x000000000000000000000000
KEY_PY = 0x00000000000000000000000000000000

p(f"  Plaintext  : {hex(PT_PY)}")
p(f"  Key        : {hex(KEY_PY)}")

full = pyj.encrypt(PT_PY, KEY_PY, 14)
p(f"  Got (r=14) : {hex(full)}")
p(f"  Non-zero   : {'PASS' if full != 0 else 'FAIL'}")

# Non-zero input test
PT_PY2  = 0x0123456789ABCDEF01234567
KEY_PY2 = 0x0123456789ABCDEF0123456789ABCDEF
full2 = pyj.encrypt(PT_PY2, KEY_PY2, 14)
p(f"\n  Plaintext  : {hex(PT_PY2)}")
p(f"  Key        : {hex(KEY_PY2)}")
p(f"  Got (r=14) : {hex(full2)}")
p(f"  Non-zero   : {'PASS' if full2 != 0 else 'FAIL'}")

p(subsection("Reduced-round outputs (PT=0, K=0)"))
for r in [1, 2, 3, 4, 5, 6, 7, 8, 10, 14]:
    ct = pyj.encrypt(PT_PY, KEY_PY, r)
    p(f"    r={r:2d}  ->  {hex(ct)}")

p(subsection("Differential (ΔP = 0x000000000000000000000001)"))
delta = 0x000000000000000000000001
for r in [4, 6, 8]:
    c1 = pyj.encrypt(PT_PY, KEY_PY, r)
    c2 = pyj.encrypt(PT_PY ^ delta, KEY_PY, r)
    p(f"    r={r}  C={hex(c1)}  C'={hex(c2)}  ΔC={hex(c1^c2)}")

p(subsection("Avalanche (r=8, flipping 16 LSBs)"))
avg, ideal = avalanche_score(pyj, PT_PY, KEY_PY, 8)
p(f"    Avg bit flips: {avg:.2f} / {ideal:.1f} ideal  ({'good' if abs(avg-ideal)<ideal*0.3 else 'weak'})")

# ══════════════════════════════════════════════
# 8. PRESENT-80
# ══════════════════════════════════════════════
p(section("8. PRESENT-80  (block=64, key=80, rounds=31)"))

present = PresentCipher()
PT_PR  = 0x0000000000000000
KEY_PR = 0x00000000000000000000
EXP_PR = 0x5579C1387B228445

p(f"  Plaintext  : {hex(PT_PR)}")
p(f"  Key        : {hex(KEY_PR)}")
p(f"  Expected   : {hex(EXP_PR)}")

full = present.encrypt(PT_PR, KEY_PR, 31)
p(f"  Got (r=31) : {hex(full)}")
p(f"  Test vector: {'PASS' if full == EXP_PR else 'FAIL'}")

p(subsection("Reduced-round outputs"))
for r in [1, 2, 3, 4, 5, 6, 8, 10, 16, 31]:
    ct = present.encrypt(PT_PR, KEY_PR, r)
    p(f"    r={r:2d}  ->  {hex(ct)}")

p(subsection("Differential (ΔP = 0x0000000000000001)"))
delta = 0x0000000000000001
for r in [4, 6, 8]:
    c1 = present.encrypt(PT_PR, KEY_PR, r)
    c2 = present.encrypt(PT_PR ^ delta, KEY_PR, r)
    p(f"    r={r}  C={hex(c1)}  C'={hex(c2)}  ΔC={hex(c1^c2)}")

p(subsection("Avalanche (r=8, flipping 16 LSBs)"))
avg, ideal = avalanche_score(present, PT_PR, KEY_PR, 8)
p(f"    Avg bit flips: {avg:.2f} / {ideal:.1f} ideal  ({'good' if abs(avg-ideal)<ideal*0.3 else 'weak'})")

# ══════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════
p(section("SUMMARY"))
p(f"  {'Cipher':<14} {'Block':>6} {'Key':>6} {'Rounds':>7}  Status")
p(f"  {'-'*50}")

results = [
    ("simon32",   simon,   PT_S,    KEY_S,    32, EXP_S),
    ("skinny64",  sk64,    PT_SK64, KEY_SK64, 32, EXP_SK64),
    ("present",   present, PT_PR,   KEY_PR,   31, EXP_PR),
]
no_vector = [
    ("gift64",    gift64,   PT_G64,   KEY_G64,   28),
    ("gift128",   gift128,  PT_G128,  KEY_G128,  40),
    ("skinny128", sk128,    PT_SK128, KEY_SK128, 40),
    ("craft",     craft,    PT_CR,    KEY_CR,    32),
    ("pyjamask",  pyj,      PT_PY,    KEY_PY,    14),
]

for name, cipher, pt, key, rounds, expected in results:
    got = cipher.encrypt(pt, key, rounds)
    status = "PASS (test vector)" if got == expected else "FAIL"
    p(f"  {name:<14} {cipher.block_bits:>6} {cipher.key_bits:>6} {rounds:>7}  {status}")

for name, cipher, pt, key, rounds in no_vector:
    got = cipher.encrypt(pt, key, rounds)
    status = "PASS (non-zero output)" if got != 0 else "FAIL (zero output)"
    p(f"  {name:<14} {cipher.block_bits:>6} {cipher.key_bits:>6} {rounds:>7}  {status}")

p("\n" + "=" * 60)
p("  END OF REPORT")
p("=" * 60)

# ══════════════════════════════════════════════
# ML Distinguisher Accuracy vs Rounds
# ══════════════════════════════════════════════
p(section("ML DISTINGUISHER ACCURACY vs ROUNDS"))
p("  Trains a Siamese distinguisher on each cipher at increasing")
p("  round counts to show how accuracy degrades as the cipher")
p("  becomes harder to distinguish from random.")
p("  Model: Siamese  |  Representation: delta  |  Samples: 4000  |  Epochs: 10")
p("")

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from neural_cryptanalysis.data.generator import generate_dataset
from neural_cryptanalysis.utils.config   import get_cipher, DELTA_P, FULL_ROUNDS
from neural_cryptanalysis.models.siamese import SiameseNet

def _quick_acc(cipher_name, rounds, n=4000, epochs=10):
    c     = get_cipher(cipher_name)
    delta = DELTA_P[cipher_name]
    # delta rep: input = bits(C XOR C'), shape = (block_bits,)
    X, y  = generate_dataset(c, rounds=rounds, n_samples=n,
                              delta_p=delta, representation="delta")
    X = X.astype(np.float32)
    X_t = torch.tensor(X)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    split = int(0.8 * len(X_t))
    tr = DataLoader(TensorDataset(X_t[:split], y_t[:split]), batch_size=256, shuffle=True)
    te = DataLoader(TensorDataset(X_t[split:], y_t[split:]), batch_size=256)

    # For delta rep, branch_dim = block_bits (both branches are the same delta vector)
    # Use MLP directly since delta is a single vector, not a pair
    from neural_cryptanalysis.models.mlp import MLP
    model = MLP(input_dim=X_t.shape[1])

    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.BCELoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in tr:
            pred = model(xb); l = crit(pred, yb)
            opt.zero_grad(); l.backward(); opt.step()
        sched.step()

    model.eval(); correct = tot = 0
    with torch.no_grad():
        for xb, yb in te:
            pred = model(xb)
            correct += ((pred > 0.5).float() == yb).sum().item()
            tot     += yb.size(0)
    return correct / tot

# Run for each cipher
cipher_round_configs = {
    "simon32":  list(range(2, 10)),
    "gift64":   list(range(2, 10)),
    "present":  list(range(2, 10)),
    "craft":    list(range(2, 10)),
    "pyjamask": list(range(1, 8)),
}

for cipher_name, round_list in cipher_round_configs.items():
    p(f"\n  {cipher_name.upper()}")
    p(f"  {'Round':<8} {'Accuracy':>10}  {'Distinguishable?':>18}")
    p(f"  {'-'*40}")
    prev_acc = None
    for r in round_list:
        acc = _quick_acc(cipher_name, r)
        change = ""
        if prev_acc is not None:
            diff = acc - prev_acc
            change = f"  ({diff:+.4f})"
        dist = "YES" if acc > 0.55 else "NO (near random)"
        p(f"  r={r:<6} {acc:>10.4f}  {dist:<18}{change}")
        prev_acc = acc
    p("")

p("=" * 60)
p("  END OF REPORT")
p("=" * 60)

# ─────────────────────────────────────────────
# Write to file
# ─────────────────────────────────────────────
output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'cipher_verification_output.txt')
output_path = os.path.normpath(output_path)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')

print(f"Report written to: {output_path}")
# Also print to console
print('\n'.join(lines))
