from neural_cryptanalysis.ciphers.simon import SimonCipher

c = SimonCipher()
key       = 0x1918111009080100
plaintext = 0x65656877
expected  = 0xc69be9bb   # from the official SIMON paper

# Full 32-round result should match test vector
result = c.encrypt(plaintext, key, rounds=32)
print(f"Full rounds: {hex(result)} == {hex(expected)} → {result == expected}")

# # Reduced rounds should produce DIFFERENT output (sanity check)
# r5 = c.encrypt(plaintext, key, rounds=5)
# r6 = c.encrypt(plaintext, key, rounds=6)
# print(f"5-round:  {hex(r5)}")
# print(f"6-round:  {hex(r6)}")
# print(f"Different outputs per round: {r5 != r6}")  # should be True


from gift64  import Gift64Cipher
from skinny64 import Skinny64Cipher
from present  import PresentCipher



g = Gift64Cipher()
key = 0x00000000000000000000000000000000
pt  = 0x0000000000000000

# # Basic sanity checks — no hardcoded expected value
# c_r5  = g.encrypt(pt, key, rounds=5)
# c_r6  = g.encrypt(pt, key, rounds=6)
c_r28 = g.encrypt(pt, key, rounds=28)

# print(f"5-round:   {hex(c_r5)}")
# print(f"6-round:   {hex(c_r6)}")
# print(f"28-round:  {hex(c_r28)}")
# print(f"Different per round:  {c_r5 != c_r6 != c_r28}") # must be True
# print(f"Non-zero output:      {c_r28 != 0}")              # must be True

# # Avalanche check — 1 bit flip in plaintext changes ~50% output bits
c2 = g.encrypt(pt ^ 1, key, rounds=28)
diff = bin(c_r28 ^ c2).count('1')
print(f"Bit flips for 1-bit input diff: {diff}/64")       # should be ~32

# SKINNY-64-64 test vector from the paper
sk = Skinny64Cipher()
r = sk.encrypt(0x06034F957724D19D, 0xF5269826FC681238, 32)
print(f"SKINNY-64: {hex(r)} == 0xbb39dfb2429b8ac7 → {r == 0xBB39DFB2429B8AC7}")

# PRESENT-80: PT=0, K=0 → 0x5579C1387B228445
p = PresentCipher()
r = p.encrypt(0x0000000000000000, 0x00000000000000000000, 31)
print(f"PRESENT:   {hex(r)} == 0x5579c1387b228445 → {r == 0x5579C1387B228445}")