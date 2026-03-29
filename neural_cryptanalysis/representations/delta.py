from .utils import int_to_bits

def transform(c, c_, bits):
    """
    ΔC representation: bits(C ⊕ C′)
    """
    return int_to_bits(c ^ c_, bits)