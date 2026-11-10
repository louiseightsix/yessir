"""Namespace constants, byte token allocations, and architectural dimensions.
Strictly preserves the 1-byte = 1-token invariant.

Byte assignments mirror spec/README.md Section 2, which is the single source of truth;
tests/test_mechanics.py asserts that the two agree.
"""

# Universal Byte Vocabulary
VOCAB_SIZE = 256  # 0x00 through 0xFF

# Canonical Representation Dimensions
D_MODEL = 512
D_SYMBOL = 256
D_POS = 256
D_ROLE = 64
D_SIG = 192
D_FFN = 1376
N_HEADS = 8
N_LAYERS = 6

# Numeral Symbol Pool Sigma (240 candidate symbols)
SIGMA_POOL = list(range(0x00, 0xF0))

# Canonical evaluation axiom: conventional ASCII digits '0'-'9', 'A', 'B' (truncated to the base)
CANONICAL_AXIOM = list(b"0123456789AB")

# Control & Delimiter Tokens V_ctrl (0xF0 - 0xFF)
DELIM_AXIOM = 0xF0     # Tier 2/5 axiom block delimiter
DELIM_A = 0xF1         # Tier 2/5 operand A delimiter
DELIM_B = 0xF2         # Tier 2/5 operand B delimiter
DELIM_CARRY = 0xF3     # Tier 2/5 carry delimiter (SM only)
DELIM_OUT = 0xF4       # Tier 2/5 output delimiter (SM only)
START = 0xF5           # AR trace trigger
TOKEN_S = 0xF6         # Sum query: SM prompt tail / SM-SeqRev Pass 2 trigger / AR sum marker
TOKEN_C = 0xF7         # Carry query: SM-Seq Pass 2 trigger / SM-SeqRev, SM-Fork prompt tail / AR carry marker
FLAG_CONTINUE = 0xF8   # Loop continuation flag (SM only)
FLAG_FINISHED = 0xF9   # Loop termination flag (SM) / trace terminator (AR)
DELIM_ORIGIN = 0xFA    # Tier 5 zero-origin spatial anchor / Tier 6 address-value binder
DELIM_GENERIC = 0xFB   # Tier 3/4 neutral block delimiter / Tier 6 entry opener, rendered as '|'
TOKEN_FLAG = 0xFC      # SM-Seq / SM-SeqRev Pass 3 flag trigger / SM-Fork prompt tail
PAD = 0xFD             # Batch padding (right-padding only)
BOS = 0xFE             # Sequence origin; attention sink and NoPE count anchor
TOKEN_PROBE = 0xFF     # Successor binding probe trigger: [PROBE] s_i -> s_{i+1} (spec/05 Section 4)
IGNORE_LABEL = -100    # PyTorch loss masking ignore index

# head_flag class indices (SM-MH)
FLAG_CLASS_CONTINUE = 0
FLAG_CLASS_FINISHED = 1

# Positional Role Subspace IDs (Tier 1: r in {0 .. 7})
ROLE_AXIOM = 0
ROLE_OPERAND_A = 1
ROLE_OPERAND_B = 2
ROLE_CARRY = 3         # Carry digits, and the [C] trigger
ROLE_OUT = 4           # Sum digits, and the [S] query
ROLE_FLAG = 5          # The [F] flag trigger of SM-Seq, SM-SeqRev and SM-Fork
ROLE_START = 6         # The [START] trigger of AR
ROLE_CTRL = 7          # [BOS] and [PAD]
NUM_ROLES = 8

# Human-readable glyph mappings for terminal visualisations (spec/README.md display aliases)
TOKEN_NAMES = {
    DELIM_AXIOM: "[AXIOM]",
    DELIM_A: "[A]",
    DELIM_B: "[B]",
    DELIM_CARRY: "[CARRY]",
    DELIM_OUT: "[OUT]",
    START: "[START]",
    TOKEN_S: "[S]",
    TOKEN_C: "[C]",
    FLAG_CONTINUE: "[CONTINUE]",
    FLAG_FINISHED: "[FINISHED]",
    DELIM_ORIGIN: "[ORIGIN]",
    DELIM_GENERIC: "|",
    TOKEN_FLAG: "[F]",
    PAD: "[PAD]",
    BOS: "[BOS]",
    TOKEN_PROBE: "[PROBE]",
}


def token_to_str(token: int) -> str:
    """Format token integer into readable terminal representation."""
    if token in TOKEN_NAMES:
        return TOKEN_NAMES[token]
    # Check ASCII printable
    if 32 <= token <= 126:
        return f"'{chr(token)}'"
    return f"0x{token:02X}"
