"""yessir: shared code for rule-following experiments.

Only code that is clearly experiment-agnostic lives here. Everything else stays inside its
experiment under experiments/ until a second experiment actually needs it.

    blocks    Transformer building blocks (RMSNorm, SwiGLU/GELU FFN, SDPA attention, Pre-LN block)
    training  Seeding, AdamW parameter groups, warm-up + cosine LR schedule
    logging   JSONL metrics logger
"""
