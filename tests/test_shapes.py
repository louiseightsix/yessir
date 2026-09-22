"""Tests for PyTorch tensor shapes, embedding layouts, heads, masks, batching, and training."""

import math

import torch

from transformadder.core.blocks import RMSNorm, prefix_lm_mask
from transformadder.core.constants import D_MODEL, D_ROLE, D_SYMBOL, IGNORE_LABEL, PAD, ROLE_CTRL, TOKEN_S
from transformadder.core.data import ProceduralArithmeticGenerator
from transformadder.core.embedding import SubspaceEmbedding
from transformadder.core.heads import MultiHeadOutput, TiedIdentityHead
from transformadder.core.model import TransformAdder
from transformadder.core.sequences import Seq, collate
from transformadder.core.training import build_optimizer, build_scheduler
from transformadder.utils.evaluation import evaluate
from transformadder.utils.runner import Variant, accumulate_gradients, compute_loss


def small_model(tier, head_type):
    torch.manual_seed(0)
    return TransformAdder(tier=tier, n_layers=2, head_type=head_type)


def test_embedding_tier1_unit_norm_components():
    embed = SubspaceEmbedding(tier=1)
    token_ids = torch.tensor([[0, 10, 48, 239, 255]])
    role_ids = torch.tensor([[0, 1, 2, 3, 4]])
    sig_ids = torch.tensor([[0, 1, 2, 19, 40]])
    x = embed(token_ids, role_ids=role_ids, sig_ids=sig_ids)
    assert x.shape == (1, 5, 512)

    one_hot, role, sig = x[..., :D_SYMBOL], x[..., D_SYMBOL:D_SYMBOL + D_ROLE], x[..., D_SYMBOL + D_ROLE:]
    assert torch.equal(one_hot.argmax(-1), token_ids) and torch.all(one_hot.sum(-1) == 1.0)
    assert torch.allclose(role.norm(dim=-1), torch.ones(1, 5), atol=1e-5)
    assert torch.allclose(sig.norm(dim=-1), torch.ones(1, 5), atol=1e-5)
    # Interleaved layout: p = 0 gives (sin, cos) = (0, 1) pairs, scaled by 1/sqrt(96)
    assert torch.allclose(sig[0, 0, 0::2], torch.zeros(96), atol=1e-6)
    assert torch.allclose(sig[0, 0, 1::2], torch.full((96,), 1 / math.sqrt(96)), atol=1e-6)


def test_embedding_nope_tiers_shapes_and_zeros():
    for tier in [2, 3, 5, 6, 7]:
        x = SubspaceEmbedding(tier=tier)(torch.randint(0, 256, (3, 7)))
        assert x.shape == (3, 7, 512)
        assert torch.all(x[:, :, D_SYMBOL:] == 0.0), f"Tier {tier} must have zero positional vectors"


def test_embedding_tier4_role_subspace_and_zero_sig():
    embed = SubspaceEmbedding(tier=4)
    token_ids = torch.tensor([[0, 10, 48, 239, 255]])
    role_ids = torch.tensor([[0, 1, 2, 3, 4]])
    x = embed(token_ids, role_ids=role_ids)
    assert x.shape == (1, 5, 512)

    one_hot, role, sig = x[..., :D_SYMBOL], x[..., D_SYMBOL:D_SYMBOL + D_ROLE], x[..., D_SYMBOL + D_ROLE:]
    assert torch.equal(one_hot.argmax(-1), token_ids) and torch.all(one_hot.sum(-1) == 1.0)
    assert torch.allclose(role.norm(dim=-1), torch.ones(1, 5), atol=1e-5)
    assert torch.all(sig == 0.0)


def test_tied_identity_head_has_no_parameters():
    head = TiedIdentityHead(d_model=D_MODEL)
    assert list(head.parameters()) == []
    assert head(torch.randn(2, 4, 512)).shape == (2, 4, 256)
    assert list(RMSNorm(8, gain=False).parameters()) == []


def test_multi_head_output_reads_true_tail_not_padding():
    head = MultiHeadOutput(d_model=D_MODEL)
    h = torch.randn(2, 10, 512)
    lengths = torch.tensor([10, 4])
    logits_sum, logits_carry, logits_flag = head(h, lengths)
    assert logits_sum.shape == (2, 256) and logits_carry.shape == (2, 256) and logits_flag.shape == (2, 2)
    # Row 1 must be read at index 3, and must not depend on the padded tail
    alone = head(h[1:2, :4], torch.tensor([4]))[0]
    assert torch.allclose(logits_sum[1], alone[0], atol=1e-6)


def test_collate_right_pads():
    batch = collate([Seq([1, 2, 3], [0, 1, 2], [0, 1, 2]), Seq([4], [4], [7])])
    assert batch["token_ids"].tolist() == [[1, 2, 3], [4, PAD, PAD]]
    assert batch["role_ids"].tolist() == [[0, 1, 2], [4, ROLE_CTRL, ROLE_CTRL]]
    assert batch["sig_ids"].tolist() == [[0, 1, 2], [7, 0, 0]]
    assert batch["lengths"].tolist() == [3, 1]
    nope = collate([Seq([1, 2]), Seq([3])])
    assert nope["role_ids"] is None and nope["sig_ids"] is None

    t4 = collate([Seq([1, 2], [0, 1], None), Seq([4], [4], None)])
    assert t4["token_ids"].tolist() == [[1, 2], [4, PAD]]
    assert t4["role_ids"].tolist() == [[0, 1], [4, ROLE_CTRL]]
    assert t4["sig_ids"] is None


def test_padding_invariance_under_causal_mask():
    """Right-padding must not change any real position's output (no padding mask needed)."""
    for tier, head_type in [(1, "multi_head"), (2, "single_head"), (4, "multi_head"), (5, "single_head")]:
        model = small_model(tier, head_type).eval()
        gen = ProceduralArithmeticGenerator(bases=[2, 10], seed=3)
        batch = gen.sm_mh_batch(6, tier) if head_type == "multi_head" else gen.sm_seq_batch(6, tier)
        args = (batch["token_ids"], batch["role_ids"], batch["sig_ids"])
        with torch.no_grad():
            for i in range(6):
                n = int(batch["lengths"][i])
                solo = tuple(t[i:i + 1, :n] if t is not None else None for t in args)
                if head_type == "multi_head":
                    full = model.forward_sm_mh(*args, batch["lengths"])
                    single = model.forward_sm_mh(*solo, torch.tensor([n]))
                    assert all(torch.allclose(f[i], s[0], atol=1e-4) for f, s in zip(full, single))
                else:
                    assert torch.allclose(model(*args)[i, :n], model(*solo)[0], atol=1e-4)


def test_prefix_lm_mask():
    mask = prefix_lm_mask(torch.tensor([3, 1]), T=5)
    assert mask.shape == (2, 1, 5, 5)
    m = mask[0, 0]
    assert m[0, 2] and m[2, 0], "prefix is bidirectional"
    assert not m[0, 3] and not m[3, 4], "non-prefix tokens stay causal"
    assert m[4, 0] and m[4, 4]
    assert torch.equal(mask[1, 0], torch.tril(torch.ones(5, 5, dtype=torch.bool)))

    model = small_model(1, "single_head").eval()
    batch = ProceduralArithmeticGenerator(bases=[2, 10], seed=1).ar_batch(2, tier=1)
    T = batch["token_ids"].size(1)
    out = model(batch["token_ids"], batch["role_ids"], batch["sig_ids"],
                attn_mask=prefix_lm_mask(torch.tensor([5, 5]), T))
    assert out.shape == (2, T, 256) and torch.isfinite(out).all()


def test_batch_generators_all_paradigms_and_tiers():
    gen = ProceduralArithmeticGenerator(bases=[2, 4, 8, 10], seed=42)
    for tier in (1, 2, 3):
        mh = gen.sm_mh_batch(8, tier)
        assert mh["target_sum"].shape == mh["target_carry"].shape == mh["target_flag"].shape == (8,)
        rows = torch.arange(8)
        assert torch.all(mh["token_ids"][rows, mh["lengths"] - 1] == TOKEN_S), "SM prompts end on [S]"
        assert (mh["role_ids"] is not None) == (tier == 1)

        seq = gen.sm_seq_batch(8, tier)
        assert torch.all((seq["labels"] != IGNORE_LABEL).sum(dim=1) == 3), "exactly three loss positions"

        seq_rev = gen.sm_seq_rev_batch(8, tier)
        assert torch.all((seq_rev["labels"] != IGNORE_LABEL).sum(dim=1) == 3), "exactly three loss positions"

        fork = gen.sm_fork_batch(8, tier)
        assert fork["token_ids"].shape[0] == 24
        assert torch.all((fork["labels"] != IGNORE_LABEL).sum(dim=1) == 1), "exactly one loss position per fork branch"

        ar = gen.ar_batch(8, tier)
        assert ar["labels"].shape == ar["token_ids"].shape
        assert torch.all(ar["labels"][ar["token_ids"] == PAD] == IGNORE_LABEL)


def test_terminal_step_upweighting():
    gen = ProceduralArithmeticGenerator(bases=[10], min_len=10, max_len=20, p_terminal=0.5, seed=0)
    flags = gen.sm_mh_batch(400, tier=2)["target_flag"].float().mean().item()
    assert 0.4 < flags < 0.65


def test_gradient_accumulation_matches_single_batch():
    """accum_steps micro-batches must produce the gradient of one batch of the same samples."""
    for paradigm, tier in [("sm_mh", 2), ("sm_seq", 6), ("sm_fork", 7)]:
        variant = Variant(name="t", paradigm=paradigm, tier=tier, log_dir="", n_layers=2)
        model = small_model(tier, "multi_head" if paradigm == "sm_mh" else "single_head")
        grads = []
        for micro, accum in ((8, 1), (2, 4)):        # same 8 transitions, drawn in the same order
            gen = ProceduralArithmeticGenerator(bases=[2, 10], seed=11, tier=tier)
            make_batch = getattr(gen, f"{paradigm}_batch")
            model.zero_grad(set_to_none=True)
            loss, _ = accumulate_gradients(model, variant, make_batch, micro, accum, torch.device("cpu"), False)
            grads.append((loss, [p.grad.clone() for p in model.parameters() if p.grad is not None]))
        (loss1, g1), (loss4, g4) = grads
        assert abs(loss1 - loss4) < 1e-4, (paradigm, tier)
        assert len(g1) == len(g4) > 0
        for a, b in zip(g1, g4):
            assert torch.allclose(a, b, atol=1e-5, rtol=1e-4), (paradigm, tier)


def test_training_step_reduces_loss_each_paradigm():
    for paradigm, tier in [
        ("sm_mh", 1),
        ("sm_seq", 2),
        ("sm_seq_rev", 2),
        ("ar", 3),
        ("ar", 1),
        ("sm_mh", 4),
        ("sm_seq", 4),
        ("sm_seq_rev", 4),
        ("sm_mh", 5),
        ("sm_seq", 5),
        ("sm_seq_rev", 5),
        ("sm_fork", 2),
        ("sm_fork", 4),
        ("sm_fork", 5),
        ("ar", 4),
        ("ar", 5),
        ("sm_seq", 6),
        ("ar", 6),
    ]:
        variant = Variant(name="t", paradigm=paradigm, tier=tier, log_dir="", n_layers=2)
        model = small_model(tier, "multi_head" if paradigm == "sm_mh" else "single_head")
        optimizer = build_optimizer(model, lr=1e-3)
        batch = getattr(ProceduralArithmeticGenerator(bases=[2, 10], max_len=4, seed=0), f"{paradigm}_batch")(8, tier)
        losses = []
        for _ in range(8):
            loss, _ = compute_loss(model, variant, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], f"{paradigm}/T{tier}: {losses}"


def test_optimizer_groups_and_schedule():
    model = small_model(1, "multi_head")
    optimizer = build_optimizer(model)
    decay, no_decay = optimizer.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decay["params"])
    assert any(p is model.embedding.role_embed.weight for p in no_decay["params"])
    assert sum(len(g["params"]) for g in optimizer.param_groups) == len(list(model.parameters()))

    scheduler = build_scheduler(optimizer, total_steps=100, warmup_steps=10)
    lrs = []
    for _ in range(100):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    assert lrs[0] < lrs[9] == max(lrs) and abs(lrs[9] - 3e-4) < 1e-9
    assert abs(lrs[-1] - 3e-5) < 3e-6


def test_residual_projection_init_scaling():
    model = TransformAdder(tier=2, n_layers=8)
    q_std = model.layers[0].attn.q_proj.weight.std().item()
    out_std = model.layers[0].attn.out_proj.weight.std().item()
    assert abs(q_std - 0.02) < 0.002
    assert abs(out_std - 0.02 / 4) < 0.001  # 1 / sqrt(2 * 8) = 1/4


def test_evaluate_runs_end_to_end():
    for paradigm, tier in [("sm_mh", 1), ("sm_seq", 2), ("sm_seq_rev", 4), ("sm_fork", 4), ("sm_fork", 5), ("ar", 3), ("sm_mh", 4), ("sm_seq", 5), ("sm_mh", 6), ("ar", 6), ("sm_fork", 7)]:
        model = small_model(tier, "multi_head" if paradigm == "sm_mh" else "single_head")
        results = evaluate(model, paradigm, tier, train_bases=[2, 10], num_problems=1)
        for cell in ("id", "ood_base3", "ood_base12", "extrap_len_base12", "canon_ood_base12"):
            assert 0.0 <= results[f"{cell}_em"] <= 1.0
            assert 0.0 <= results[f"{cell}_step_acc"] <= 1.0
