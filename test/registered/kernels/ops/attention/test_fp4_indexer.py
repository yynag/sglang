from __future__ import annotations

import sys
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.attention.dsv4 import (
    CompressorDecodePlan,
    compress_norm_rope_store,
    fused_q_indexer_rope_hadamard_fp4_quant,
)
from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    finish_paged_indexer_topk,
    fp4_index_logits_decode,
    fp4_index_logits_paged,
)
from sglang.kernels.ops.attention.dsv4.rope import (
    apply_rotary_emb_triton,
    precompute_freqs_cis,
)
from sglang.kernels.ops.kvcache.dsv4_fp4_index_k_cache import store_fp4_index_k_cache
from sglang.srt.utils import get_device, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.deepseek_v4.common import (
    quantize_fp4_indexer_ref,
    store_fp4_index_cache_ref,
)

register_cuda_ci(est_time=50, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=50, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-small")

_is_xpu = is_xpu()
if _is_xpu:
    from sgl_kernel import hadamard_transform
else:
    from sglang.kernels.ops.quantization.hadamard import hadamard_transform

HEAD_DIM = 128
FP4_DIM = HEAD_DIM // 2
SCALE_BYTES = 4
PAGE_SIZE = 64


@pytest.mark.parametrize("num_tokens", [1, 16, 96])
def test_fp4_fused_norm_rope_store_layout(num_tokens: int) -> None:
    torch.manual_seed(num_tokens + 100)
    num_pages = max(1, (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE)
    compress_ratio = 4
    kv = torch.randn(num_tokens, HEAD_DIM, device=get_device(), dtype=torch.bfloat16)
    norm_weight = torch.randn(HEAD_DIM, device=get_device(), dtype=torch.bfloat16)
    seq_lens = (
        torch.arange(1, num_tokens + 1, device=get_device(), dtype=torch.int64)
        * compress_ratio
    )
    req_pool_indices = torch.arange(num_tokens, device=get_device(), dtype=torch.int64)
    plan = CompressorDecodePlan.generate_legacy(
        compress_ratio, req_pool_indices, seq_lens
    )
    loc = torch.arange(num_tokens, device=get_device(), dtype=torch.int64)
    freqs_cis = precompute_freqs_cis(
        64, int(seq_lens.max().item()) + 1, 0, 10000, 1, 32, 1
    ).to(get_device())
    cache = torch.zeros(
        num_pages,
        PAGE_SIZE * (FP4_DIM + SCALE_BYTES),
        device=get_device(),
        dtype=torch.uint8,
    )

    compress_norm_rope_store(
        kv.clone(),
        plan,
        norm_weight=norm_weight,
        norm_eps=1.0e-6,
        freq_cis=freqs_cis,
        out_loc=loc,
        kvcache=cache,
        page_size=PAGE_SIZE,
        use_fp4=True,
    )

    ref = kv.float()
    ref = ref * torch.rsqrt((ref * ref).sum(dim=-1, keepdim=True) / HEAD_DIM + 1.0e-6)
    ref = ref * norm_weight.float()
    freqs = torch.view_as_real(freqs_cis).flatten(-2)[
        (seq_lens - compress_ratio).long()
    ]
    rope = ref[:, 64:].reshape(num_tokens, 32, 2)
    freqs = freqs.reshape(num_tokens, 32, 2)
    rope_out = torch.empty_like(rope)
    rope_out[..., 0] = rope[..., 0] * freqs[..., 0] - rope[..., 1] * freqs[..., 1]
    rope_out[..., 1] = rope[..., 0] * freqs[..., 1] + rope[..., 1] * freqs[..., 0]
    ref[:, 64:] = rope_out.reshape(num_tokens, 64)
    ref = hadamard_transform(ref.contiguous(), scale=HEAD_DIM**-0.5)
    ref_fp4, ref_sf = quantize_fp4_indexer_ref(ref)

    expected = store_fp4_index_cache_ref(
        ref_fp4,
        ref_sf,
        loc.to(torch.int64),
        num_pages,
    )
    torch.testing.assert_close(cache, expected)


@pytest.mark.skipif(
    _is_xpu,
    reason="fused_q_indexer_rope_hadamard_fp4_quant is not supported by Intel GPU",
)
@pytest.mark.parametrize("batch_size", [1, 5, 17])
def test_fp4_fused_q_indexer_rope_hadamard_quant(batch_size: int) -> None:
    torch.manual_seed(batch_size + 200)
    num_heads = 8
    rope_dim = 64
    weight_scale = HEAD_DIM**-0.5 * num_heads**-0.5
    q = torch.randn(
        batch_size, num_heads, HEAD_DIM, device=get_device(), dtype=torch.bfloat16
    )
    weight = torch.randn(
        batch_size, num_heads, device=get_device(), dtype=torch.bfloat16
    )
    positions = (
        torch.arange(batch_size, device=get_device(), dtype=torch.int32) * 7
    ) % 63
    freqs_cis = precompute_freqs_cis(rope_dim, 64, 0, 10000, 1, 32, 1).to(get_device())

    (q_fp4, q_sf), weights_out = fused_q_indexer_rope_hadamard_fp4_quant(
        q, weight, weight_scale, freqs_cis, positions
    )

    ref = q.clone()
    apply_rotary_emb_triton(ref[..., -rope_dim:], freqs_cis, positions=positions)
    ref = hadamard_transform(ref.contiguous(), scale=HEAD_DIM**-0.5)
    ref_fp4, ref_sf = quantize_fp4_indexer_ref(ref.view(-1, HEAD_DIM))
    ref_fp4 = ref_fp4.view(batch_size, num_heads, FP4_DIM)
    ref_sf = ref_sf.view(batch_size, num_heads)

    torch.testing.assert_close(q_fp4.view(torch.uint8), ref_fp4)
    torch.testing.assert_close(q_sf, ref_sf)
    torch.testing.assert_close(weights_out.squeeze(-1), weight.float() * weight_scale)


def _index_q_inputs(tokens: int, heads: int, kind: str, gen: torch.Generator):
    shape = (tokens, heads, HEAD_DIM)
    device = get_device()
    x = torch.randn(shape, device=device, generator=gen)
    if kind == "row_scales":
        x = x * torch.exp(
            torch.randn(tokens, heads, 1, device=device, generator=gen) * 6
        )
    elif kind == "sparse":
        x = x * (torch.rand(shape, device=device, generator=gen) < 0.1)
    elif kind == "tiny":
        # Below both fp4 amax floors.
        x = x * 1e-30
    elif kind == "grid_ties":
        grid = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], device=device)
        x = x.sign() * grid[torch.randint(0, 8, shape, device=device, generator=gen)]
    elif kind == "extreme":
        # Per-row scales across the whole bf16 exponent range, subnormals included.
        exponent = torch.randint(
            -150, 126, (tokens, heads, 1), device=device, generator=gen
        )
        x = x * torch.exp2(exponent.float())
    elif kind == "nonfinite":
        u = torch.rand(shape, device=device, generator=gen)
        x = torch.where(u < 0.003, float("inf"), x)
        x = torch.where((u >= 0.003) & (u < 0.006), float("-inf"), x)
        x = torch.where((u >= 0.006) & (u < 0.009), float("nan"), x)
    x = x.bfloat16()
    if kind == "sparse":
        # Signed zeros: the packer clears the -0 code.
        negative = torch.rand(shape, device=device, generator=gen) < 0.5
        x = torch.where((x == 0) & negative, -0.0, x).bfloat16()
    positions = torch.randint(0, 200000, (tokens,), device=device, generator=gen)
    head_weights = (
        torch.randn(tokens, heads, device=device, generator=gen) * 3
    ).bfloat16()
    return x, positions, head_weights


@pytest.mark.skipif(
    _is_xpu
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10,
    reason="Only SM100-family GPUs take the block kernel; elsewhere both calls use the warp kernel",
)
@pytest.mark.parametrize(
    "kind",
    ["normal", "row_scales", "sparse", "tiny", "grid_ties", "extreme", "nonfinite"],
)
@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64])
def test_index_q_rope_pack_prefill_matches_decode_kernel(kind, pos_dtype) -> None:
    """Calls of >= 8192 (token, head) rows take the one-thread-per-fp4-block kernel
    on SM100-family GPUs; 128-token calls keep the warp-per-row kernel. Both must
    write the same bytes."""
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_rope import (
        index_q_rope_pack_weights,
    )

    heads, tokens = 32, 4100
    gen = torch.Generator(device=get_device()).manual_seed(0)
    freqs = torch.randn(200000, 32, 2, device=get_device(), generator=gen)
    freqs = (freqs / freqs.norm(dim=-1, keepdim=True)).flatten(1).contiguous()
    x, positions, head_weights = _index_q_inputs(tokens, heads, kind, gen)
    positions = positions.to(pos_dtype)

    def call(rows: slice):
        return index_q_rope_pack_weights(
            x[rows], freqs, positions[rows], head_weights[rows], 0.0625
        )

    whole = call(slice(None))
    parts = [call(slice(i, i + 128)) for i in range(0, tokens, 128)]
    for j, name in enumerate(("payload", "scale", "weights")):
        assert torch.equal(whole[j], torch.cat([p[j] for p in parts])), name


def _make_logits_case(rows: int, heads: int, width: int, seed: int = 719):
    """Use exact dyadic inputs to isolate logits/masking from FP32 dot order."""
    torch.manual_seed(seed)
    device = get_device()
    q = (
        torch.randint(-8, 9, (rows, heads, HEAD_DIM), device=device).to(torch.bfloat16)
        / 8
    )
    weights = torch.randint(-8, 9, (rows, heads), device=device).to(torch.bfloat16) / 8
    # Six adjacent verify rows share one request, with shuffled physical pages.
    groups = (rows + 5) // 6
    padded = max(PAGE_SIZE, (width + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE)
    pages = torch.randperm(groups * padded // PAGE_SIZE, device=device)
    locations = (
        pages[:, None] * PAGE_SIZE + torch.arange(PAGE_SIZE, device=device)
    ).flatten()
    keys = (
        torch.randint(-8, 9, (groups * padded, HEAD_DIM), device=device).to(
            torch.bfloat16
        )
        / 8
    )
    table = torch.zeros(
        groups * padded // PAGE_SIZE,
        PAGE_SIZE * 68,
        device=device,
        dtype=torch.uint8,
    )
    store_fp4_index_k_cache(keys, table, locations, page_size=PAGE_SIZE, rne=True)
    slots = locations.reshape(groups, padded)[
        torch.arange(rows, device=device) // 6, :width
    ].contiguous()
    patterns = [0, 1, 63, 64, 65, max(0, width - 1), width]
    lens = torch.tensor(
        [min(width, patterns[i % len(patterns)]) for i in range(rows)],
        device=device,
        dtype=torch.int64,
    )
    lens[-1] = width
    return q, weights, slots, lens, table


def _reference_logits(q, weights, slots, lens, table):
    """Independent E2M1/UE8M0 decoding with the prescribed BF16 rounds."""
    rows, width = slots.shape
    position = torch.arange(width, device=q.device)
    # Invisible positions have no cache mapping contract.
    safe_slots = torch.where(position[None, :] < lens[:, None], slots, 0)
    page, offset = safe_slots // PAGE_SIZE, safe_slots % PAGE_SIZE
    payload = table[
        page[:, :, None],
        offset[:, :, None] * 64 + torch.arange(64, device=q.device),
    ]
    codes = torch.stack((payload & 15, payload >> 4), dim=-1).flatten(-2).long()
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=q.device)
    values = levels[codes & 7]
    values = torch.where((codes & 8) != 0, -values, values)
    exponents = table[
        page[:, :, None],
        PAGE_SIZE * 64 + offset[:, :, None] * 4 + torch.arange(4, device=q.device),
    ].int()
    scales = torch.ldexp(
        torch.ones_like(exponents, dtype=torch.float32), exponents - 127
    )
    keys = (values * scales.repeat_interleave(32, dim=-1)).to(torch.bfloat16)
    scores = torch.bmm(q.double(), keys.double().transpose(1, 2)).to(torch.bfloat16)
    products = (scores.double().clamp_min(0) * weights.double()[:, :, None]).to(
        torch.bfloat16
    )
    logits = products.double().sum(dim=1).to(torch.bfloat16).float()
    return logits.masked_fill(position[None, :] >= lens[:, None], -torch.inf)


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("rows", [1, 6, 13])
@pytest.mark.parametrize("width", [0, 1, 63, 64, 65, 129])
def test_fp4_logits_visibility_boundaries(heads, rows, width):
    args = _make_logits_case(rows, heads, width)
    actual = fp4_index_logits_decode(*args, PAGE_SIZE)
    expected = _reference_logits(*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.shape == (rows, width)
    assert actual.dtype == torch.float32


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("rows", [1, 6])
def test_fp4_logits_noncontiguous_queries_and_weights(heads, rows):
    q, weights, slots, lens, table = _make_logits_case(rows, heads, 129)
    q_storage = torch.empty(rows, heads, HEAD_DIM * 2, device=q.device, dtype=q.dtype)
    q_view = q_storage[..., ::2]
    q_view.copy_(q)
    weight_storage = torch.empty(rows, heads * 2, device=q.device, dtype=weights.dtype)
    weight_view = weight_storage[:, ::2]
    weight_view.copy_(weights)
    actual = fp4_index_logits_decode(q_view, weight_view, slots, lens, table, PAGE_SIZE)
    expected = _reference_logits(q, weights, slots, lens, table)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if rows == 1:
        # A singleton batch can be contiguous even with an arbitrary batch stride.
        q_view = q.as_strided(q.shape, (0, HEAD_DIM, 1))
        assert q_view.is_contiguous()
        actual = fp4_index_logits_decode(q_view, weights, slots, lens, table, PAGE_SIZE)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.skipif(_is_xpu, reason="CUDA graph replay requires a CUDA device")
def test_fp4_logits_graph_replay_updates_visibility_and_mapping(heads):
    args = _make_logits_case(13, heads, 193)
    for _ in range(3):
        fp4_index_logits_decode(*args, PAGE_SIZE)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fp4_index_logits_decode(*args, PAGE_SIZE)
    q, weights, slots, lens, table = args
    full_slots = slots.clone()
    for step, visible in enumerate([193, 0, 1, 63, 64, 65, 129, 193]):
        lens.copy_((visible - torch.arange(13, device=q.device) % 6).clamp_min(0))
        slots.copy_(full_slots.roll(step, dims=1))
        q.neg_()
        weights.neg_()
        table.copy_(table.roll(1, dims=0))
        graph.replay()
        expected = _reference_logits(*args)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        invalid = torch.arange(193, device=q.device)[None, :] >= lens[:, None]
        assert torch.isneginf(actual[invalid]).all().item()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_fp4_logits_invisible_tiles_ignore_nonfinite_queries(value):
    args = _make_logits_case(6, 32, 129)
    q, weights, slots, lens, table = args
    q.fill_(value)
    weights.fill_(value)
    lens.zero_()
    slots.fill_(-1)
    actual = fp4_index_logits_decode(*args, PAGE_SIZE)
    assert torch.isneginf(actual).all().item()


@pytest.mark.skipif(_is_xpu, reason="Paged decode uses CUDA persistent kernels")
@pytest.mark.parametrize(
    "batch,ratio,width,masked",
    [
        (6, 1, 193, False),
        (6, 2, 193, True),
        (6, 1, 16385, True),
        (6, 2, 8193, False),
        (6, 1, 32769, False),
        (64, 1, 32769, True),
    ],
)
def test_fp4_paged_logits_replay(batch, ratio, width, masked):
    from sglang.kernels.ops.attention.dsv4.candidate_blocks import amax_topk_blocks
    from sglang.kernels.ops.attention.dsv4.topk import (
        plan_topk_v2,
        topk_transform_paged_v2,
    )

    q, weights, slots, lens, table = _make_logits_case(batch, 32, width)
    capacity = 1048580 // ratio
    req = torch.arange(batch, device=q.device, dtype=torch.int32)
    req_table = torch.full(
        (batch, capacity * ratio), -1, device=q.device, dtype=torch.int32
    )
    req_table[:, : width * ratio : ratio] = (slots * ratio).to(torch.int32)
    lengths = lens.to(torch.int32)
    candidate_mask = (
        torch.rand(batch, capacity, device=q.device) > 0.3 if masked else None
    )
    k = min(512, width)

    def run():
        plan = plan_topk_v2(lengths)
        scores = fp4_index_logits_paged(
            q,
            weights,
            req,
            req_table,
            lengths,
            table,
            PAGE_SIZE,
            capacity,
            ratio,
            candidate_mask,
        )
        indices = torch.empty((batch, k), dtype=torch.int32, device=q.device)
        topk_transform_paged_v2(scores, lengths, None, indices, 1, plan)
        pages = torch.empty((batch + 1, 528), dtype=torch.int32, device=q.device)[
            :, :512
        ]
        raw = torch.empty_like(pages) if ratio == 1 else None
        finish_paged_indexer_topk(
            indices, scores, lengths, req, req_table, pages, raw, ratio, masked
        )
        return scores, indices, pages, raw

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scores, indices, pages, raw = run()
    for step, visible in enumerate([width, 0, 1, 63, 64, 65, width]):
        lens.copy_((visible - torch.arange(batch, device=q.device)).clamp_min(0))
        lengths.copy_(lens)
        req.copy_(req.roll(1))
        slots.copy_(slots.roll(step, dims=1))
        req_table[req.long(), : width * ratio : ratio] = (slots * ratio).to(torch.int32)
        q.neg_()
        # Poison the unused capacity: length-aware top-k must never select it.
        scores.fill_(torch.inf)
        graph.replay()
        expected = _reference_logits(q, weights, slots, lens, table)
        if masked:
            expected.masked_fill_(~candidate_mask[:, :width], -torch.inf)
        visible_mask = torch.arange(width, device=q.device)[None, :] < lens[:, None]
        torch.testing.assert_close(
            scores[:, :width][visible_mask], expected[visible_mask], atol=0, rtol=0
        )
        blocks = amax_topk_blocks(scores, lengths, (lengths + 7) // 8, 2048)
        assert (pages[batch] == -1).all().item()
        if raw is not None:
            assert (raw[batch] == -1).all().item()
        for row, length in enumerate(lens.tolist()):
            selected = indices[row][indices[row] >= 0].long()
            assert selected.numel() == selected.unique().numel() == min(k, length)
            assert not selected.numel() or selected.max().item() < length
            torch.testing.assert_close(
                expected[row, selected].sort().values,
                expected[row, :length].topk(min(k, length)).values.sort().values,
                atol=0,
                rtol=0,
            )
            kept = (
                selected[expected[row, selected] > -torch.inf] if masked else selected
            )
            kept = kept.sort().values
            expected_pages = torch.full_like(pages[row], -1)
            expected_pages[: kept.numel()] = slots[row, kept].to(torch.int32)
            torch.testing.assert_close(pages[row], expected_pages, rtol=0, atol=0)
            if raw is not None:
                expected_raw = torch.full_like(raw[row], -1)
                expected_raw[: kept.numel()] = kept.to(torch.int32)
                torch.testing.assert_close(raw[row], expected_raw, rtol=0, atol=0)
            nblocks = (length + 7) // 8
            chosen = blocks[row][blocks[row] >= 0].long()
            assert chosen.numel() == chosen.unique().numel() == min(2048, nblocks)
            if nblocks:
                assert chosen.max().item() < nblocks
                assert nblocks - 1 in chosen.tolist()
                padded = torch.full((nblocks * 8,), -torch.inf, device=q.device)
                padded[:length] = expected[row, :length]
                maxima = padded.view(-1, 8).amax(1)
                maxima[-1] = torch.inf
                torch.testing.assert_close(
                    maxima[chosen].sort().values,
                    maxima.topk(min(2048, nblocks)).values.sort().values,
                    atol=0,
                    rtol=0,
                )


@pytest.mark.skipif(_is_xpu, reason="Paged decode uses CUDA persistent kernels")
@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("ratio,masked", [(1, False), (2, True)])
def test_fp4_paged_logits_balanced_matches_per_row_replay(heads, ratio, masked):
    """One long verify group among short rows: the balanced split and the per-row grid
    must produce the same bits for every visible score, across graph replays."""
    from sglang.kernels.ops.attention.dsv4 import fp4_indexer

    batch, width = 40, 20000
    q, weights, slots, lens, table = _make_logits_case(batch, heads, width)
    capacity = 32768
    req = (torch.arange(batch, device=q.device) // 6).to(torch.int32)
    # Positions past ``width`` map to slot 0, so a length past max_len stays readable.
    req_table = torch.zeros(
        (req.max().item() + 1, capacity * ratio), device=q.device, dtype=torch.int32
    )
    req_table[req.long(), : width * ratio : ratio] = (slots * ratio).to(torch.int32)
    lengths = torch.zeros(batch, device=q.device, dtype=torch.int32)
    candidate_mask = (
        torch.rand(batch, capacity, device=q.device) > 0.3 if masked else None
    )
    args = (q, weights, req, req_table, lengths, table, PAGE_SIZE, capacity, ratio)

    def score(balanced):
        if balanced:
            return fp4_index_logits_paged(*args, candidate_mask)
        with patch.object(fp4_indexer, "_PAGED_BALANCED_MAX_BATCH", 0):
            return fp4_index_logits_paged(*args, candidate_mask)

    for _ in range(3):
        fp4_index_logits_paged(*args, candidate_mask)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fp4_index_logits_paged(*args, candidate_mask)
    for step, long_len in enumerate([width, 0, 1, 129, 20000 - 63, width]):
        short = torch.tensor([0, 1, 63, 64, 65, 127, 128, 3000], device=q.device)
        lengths.copy_(short[torch.arange(batch, device=q.device) % 8].to(torch.int32))
        lengths[6 * (step % 6) : 6 * (step % 6) + 6] = long_len
        lengths[-1] = capacity + 7  # past max_len: clamped, not read past the table
        q.neg_()
        graph.replay()
        expected = score(False)
        visible = (
            torch.arange(capacity, device=q.device)[None, :]
            < lengths.clamp(max=capacity)[:, None]
        )
        assert torch.equal(
            actual[visible].view(torch.int32), expected[visible].view(torch.int32)
        )
        assert torch.equal(
            score(True)[visible].view(torch.int32), expected[visible].view(torch.int32)
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="SM120 DeepGEMM kernel/planner contract",
)
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("c4_fp4", [False, True])
def test_sm120_full_topk_native_planner_replay(ratio, c4_fp4):
    from sglang.kernels.ops.attention.dsv4.index_logits import flat_index_logits_tiles
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    from sglang.srt.layers.attention.dsv4.v41_indexer import make_full_topk_indexer
    from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import (
        get_deep_gemm_decode_data,
    )
    from sglang.srt.layers.attention.dsv4.v41_indexer.types import DecodeInputs

    torch.manual_seed(1200 + ratio)
    batch, heads, capacity, topk = 3, 32, 512 // ratio, 64
    device = torch.device("cuda")
    full_pages = torch.arange(
        1, batch * 2 + 1, dtype=torch.int32, device=device
    ).view(batch, 2)
    req_table = (
        full_pages[:, :, None] * 256 + torch.arange(256, device=device)
    ).reshape(batch, 512).int()
    slots = req_table[:, ::ratio].long() // ratio
    num_pages = (batch * 2 + 1) * 256 // ratio // PAGE_SIZE
    table = torch.zeros(num_pages, PAGE_SIZE * 68, dtype=torch.uint8, device=device)
    keys = torch.randn(
        num_pages * PAGE_SIZE, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    store_fp4_index_k_cache(
        keys,
        table,
        torch.arange(keys.shape[0], device=device),
        page_size=PAGE_SIZE,
        rne=True,
    )
    pool = SimpleNamespace(
        stage_layer_ids=(0,),
        get_index_k_page_size=lambda _: PAGE_SIZE,
        get_index_k_with_scale_buffer=lambda _: table,
    )
    backend = object.__new__(DeepseekV4AttnBackend)
    backend.page_size = 256
    backend.token_to_kv_pool = pool
    backend.full_topk_indexer = make_full_topk_indexer(
        token_to_kv_pool=pool,
        req_to_token=req_table,
        index_heads=heads,
        index_topk=topk,
    )
    assert backend.full_topk_indexer.use_deep_gemm_decode
    backend.decode_candidates = None
    backend._deep_gemm_decode_ratios = backend._get_deep_gemm_decode_ratios(
        SimpleNamespace(
            compress_ratios=[ratio],
            index_source_layer_ids=[0],
            candidate_source_layer_id=-1,
        )
    )
    backend.enable_deepseek_v4_fp4_indexer = c4_fp4
    backend.dsa_topk_backend = SimpleNamespace(should_use_topk_v2=lambda: True)
    lens = torch.tensor([1, 129, 193], dtype=torch.int32, device=device)
    positions = (lens * ratio - 1).long()
    core = SimpleNamespace(
        page_table=full_pages,
        c1_topk_lengths_clamp1=lens,
        c2_topk_lengths_clamp1=lens,
    )
    q_lora = torch.randn(batch, heads * HEAD_DIM, dtype=torch.bfloat16, device=device)
    weights = torch.randn(batch, heads, dtype=torch.bfloat16, device=device)
    indexer = SimpleNamespace(
        n_local_heads=heads,
        n_heads=heads,
        index_head_dim=HEAD_DIM,
        index_topk=topk,
        head_weight_scale=1.0,
        wq_b=lambda x: (x, None),
        head_weights_raw=lambda x: x,
    )
    freqs = torch.ones(512 * ratio, 32, dtype=torch.complex64, device=device)
    out = torch.empty(batch, topk, dtype=torch.int32, device=device)

    def inputs():
        metadata = backend.init_forward_metadata_indexer(core, compress_ratio=ratio)
        assert metadata.force_deep_gemm_metadata
        return DecodeInputs(
            indexer=indexer,
            layer_id=0,
            compress_ratio=ratio,
            freqs_cis=freqs,
            x=weights,
            q_lora=q_lora,
            positions=positions,
            req_rows=torch.arange(batch, device=device),
            paged_metadata=metadata,
            is_verify=False,
            out_page_indices=out,
        )

    def run():
        backend.full_topk_indexer.topk_decode(inputs())

    # These switches concern C4 FP8 and must not suppress a low-ratio FP4 plan.
    with (
        envs.SGLANG_OPT_USE_JIT_INDEXER_METADATA.override(True),
        envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(True),
    ):
        for _ in range(3):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        flat_slots = slots.flatten()
        page, offset = flat_slots // PAGE_SIZE, flat_slots % PAGE_SIZE
        payload = table[
            page[:, None], offset[:, None] * 64 + torch.arange(64, device=device)
        ]
        scales = table[
            page[:, None],
            PAGE_SIZE * 64 + offset[:, None] * 4 + torch.arange(4, device=device),
        ]
        kv = (
            payload.contiguous().view(torch.int8),
            scales.contiguous().view(torch.int32).flatten(),
        )
        starts = torch.arange(batch, dtype=torch.int32, device=device) * capacity
        for lengths in ([1, 127, 128], [129, 193, 65], [193, 1, 129]):
            lens.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
            positions.copy_((lens * ratio - 1).long())
            q_lora.neg_()
            out.fill_(-12345)
            graph.replay()
            data = get_deep_gemm_decode_data(inputs(), pool)
            expected = next(
                flat_index_logits_tiles(
                    q=(data.q_fp4.squeeze(1), data.q_sf.squeeze(1)),
                    kv=kv,
                    weights=data.weights,
                    starts=starts,
                    lengths=lens,
                    context_lengths=[capacity] * batch,
                    budget_bytes=2 << 20,
                )
            )[1]
            for row, length in enumerate(lengths):
                pages = out[row][out[row] >= 0]
                selected = (slots[row, :, None] == pages[None, :]).nonzero()[:, 0]
                count = min(topk, length)
                assert pages.numel() == selected.unique().numel() == count
                assert (selected < length).all().item()
                assert (out[row][count:] == -1).all().item()
                torch.testing.assert_close(
                    expected[row, selected].sort().values,
                    expected[row, :length].topk(count).values.sort().values,
                    atol=1e-4,
                    rtol=1e-4,
                )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="SM120 grouped verify and shared early metadata",
)
@pytest.mark.parametrize("ratio", [1, 2])
def test_sm120_early_verify_chunked_variants_replay(ratio):
    """Real raw-verify builder, early install/copy and variant consumer dispatch."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.deepseek_v4_backend import (
        DeepseekV4AttnBackend,
        DSV4RawVerifyMetadata,
        _GraphBucket,
    )
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Indexer
    from sglang.srt.layers.attention.dsv4.v41_indexer import (
        make_candidate_indexer,
        make_full_topk_indexer,
    )
    from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
        DenseBlocksBackend,
    )
    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.srt.model_executor.runner_utils import capture_mode

    torch.manual_seed(1210 + ratio)
    requests, group, heads, full_capacity, topk = 513, 8, 32, 1536, 512
    rows = requests * group
    device = torch.device("cuda")
    full_pages = torch.arange(
        1, requests * (full_capacity // 256) + 1, dtype=torch.int32, device=device
    ).view(requests, -1)
    req_table = (
        full_pages[:, :, None] * 256 + torch.arange(256, device=device)
    ).reshape(requests, full_capacity).int()
    slots = (requests * full_capacity + 256) // ratio
    keys = torch.randn(slots, HEAD_DIM, dtype=torch.bfloat16, device=device)
    table = torch.zeros(
        slots // PAGE_SIZE, PAGE_SIZE * 68, dtype=torch.uint8, device=device
    )
    store_fp4_index_k_cache(
        keys, table, torch.arange(slots, device=device), page_size=PAGE_SIZE, rne=True
    )
    del keys
    pool = SimpleNamespace(
        stage_layer_ids=(0,),
        request_window=None,
        swa_req_ring_size=None,
        full_to_swa_index_mapping=torch.arange(
            requests * full_capacity + 256, device=device
        ),
        translate_loc_from_full_to_swa=lambda loc: loc,
        get_index_k_page_size=lambda _: PAGE_SIZE,
        get_index_k_with_scale_buffer=lambda _: table,
    )
    backend = object.__new__(DeepseekV4AttnBackend)
    backend.page_size = 256
    backend.req_to_token = req_table
    backend.token_to_kv_pool = pool
    backend.cuda_int32_kwargs = {"device": device, "dtype": torch.int32}
    backend.MAX_SEQ_LEN_FOR_CAPTURE = full_capacity
    backend.index_topk = topk
    backend.present_ratios = backend.low_ratios = (ratio,)
    backend.has_c4 = backend.has_c128 = False
    backend.enable_deepseek_v4_fp4_indexer = False
    backend.speculative_num_draft_tokens = group
    backend.dsa_topk_backend = DSATopKBackend.SGL_KERNEL
    backend.online_c128_mtp = SimpleNamespace(state_slot_offset=lambda: 0)
    backend.cuda_graph_metadata_of_bucket_and_bs = {_GraphBucket.TARGET_VERIFY: {}}
    backend._early_verify_metadata_ids = set()
    backend.enable_early_verify_metadata()
    backend.prefill_candidates, backend.decode_candidates = make_candidate_indexer(
        token_to_kv_pool=pool,
        req_to_token=req_table,
        page_size=256,
        candidate_topk_blocks=4,
        candidate_block_size=8,
    )
    assert isinstance(backend.decode_candidates, DenseBlocksBackend)
    backend.full_topk_indexer = make_full_topk_indexer(
        token_to_kv_pool=pool,
        req_to_token=req_table,
        index_heads=heads,
        index_topk=topk,
    )
    backend._deep_gemm_decode_ratios = backend._get_deep_gemm_decode_ratios(
        SimpleNamespace(
            compress_ratios=[ratio],
            index_source_layer_ids=[0],
            candidate_source_layer_id=0,
        )
    )
    assert not backend._deep_gemm_decode_ratios
    indexer = SimpleNamespace(
        n_local_heads=heads,
        n_heads=heads,
        index_head_dim=HEAD_DIM,
        rope_head_dim=64,
        index_topk=topk,
        head_weight_scale=HEAD_DIM**-0.5 * heads**-0.5,
        is_candidate_source=True,
        uses_candidates=False,
        wq_b=lambda x: (x, None),
        head_weights_raw=lambda x: x,
    )
    indexer.queries = MethodType(DeepseekV41Indexer.queries, indexer)
    indexer.head_weights = MethodType(DeepseekV41Indexer.head_weights, indexer)
    layer = SimpleNamespace(
        layer_id=0,
        compress_ratio=ratio,
        indexer=indexer,
        freqs_cis=torch.ones(full_capacity, 32, dtype=torch.complex64, device=device),
    )
    q = torch.randn(rows, heads * HEAD_DIM, dtype=torch.bfloat16, device=device)
    weights = torch.randn(rows, heads, dtype=torch.bfloat16, device=device)
    req_rows = torch.arange(
        requests, device=device, dtype=torch.int32
    ).repeat_interleave(group)
    raw = DSV4RawVerifyMetadata(
        req_pool_indices=torch.arange(requests, dtype=torch.int32, device=device),
        seq_lens=torch.empty(requests, dtype=torch.int32, device=device),
        out_cache_loc=torch.empty(rows, dtype=torch.int64, device=device),
        extend_seq_lens=torch.full(
            (requests,), group, dtype=torch.int32, device=device
        ),
    )
    batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)

    def install(base, variant):
        raw.seq_lens.fill_(base * ratio)
        positions = base * ratio + torch.arange(group, device=device)
        raw.out_cache_loc.copy_(req_table[:, positions].flatten())
        with patch.object(capture_mode, "_capture_attention_variant", variant):
            backend._install_early_verify_metadata(
                graph_key=requests, raw_metadata=raw, out_cache_loc=raw.out_cache_loc
            )

    def run(variant):
        with patch.object(capture_mode, "_capture_attention_variant", variant):
            backend._low_ratio_index_topk(
                layer,
                weights,
                q,
                req_rows,
                backend.forward_metadata.core_metadata.positions,
                batch,
            )

    with envs.SGLANG_OPT_USE_TOPK_V2.override(True):
        install(127, "candidate_unfiltered")
        shared = backend.forward_metadata
        metadata = (
            shared.c1_indexer_metadata if ratio == 1 else shared.c2_indexer_metadata
        )
        assert metadata.rows_per_request == group
        assert [(s.start, s.stop) for s, _ in metadata.row_chunks()] == [
            (0, 4096), (4096, 4104)
        ]
        plans = tuple(metadata.deep_gemm_metadata)
        topk_plans = tuple(metadata.topk_metadata_chunks)
        pointers = tuple(t.data_ptr() for t in plans + topk_plans)
        graphs = {}
        for base, variant in (
            (127, "candidate_unfiltered"), (129, "candidate_filtered")
        ):
            install(base, variant)
            assert backend.forward_metadata is shared
            for _ in range(3):
                run(variant)
            graphs[variant] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[variant]):
                run(variant)
        for base in (127, 129, 511, 513):
            # Refresh through the filtered early path, then replay the already
            # captured unfiltered graph: both variants share these plan owners.
            install(base, "candidate_filtered")
            assert all(a is b for a, b in zip(plans, metadata.deep_gemm_metadata))
            assert all(
                a is b for a, b in zip(topk_plans, metadata.topk_metadata_chunks)
            )
            assert tuple(t.data_ptr() for t in plans + topk_plans) == pointers
            graphs["candidate_filtered"].replay()
            graphs["candidate_unfiltered"].replay()
            actual = shared.core_metadata.sparse_page_indices(ratio).clone()
            with patch.object(
                capture_mode, "_capture_attention_variant", "candidate_unfiltered"
            ):
                backend.forward_metadata = (
                    backend.make_forward_metadata_from_raw_verify(
                        raw, graph_backed=True
                    )
                )
                run("candidate_unfiltered")
            expected = backend.forward_metadata.core_metadata.sparse_page_indices(ratio)
            torch.testing.assert_close(
                actual.sort(-1).values, expected.sort(-1).values, atol=0, rtol=0
            )
            lens = shared.core_metadata.seq_lens_casual // ratio
            assert torch.equal((actual >= 0).sum(-1), lens.clamp_max(topk).long())
            assert (actual[actual < 0] == -1).all().item()
            backend.forward_metadata = shared


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="SM120 C4 paged consumer and native planner",
)
@pytest.mark.parametrize("fp4", [False, True])
def test_sm120_c4_native_planner_consumer_replay(fp4):
    """C4's real quantizer/forward versus flat DeepGEMM scores, not a low-ratio route."""
    import deep_gemm

    from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
    from sglang.srt.layers.attention.dsv4.indexer import FP8_DTYPE, C4Indexer
    from sglang.srt.model_executor.forward_mode import ForwardMode

    torch.manual_seed(1240 + fp4)
    rows, heads, capacity, topk = 3, 32, 192, 64
    device = torch.device("cuda")
    pages = torch.arange(1, 10, dtype=torch.int32, device=device).view(rows, 3)
    keys = torch.randn(10 * PAGE_SIZE, HEAD_DIM, dtype=torch.bfloat16, device=device)
    table = torch.zeros(
        10, PAGE_SIZE * (68 if fp4 else 132), dtype=torch.uint8, device=device
    )
    loc = torch.arange(keys.shape[0], device=device)
    if fp4:
        store_fp4_index_k_cache(keys, table, loc, page_size=PAGE_SIZE, rne=True)
    else:
        fused_store_cache(keys, table, loc, page_size=PAGE_SIZE, type="indexer")
    backend = object.__new__(DeepseekV4AttnBackend)
    backend.page_size = 256
    backend.enable_deepseek_v4_fp4_indexer = fp4
    backend.dsa_topk_backend = DSATopKBackend.SGL_KERNEL
    backend.hisparse_coordinator = None
    backend.debug_use_external_c4_sparse_indices = False
    backend.token_to_kv_pool = SimpleNamespace(
        get_index_k_with_scale_buffer=lambda **_: table
    )
    lens = torch.tensor([1, 127, 128], dtype=torch.int32, device=device)
    positions = (lens * 4 - 1).long()
    out = torch.empty(rows, topk, dtype=torch.int32, device=device)
    raw_out = torch.empty_like(out)
    core = SimpleNamespace(
        page_table=pages,
        c4_topk_lengths_raw=lens,
        positions=positions,
        c4_sparse_page_indices=out,
        c4_sparse_raw_indices=raw_out,
    )
    indexer = SimpleNamespace(
        layer_id=0,
        use_fp4_indexer=fp4,
        n_local_heads=heads,
        head_dim=HEAD_DIM,
        weight_scale=HEAD_DIM**-0.5 * heads**-0.5,
        freqs_cis=torch.ones(1024, 32, dtype=torch.complex64, device=device),
        wq_b=lambda x: (x, None),
        weights_proj=lambda x: (x, None),
    )
    indexer.compute_q = MethodType(C4Indexer.compute_q, indexer)
    indexer.compute_weights = MethodType(C4Indexer.compute_weights, indexer)
    q = torch.randn(rows, heads * HEAD_DIM, dtype=torch.bfloat16, device=device)
    weights = torch.randn(rows, heads, dtype=torch.bfloat16, device=device)
    batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)

    def run():
        backend.forward_metadata = SimpleNamespace(
            indexer_metadata=backend.init_forward_metadata_indexer(core),
            core_metadata=core,
        )
        backend.forward_c4_indexer(
            x=weights,
            q_lora=q,
            c4_indexer=indexer,
            forward_batch=batch,
            skip_compressor=True,
        )

    slots = (
        pages[:, :, None] * PAGE_SIZE + torch.arange(PAGE_SIZE, device=device)
    ).flatten()
    pg, off = slots // PAGE_SIZE, slots % PAGE_SIZE
    dim = 64 if fp4 else 128
    payload = table[pg[:, None], off[:, None] * dim + torch.arange(dim, device=device)]
    scales = table[
        pg[:, None],
        PAGE_SIZE * dim + off[:, None] * 4 + torch.arange(4, device=device),
    ]
    kv = (
        payload.contiguous().view(torch.int8 if fp4 else FP8_DTYPE),
        scales.contiguous().view(torch.int32 if fp4 else torch.float32).flatten(),
    )
    starts = torch.arange(rows, dtype=torch.int32, device=device) * capacity
    with (
        envs.SGLANG_OPT_USE_TOPK_V2.override(True),
        envs.SGLANG_OPT_USE_JIT_INDEXER_METADATA.override(True),
        envs.SGLANG_OPT_USE_TILELANG_INDEXER.override(False),
        envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(False),
    ):
        for _ in range(3):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for lengths in ([1, 127, 128], [129, 191, 65], [63, 64, 129]):
            lens.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
            positions.copy_((lens * 4 - 1).long())
            q.neg_()
            graph.replay()
            packed_q, w = indexer.compute_q(q, positions, weights)
            if fp4:
                expected = deep_gemm.fp8_fp4_mqa_logits(
                    packed_q,
                    kv,
                    w.squeeze(-1).float(),
                    starts,
                    starts + lens,
                    False,
                    capacity,
                    logits_dtype=torch.float32,
                )
            else:
                expected = deep_gemm.fp8_mqa_logits(
                    packed_q,
                    kv,
                    w.squeeze(-1).float(),
                    starts,
                    starts + lens,
                    False,
                    capacity,
                )
            for row, length in enumerate(lengths):
                selected = raw_out[row][raw_out[row] >= 0].long()
                count = min(topk, length)
                assert selected.numel() == selected.unique().numel() == count
                assert (selected < length).all().item()
                assert (out[row][count:] == -1).all().item()
                expected_pages = (
                    pages[row, selected // PAGE_SIZE] * PAGE_SIZE + selected % PAGE_SIZE
                )
                torch.testing.assert_close(
                    out[row, :count], expected_pages.int(), atol=0, rtol=0
                )
                torch.testing.assert_close(
                    expected[row, selected].sort().values,
                    expected[row, :length].topk(count).values.sort().values,
                    atol=1e-4,
                    rtol=1e-4,
                )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (9, 12),
    reason="Hopper or SM120 paged candidate indexer dispatch",
)
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("zero_queries", [False, True])
def test_paged_candidate_backends_replay(ratio, heads, zero_queries):
    from sglang.srt.layers.attention.dsv4.v41_indexer import scoring
    from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
        DenseBlocksBackend,
    )
    from sglang.srt.layers.attention.dsv4.v41_indexer.full_topk import FullTopKIndexer
    from sglang.srt.layers.attention.dsv4.v41_indexer.types import (
        DecodeInputs,
    )

    batch, width, capacity, topk = 12, 193, 32772, 64
    q, weights, slots, lens, table = _make_logits_case(batch, heads, width)
    if zero_queries:
        q.zero_()
    consumer_q = -q
    req = torch.arange(batch, device=q.device, dtype=torch.int32) // 6
    req_table = torch.full(
        (2, capacity * ratio), -1, device=q.device, dtype=torch.int32
    )
    req_table[:, : width * ratio : ratio] = (slots[::6] * ratio).int()
    positions = (lens * ratio - 1).clamp_min(0)
    pool = SimpleNamespace(get_index_k_with_scale_buffer=lambda layer: table)
    # A unit page keeps the per-row table valid under the shuffled slots below.
    page_table = torch.empty((batch, capacity), device=q.device, dtype=torch.int32)
    page_table.copy_(req_table[req.long(), ::ratio] // ratio)
    metadata = SimpleNamespace(
        max_compressed_seq_len=capacity,
        page_table=page_table,
        compressed_page_size=1,
    )
    indexer = SimpleNamespace(
        queries=lambda q, freqs: q, head_weights=lambda x: x, index_topk=topk
    )
    common = dict(
        indexer=indexer,
        layer_id=0,
        compress_ratio=ratio,
        freqs_cis=torch.zeros(width * ratio + 1, device=q.device),
        x=weights,
        positions=positions,
        req_rows=req,
        paged_metadata=metadata,
        is_verify=True,
    )
    kwargs = dict(token_to_kv_pool=pool, req_to_token=req_table)
    # Hopper FullTopK shares this paged Triton scorer. SM120 FullTopK is tested
    # separately with its real DeepGEMM planner, not forced onto a fallback here.
    full = (
        FullTopKIndexer(
            **kwargs, use_deep_gemm_prefill=False, use_deep_gemm_decode=False
        )
        if torch.cuda.get_device_capability()[0] == 9
        else None
    )
    candidates = DenseBlocksBackend(
        **kwargs,
        candidate_topk_blocks=4,
        candidate_block_size=8,
        use_deep_gemm_publish=False,
        use_deep_gemm_consume=False,
    )
    # DeepSelect fills a full top-k buffer of exactly live rows by topk, as at
    # runtime; the candidate paths also clear padded rows and columns.
    outputs = [
        SimpleNamespace(
            page_indices=torch.empty(shape, device=q.device, dtype=torch.int32),
            raw_indices=None,
        )
        for shape in (
            ([(batch, topk)] if full is not None else [])
            + [(batch + 1, topk + 8), (batch + 1, topk + 8)]
        )
    ]
    decode_inputs = [
        DecodeInputs(
            q_lora=q if i < len(outputs) - 1 else consumer_q,
            out_page_indices=out.page_indices,
            **common,
        )
        for i, out in enumerate(outputs)
    ]
    captured_scores = []
    captured_masks = []

    def record_scores(*args, **kwargs):
        result = fp4_index_logits_paged(*args, **kwargs)
        captured_scores.append(result)
        captured_masks.append(args[-1])
        return result

    def run():
        if full is not None:
            full.topk_decode(decode_inputs[0])
        published = candidates.publish_decode(decode_inputs[-2])
        candidates.consume_decode(decode_inputs[-1], published)
        return published

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with patch.object(scoring, "fp4_index_logits_paged", side_effect=record_scores):
        with torch.cuda.graph(graph):
            published = run()
    assert (
        len(captured_scores) == len(outputs)
    )  # every tested runtime entrypoint took the paged path
    mask = published.decode_mask
    assert captured_masks[:-1] == [None] * (len(outputs) - 1)
    assert captured_masks[-1] is mask
    for step, visible in enumerate(
        [193, 0, 1, 7, 8, 9, 63, 64, 65, 127, 128, 129, 193]
    ):
        # -1 positions exercise zero-length padded rows, also for ratio 1.
        lens.copy_((visible - torch.arange(batch, device=q.device) % 6).clamp_min(0))
        positions.copy_(lens * ratio - 1)
        req.copy_(req.roll(1))
        req_table[:, : width * ratio : ratio] = (slots[::6].roll(step, 1) * ratio).int()
        page_table.copy_(req_table[req.long(), ::ratio] // ratio)
        q.neg_()
        for scores in captured_scores:
            scores.fill_(torch.inf)
        for out in outputs:
            out.page_indices.fill_(12345)
            if out.raw_indices is not None:
                out.raw_indices.fill_(12345)
        graph.replay()
        current_slots = req_table[req.long(), : width * ratio : ratio].long() // ratio
        source_scores = _reference_logits(q, weights, current_slots, lens, table)
        consumer_scores = _reference_logits(
            consumer_q, weights, current_slots, lens, table
        )
        for row, length in enumerate(lens.tolist()):
            blocks = published.blocks[row]
            blocks = blocks[blocks >= 0].long()
            nblocks = (length + 7) // 8
            assert blocks.numel() == blocks.unique().numel() == min(4, nblocks)
            if nblocks:
                assert nblocks - 1 in blocks.tolist()
                maxima = torch.full((nblocks * 8,), -torch.inf, device=q.device)
                maxima[:length] = source_scores[row, :length]
                maxima = maxima.view(-1, 8).amax(-1)
                maxima[-1] = torch.inf
                torch.testing.assert_close(
                    maxima[blocks].sort().values,
                    maxima.topk(min(4, nblocks)).values.sort().values,
                )
            keep = torch.isin(torch.arange(width, device=q.device) // 8, blocks)
            torch.testing.assert_close(mask[row, :width], keep)
            consumer_scores[row].masked_fill_(~keep, -torch.inf)
            for out, scores in zip(
                outputs,
                (source_scores,) * (len(outputs) - 1) + (consumer_scores,),
            ):
                pages = out.page_indices[row]
                pages = pages[pages >= 0]
                # The shuffled request mapping is bijective within a row.
                selected = (current_slots[row, :, None] == pages[None, :]).nonzero()[
                    :, 0
                ]
                count = min(topk, int(torch.isfinite(scores[row]).sum()))
                assert pages.numel() == selected.unique().numel() == count
                torch.testing.assert_close(
                    scores[row, selected].sort().values,
                    scores[row].topk(count).values.sort().values,
                )
                expected_pages = torch.full_like(out.page_indices[row], -1)
                expected_pages[:count] = current_slots[
                    row, selected.sort().values
                ].int()
                torch.testing.assert_close(out.page_indices[row], expected_pages)
                if out.raw_indices is not None:
                    expected_raw = torch.full_like(out.raw_indices[row], -1)
                    expected_raw[:count] = selected.sort().values.int()
                    torch.testing.assert_close(out.raw_indices[row], expected_raw)
        for out in outputs[-2:]:
            assert (out.page_indices[batch] == -1).all().item()
            if out.raw_indices is not None:
                assert (out.raw_indices[batch] == -1).all().item()


@pytest.mark.parametrize(
    "bs,n,bpp",
    [(1024, 2049, 4), (1024, 2049, 2), (3, 17, 2), (0, 10, 4), (5, 9, 3)],
)
def test_expand_index_page_table_matches_cpu(bs, n, bpp):
    """The GPU expansion is bitwise the torch expression the CPU path runs; bpp 3
    stays on torch."""
    from sglang.srt.layers.attention.dsv4.metadata import expand_index_page_table

    def expand(t):
        return expand_index_page_table(
            t, full_page_size=64 * bpp, compress_ratio=1, index_page_size=64
        )

    torch.manual_seed(bs + n + bpp)
    t = torch.randint(0, 1 << 20, (2 * bs, n), device=get_device(), dtype=torch.int32)
    if bs and n >= 8:
        t[:, -3:] = -1
        t[0, :4] = t.new_tensor([2**31 - 1, 2**30, 2**29 + 7, -(2**31)])
    for table in (t[:bs], t[::2]):
        out = expand(table)
        assert out.dtype == torch.int32
        assert torch.equal(out.cpu(), expand(table.cpu()))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
