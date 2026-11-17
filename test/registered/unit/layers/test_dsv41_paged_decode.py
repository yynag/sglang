"""CPU routing contracts; the real kernels are covered by test_fp4_indexer."""

import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsv4.v41_indexer import scoring
from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import DenseBlocksBackend
from sglang.srt.layers.attention.dsv4.v41_indexer.types import DecodeInputs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestPagedDecodeRouting(CustomTestCase):
    @staticmethod
    def _inputs(ratio, rows=2):
        return DecodeInputs(
            indexer=SimpleNamespace(
                queries=lambda q, _: q, head_weights=lambda x: x, index_topk=64
            ),
            layer_id=0,
            compress_ratio=ratio,
            freqs_cis=torch.zeros(256 * ratio),
            x=torch.ones(rows, 32, dtype=torch.bfloat16),
            q_lora=torch.ones(rows, 32, 128, dtype=torch.bfloat16),
            positions=torch.tensor([0, 129 * ratio - 1][:rows]),
            req_rows=torch.arange(rows, dtype=torch.int32),
            paged_metadata=SimpleNamespace(max_compressed_seq_len=192),
            is_verify=True,
            out_page_indices=torch.full((rows + 1, 72), 12345, dtype=torch.int32),
        )

    def test_scorer_selection_keeps_deep_gemm_architecture_separate(self):
        for major, ratio, allow_paged, plan_topk in product(
            (9, 10, 12), (1, 2), (False, True), (False, True)
        ):
            with self.subTest(
                major=major, ratio=ratio, allow_paged=allow_paged, plan=plan_topk
            ):
                inputs = self._inputs(ratio)
                req_table = torch.arange(2 * 192 * ratio).reshape(2, -1).int()
                table = torch.zeros(8, 64 * 68, dtype=torch.uint8)
                pool = SimpleNamespace(get_index_k_with_scale_buffer=lambda _: table)
                scores = torch.zeros(2, 192)
                plan = torch.zeros(3, 2, dtype=torch.int32)
                paged = allow_paged and major in (9, 12)
                with (
                    patch.object(
                        scoring,
                        "get_platform",
                        return_value=SimpleNamespace(
                            is_sm90=major == 9, is_sm120=major == 12
                        ),
                    ),
                    patch.object(scoring, "plan_topk_v2", return_value=plan) as planner,
                    patch.object(
                        scoring, "fp4_index_logits_paged", return_value=scores
                    ) as paged_kernel,
                    patch.object(
                        scoring, "fp4_index_logits_decode", return_value=scores
                    ) as dense_kernel,
                ):
                    result = scoring.decode_scores(
                        inputs=inputs,
                        token_to_kv_pool=pool,
                        req_to_token=req_table,
                        allow_paged=allow_paged,
                        compute_topk_plan=plan_topk,
                    )
                self.assertIs(result.scores, scores)
                self.assertIsInstance(
                    result, scoring.PagedDecodeScores if paged else scoring.DecodeScores
                )
                if paged:
                    paged_kernel.assert_called_once()
                    dense_kernel.assert_not_called()
                    self.assertIs(paged_kernel.call_args.args[3], req_table)
                    self.assertIs(result.plan, plan if plan_topk else None)
                    self.assertEqual(planner.call_count, int(plan_topk))
                else:
                    dense_kernel.assert_called_once()
                    paged_kernel.assert_not_called()
                    planner.assert_not_called()

    def test_empty_batch_clears_padding_without_scoring(self):
        inputs = self._inputs(1, rows=0)
        with (
            patch.object(
                scoring,
                "get_platform",
                return_value=SimpleNamespace(is_sm90=False, is_sm120=True),
            ),
            patch.object(scoring, "fp4_index_logits_paged") as kernel,
        ):
            self.assertIsNone(
                scoring.decode_scores(
                    inputs=inputs,
                    token_to_kv_pool=None,
                    req_to_token=torch.empty(0, 192, dtype=torch.int32),
                )
            )
        kernel.assert_not_called()
        self.assertTrue(bool((inputs.out_page_indices == -1).all()))

    def test_non_eight_candidate_blocks_keep_the_dense_scorer(self):
        for block_size in (8, 16):
            backend = DenseBlocksBackend(
                token_to_kv_pool=None,
                req_to_token=torch.empty(0),
                candidate_topk_blocks=4,
                candidate_block_size=block_size,
                use_deep_gemm_publish=False,
                use_deep_gemm_consume=False,
            )
            with patch(
                "sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks"
                ".decode_scores",
                return_value=None,
            ) as score:
                backend.publish_decode(None)
                backend.consume_decode(None, None)
            self.assertEqual(len(score.call_args_list), 2)
            for call in score.call_args_list:
                self.assertEqual(call.kwargs["allow_paged"], block_size == 8)


if __name__ == "__main__":
    unittest.main()
