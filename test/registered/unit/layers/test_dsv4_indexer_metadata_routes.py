"""The paged scorer, not the C4 FP4 switch, owns its metadata contract."""

import sys
import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata
from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import DenseBlocksBackend
from sglang.srt.layers.attention.dsv4.v41_indexer.full_topk import FullTopKIndexer
from sglang.srt.layers.attention.dsv4.v41_indexer.sparse_table import SparseTableBackend
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.runner_utils import capture_mode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_BACKEND = "sglang.srt.layers.attention.deepseek_v4_backend"
_JIT_METADATA = "sglang.kernels.ops.attention.dsv4.get_paged_mqa_logits_metadata"


class TestIndexerMetadataRoutes(CustomTestCase):
    @staticmethod
    def _backend(major, candidate_source, fp4=False, stage=range(5)):
        backend = object.__new__(DeepseekV4AttnBackend)
        backend.page_size = 256
        backend.token_to_kv_pool = SimpleNamespace(
            stage_layer_ids=tuple(stage), get_index_k_page_size=lambda ratio: 64
        )
        backend.full_topk_indexer = FullTopKIndexer(
            token_to_kv_pool=backend.token_to_kv_pool,
            req_to_token=torch.empty(0),
            use_deep_gemm_prefill=False,
            use_deep_gemm_decode=major >= 10,
        )
        candidate_cls = (
            SparseTableBackend
            if major == 10 and candidate_source >= 0
            else DenseBlocksBackend
        )
        backend.decode_candidates = object.__new__(candidate_cls)
        cfg = SimpleNamespace(
            compress_ratios=[4, 1, 2, 1, 2],
            index_source_layer_ids=[1, 2, 3, 4],
            candidate_source_layer_id=candidate_source,
        )
        backend._deep_gemm_decode_ratios = backend._get_deep_gemm_decode_ratios(cfg)
        backend.enable_deepseek_v4_fp4_indexer = fp4
        backend.dsa_topk_backend = SimpleNamespace(should_use_topk_v2=lambda: False)
        return backend

    @staticmethod
    def _core():
        return SimpleNamespace(
            page_table=torch.tensor([[3, 1, 2]], dtype=torch.int32),
            c1_topk_lengths_clamp1=torch.tensor([257], dtype=torch.int32),
            c2_topk_lengths_clamp1=torch.tensor([129], dtype=torch.int32),
            c4_topk_lengths_raw=torch.tensor([65], dtype=torch.int32),
        )

    def test_consumers_select_the_planner_across_flags_and_graph_variants(self):
        native_plan = torch.tensor([[1, 0]], dtype=torch.int32)
        jit_plan = torch.tensor([[2, 0]], dtype=torch.int32)
        native = MagicMock(return_value=native_plan)
        jit = MagicMock(return_value=jit_plan)
        deep_gemm = SimpleNamespace(
            get_num_sms=lambda: 1, get_paged_mqa_logits_metadata=native
        )
        variants = (
            None,
            "candidate_filtered",
            "candidate_unfiltered",
            "candidate_c2_all",
            "candidate_all",
        )
        for major, ratio, fp4, use_jit, torch_fp8, variant in product(
            (9, 10, 12),
            (1, 2, 4),
            (False, True),
            (False, True),
            (False, True),
            variants,
        ):
            with self.subTest(
                major=major,
                ratio=ratio,
                fp4=fp4,
                jit=use_jit,
                torch_fp8=torch_fp8,
                variant=variant,
            ):
                # Layer 1 is a full c1 indexer; layer 2 publishes candidates.
                backend = self._backend(major, candidate_source=2, fp4=fp4)
                skipped = variant == "candidate_all" or (
                    variant == "candidate_c2_all" and ratio == 2
                )
                if ratio == 4:
                    needs_plan = fp4 or not torch_fp8
                elif skipped:
                    needs_plan = False
                else:
                    needs_plan = major >= 10 and (
                        ratio == 1
                        or major == 10
                        or variant in (
                            "candidate_unfiltered",
                            "candidate_c2_all",
                        )
                    )
                native.reset_mock()
                jit.reset_mock()
                with (
                    patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
                    patch(f"{_BACKEND}._is_cuda", True),
                    patch(
                        f"{_BACKEND}.get_platform",
                        return_value=SimpleNamespace(is_sm120=major == 12),
                    ),
                    patch(_JIT_METADATA, jit),
                    envs.SGLANG_OPT_USE_JIT_INDEXER_METADATA.override(use_jit),
                    envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(torch_fp8),
                    envs.SGLANG_OPT_USE_TILELANG_INDEXER.override(False),
                    patch.object(capture_mode, "_capture_attention_variant", variant),
                ):
                    metadata = backend.init_forward_metadata_indexer(
                        self._core(), compress_ratio=ratio
                    )
                self.assertEqual(metadata.use_deep_gemm_metadata, needs_plan)
                self.assertEqual(
                    metadata.force_deep_gemm_metadata, needs_plan and major == 12
                )
                if not needs_plan:
                    self.assertIsNone(metadata.deep_gemm_metadata)
                    native.assert_not_called()
                    jit.assert_not_called()
                else:
                    expected = (
                        native_plan if major == 12 or not use_jit else jit_plan
                    )
                    self.assertIs(metadata.deep_gemm_metadata, expected)
                    called, unused = (
                        (native, jit) if expected is native_plan else (jit, native)
                    )
                    called.assert_called_once()
                    unused.assert_not_called()
                    self.assertEqual(called.call_args.args[1:], (64, 1))
                self.assertEqual(metadata.max_compressed_seq_len, 768 // ratio)

    def test_only_local_layers_and_actual_publishers_need_the_dense_schedule(self):
        for major in (9, 10, 12):
            for source, stage, expected in (
                (-1, range(5), {1, 2} if major >= 10 else set()),
                (1, range(5), {1} if major == 10 else set()),
                (
                    2,
                    range(5),
                    ({1, 2} if major == 10 else {1}) if major >= 10 else set(),
                ),
                (2, range(3, 5), set()),
            ):
                with self.subTest(major=major, source=source, stage=stage):
                    backend = self._backend(major, source, stage=stage)
                    self.assertEqual(backend._deep_gemm_decode_ratios, expected)

    def test_tilelang_c4_and_filtered_dense_blocks_do_not_import_deep_gemm(self):
        backend = self._backend(12, candidate_source=1)
        with (
            patch.dict(sys.modules, {"deep_gemm": None}),
            patch(f"{_BACKEND}._is_cuda", True),
            patch(
                f"{_BACKEND}.get_platform", return_value=SimpleNamespace(is_sm120=True)
            ),
            envs.SGLANG_OPT_USE_TILELANG_INDEXER.override(True),
            patch.object(
                capture_mode, "_capture_attention_variant", "candidate_filtered"
            ),
        ):
            for ratio in (1, 2, 4):
                metadata = backend.init_forward_metadata_indexer(
                    self._core(), compress_ratio=ratio
                )
                self.assertIsNone(metadata.deep_gemm_metadata)

    def test_metadata_routes_match_the_forward_dispatch(self):
        for major, source, variant, mode in product(
            (9, 10, 12),
            (-1, 1, 2),
            (None, "candidate_unfiltered", "candidate_c2_all", "candidate_all"),
            (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY),
        ):
            with (
                self.subTest(major=major, source=source, variant=variant, mode=mode),
                patch.object(capture_mode, "_capture_attention_variant", variant),
            ):
                backend = self._backend(major, source)
                backend.forward_metadata = SimpleNamespace(candidate_metadata=None)
                backend.tail_forward_metadata = None
                backend._make_low_ratio_decode_indexer_inputs = MagicMock()
                full = backend.full_topk_indexer.topk_decode = MagicMock()
                publish = backend.decode_candidates.publish_decode = MagicMock()
                consume = backend.decode_candidates.consume_decode = MagicMock()
                consumers = set()
                for layer_id, ratio in enumerate((4, 1, 2, 1, 2)):
                    if layer_id == 0:
                        continue
                    full.reset_mock()
                    publish.reset_mock()
                    consume.reset_mock()
                    layer = SimpleNamespace(
                        compress_ratio=ratio,
                        indexer=SimpleNamespace(
                            is_candidate_source=layer_id == source,
                            uses_candidates=0 <= source < layer_id,
                        ),
                    )
                    backend._low_ratio_index_topk(
                        layer,
                        None,
                        None,
                        None,
                        None,
                        SimpleNamespace(forward_mode=mode),
                    )
                    if (full.called and major >= 10) or (
                        publish.called
                        and isinstance(backend.decode_candidates, SparseTableBackend)
                    ):
                        consumers.add(ratio)
                for ratio in (1, 2):
                    self.assertEqual(
                        backend._uses_deep_gemm_indexer_metadata(ratio),
                        ratio in consumers,
                    )

    def test_native_plans_chunk_decode_and_grouped_verify_at_the_sm120_cap(self):
        for ratio, group, rows in product((1, 2), (1, 8), (4096, 4104)):
            native = MagicMock(
                side_effect=lambda *args: torch.zeros((2, 2), dtype=torch.int32)
            )
            deep_gemm = SimpleNamespace(
                get_num_sms=lambda: 1, get_paged_mqa_logits_metadata=native
            )
            backend = self._backend(12, candidate_source=-1, fp4=False)
            lengths = torch.full((rows,), 129, dtype=torch.int32)
            core = SimpleNamespace(
                page_table=torch.ones((rows, 2), dtype=torch.int32),
                c1_topk_lengths_clamp1=lengths,
                c2_topk_lengths_clamp1=lengths,
            )
            with (
                self.subTest(ratio=ratio, group=group, rows=rows),
                patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
                patch(
                    f"{_BACKEND}.get_platform",
                    return_value=SimpleNamespace(is_sm120=True),
                ),
                patch(
                    "sglang.srt.layers.attention.dsv4.metadata._IS_SM120", True
                ),
                patch(f"{_BACKEND}.deep_gemm_paged_varlen", return_value=False),
                patch(_JIT_METADATA) as jit,
                envs.SGLANG_OPT_USE_JIT_INDEXER_METADATA.override(True),
                envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(True),
                patch.object(capture_mode, "_capture_attention_variant", None),
            ):
                metadata = backend.init_forward_metadata_indexer(
                    core,
                    compress_ratio=ratio,
                    rows_per_request=group,
                    use_prefill_cuda_graph=True,
                )
            jit.assert_not_called()
            chunks = metadata.row_chunks()
            self.assertEqual(
                [(s.start, s.stop) for s, _ in chunks],
                [(0, 4096)] + ([(4096, rows)] if rows > 4096 else []),
            )
            self.assertEqual(native.call_count, len(chunks))
            for (row_slice, plan), call in zip(chunks, native.call_args_list):
                self.assertIsInstance(plan, torch.Tensor)
                torch.testing.assert_close(
                    call.args[0], lengths[row_slice].view(-1, group)
                )
                self.assertEqual(call.args[1:], (64, 1))

    def test_early_verify_shares_the_unfiltered_schedule_with_filtered_capture(self):
        for ratio, rows, topk in product((1, 2), (8, 4104), (False, True)):
            backend = self._backend(12, candidate_source=1)
            backend.enable_early_verify_metadata()
            backend.dsa_topk_backend.should_use_topk_v2 = lambda: topk
            native = MagicMock(side_effect=self._cpu_schedule)
            deep_gemm = SimpleNamespace(
                get_num_sms=lambda: 1, get_paged_mqa_logits_metadata=native
            )
            with (
                self.subTest(ratio=ratio, rows=rows, topk=topk),
                patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
                patch(
                    f"{_BACKEND}.get_platform",
                    return_value=SimpleNamespace(is_sm120=True),
                ),
                patch("sglang.srt.layers.attention.dsv4.metadata._IS_SM120", True),
                patch(
                    "sglang.kernels.ops.attention.dsv4.plan_topk_v2",
                    side_effect=self._cpu_topk_plan,
                ),
                envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(True),
            ):
                metadata = []
                for variant, value in (
                    ("candidate_unfiltered", 127),
                    ("candidate_filtered", 129),
                ):
                    lengths = torch.full((rows,), value, dtype=torch.int32)
                    core = SimpleNamespace(
                        page_table=torch.full((rows, 3), 1, dtype=torch.int32),
                        c1_topk_lengths_clamp1=lengths,
                        c2_topk_lengths_clamp1=lengths,
                    )
                    with patch.object(
                        capture_mode, "_capture_attention_variant", variant
                    ):
                        metadata.append(
                            backend.init_forward_metadata_indexer(
                                core,
                                compress_ratio=ratio,
                                rows_per_request=8,
                                use_prefill_cuda_graph=True,
                            )
                        )
                captured, replay = metadata
                self.assertTrue(captured.force_deep_gemm_metadata)
                self.assertTrue(replay.force_deep_gemm_metadata)
                if rows > 4096:
                    self.assertEqual(
                        [(s.start, s.stop) for s, _ in captured.row_chunks()],
                        [(0, 4096), (4096, 4104)],
                    )
                self._assert_graph_copy(captured, replay)
            self.assertEqual(native.call_count, 4 if rows > 4096 else 2)

    @staticmethod
    def _cpu_schedule(lens, *_):
        # Stub only the GPU planner; exercise the real metadata construction/copy.
        return lens.new_full((2, 2), int(lens.flatten()[0]))

    @staticmethod
    def _cpu_topk_plan(lens):
        return torch.cat((lens.new_zeros((1, 2)), lens[:, None].expand(-1, 2)))

    def _assert_graph_copy(self, captured, replay):
        fields = (
            "page_table",
            "compressed_seq_lens",
            "deep_gemm_metadata",
            "deep_gemm_indices",
            "topk_metadata",
            "topk_metadata_chunks",
        )
        owners = {name: getattr(captured, name) for name in fields}
        children = {
            name: tuple(owner)
            for name, owner in owners.items()
            if isinstance(owner, list)
        }
        tensors = []
        for name, owner in owners.items():
            source = getattr(replay, name)
            if owner is None:
                self.assertIsNone(source)
                continue
            pairs = zip(owner, source) if isinstance(owner, list) else [(owner, source)]
            tensors.extend((dst, dst.data_ptr(), src) for dst, src in pairs)
        captured.copy_(replay)
        for name, owner in owners.items():
            self.assertIs(getattr(captured, name), owner)
            if isinstance(owner, list):
                for actual, original in zip(owner, children[name]):
                    self.assertIs(actual, original)
        for tensor, ptr, expected in tensors:
            self.assertEqual(tensor.data_ptr(), ptr)
            torch.testing.assert_close(tensor, expected)

    def _graph_metadata(self, value, *, use_deep_gemm=True, rows=4104, varlen=False):
        deep_gemm = SimpleNamespace(
            get_num_sms=lambda: 1,
            get_paged_mqa_logits_metadata=self._cpu_schedule,
        )
        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch("sglang.srt.layers.attention.dsv4.metadata._IS_SM120", not varlen),
            patch(
                "sglang.kernels.ops.attention.dsv4.plan_topk_v2",
                side_effect=self._cpu_topk_plan,
            ),
        ):
            return PagedIndexerMetadata(
                page_size=256,
                compressed_page_size=64,
                page_table=torch.full((rows, 3), value, dtype=torch.int32),
                compressed_seq_lens=torch.full((rows,), value, dtype=torch.int32),
                use_topk_v2=True,
                use_deep_gemm_metadata=use_deep_gemm,
                force_deep_gemm_metadata=use_deep_gemm,
                use_prefill_cuda_graph=True,
                compress_ratio=1,
                rows_per_request=8,
                deep_gemm_varlen=varlen,
            )

    def test_graph_copy_without_deep_gemm_keeps_tensor_owners(self):
        captured = self._graph_metadata(127, use_deep_gemm=False)
        replay = self._graph_metadata(129, use_deep_gemm=False)
        self.assertIsNone(captured.deep_gemm_metadata)
        self.assertIsNone(captured.topk_metadata_chunks)
        self._assert_graph_copy(captured, replay)

    def test_graph_copy_keeps_varlen_indices(self):
        captured = self._graph_metadata(127, rows=128, varlen=True)
        replay = self._graph_metadata(129, rows=128, varlen=True)
        self.assertIsNotNone(captured.deep_gemm_indices)
        self._assert_graph_copy(captured, replay)

    def test_graph_copy_rejects_layout_changes_before_writing(self):
        for change in (
            "rows_per_chunk", "chunk_count", "schedule_shape", "topk_shape",
            "dtype", "stride", "device", "layout", "optional", "container",
            "page_table", "graph_mode", "group",
        ):
            with self.subTest(change=change):
                captured, replay = self._graph_metadata(127), self._graph_metadata(129)
                if change == "rows_per_chunk":
                    replay.rows_per_chunk = 2048
                elif change == "chunk_count":
                    replay.deep_gemm_metadata.pop()
                elif change == "schedule_shape":
                    replay.deep_gemm_metadata[1] = replay.deep_gemm_metadata[1][:1]
                elif change == "topk_shape":
                    replay.topk_metadata_chunks[1] = replay.topk_metadata_chunks[1][:1]
                elif change == "dtype":
                    replay.deep_gemm_metadata[1] = replay.deep_gemm_metadata[1].long()
                elif change == "stride":
                    replay.deep_gemm_metadata[1] = replay.deep_gemm_metadata[1].t()
                elif change == "device":
                    replay.deep_gemm_metadata[1] = (
                        replay.deep_gemm_metadata[1].to("meta")
                    )
                elif change == "layout":
                    replay.deep_gemm_metadata[1] = (
                        replay.deep_gemm_metadata[1].to_sparse()
                    )
                elif change == "optional":
                    replay.topk_metadata_chunks = None
                elif change == "container":
                    replay.deep_gemm_metadata = tuple(replay.deep_gemm_metadata)
                elif change == "page_table":
                    replay.page_table = replay.page_table[:1]
                elif change == "group":
                    replay.rows_per_request = 1
                else:
                    replay.use_prefill_cuda_graph = False
                tensors = (
                    captured.page_table,
                    captured.compressed_seq_lens,
                    *captured.deep_gemm_metadata,
                    *captured.topk_metadata_chunks,
                )
                before = [t.clone() for t in tensors]
                with self.assertRaisesRegex(
                    ValueError, "graph-backed indexer metadata"
                ):
                    captured.copy_(replay)
                for actual, expected in zip(tensors, before):
                    torch.testing.assert_close(actual, expected)

    def test_eager_copy_can_replace_dynamic_chunk_layout(self):
        captured, replay = self._graph_metadata(127), self._graph_metadata(129)
        captured.use_prefill_cuda_graph = replay.use_prefill_cuda_graph = False
        replay.rows_per_chunk = 2048
        replay.deep_gemm_metadata = [
            torch.full((2, 2), 7, dtype=torch.int32) for _ in range(3)
        ]
        replay.topk_metadata_chunks = [
            torch.zeros(n + 1, 2, dtype=torch.int32) for n in (2048, 2048, 8)
        ]
        captured.copy_(replay)
        self.assertIs(captured.deep_gemm_metadata, replay.deep_gemm_metadata)
        self.assertIs(captured.topk_metadata_chunks, replay.topk_metadata_chunks)
        self.assertEqual(captured.rows_per_chunk, 2048)

    def test_copy_keeps_a_schedule_free_capture_schedule_free(self):
        with patch.dict(sys.modules, {"deep_gemm": None}):
            metadata = [
                PagedIndexerMetadata(
                    page_size=256,
                    compressed_page_size=64,
                    page_table=torch.full((1, 2), value, dtype=torch.int32),
                    compressed_seq_lens=torch.tensor([value], dtype=torch.int32),
                    use_topk_v2=False,
                    use_deep_gemm_metadata=False,
                    compress_ratio=1,
                )
                for value in (1, 2)
            ]
        destination, source = metadata
        ptr = destination.page_table.data_ptr()
        destination.copy_(source)
        self.assertEqual(destination.page_table.data_ptr(), ptr)
        self.assertIsNone(destination.deep_gemm_metadata)
        torch.testing.assert_close(destination.page_table, source.page_table)


if __name__ == "__main__":
    unittest.main()
