"""Check native Huginn equivalence, adaptive KV and differentiable stage loss."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import GenerationConfig

from ep_loop.config import load_config, smoke_config, validate_config
from ep_loop.data import filter_policy
from ep_loop.huginn import HuginnBackend, tiny_huginn
from ep_loop.inference import generate
from ep_loop.objective import DistillationObjective
from ep_loop.selection import make_stage_plan, make_stage_plan_from_js
from ep_loop.training import read_checkpoint_weights

ROOT = Path(__file__).resolve().parents[1]


class HuginnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(713)
        cls.backend = tiny_huginn(ROOT / "models/huginn-0125")
        cls.backend.model.eval()

    def test_each_coda_readout_matches_official_fixed_depth(self):
        b = self.backend
        ids = torch.tensor([[1, 20, 30, 40]])
        with torch.no_grad():
            torch.manual_seed(71)
            trace = b.trace(ids, 4)
            for depth in range(1, 5):
                torch.manual_seed(71)
                native = b.model(ids, num_steps=[0, depth], use_cache=False)
                torch.testing.assert_close(b.head(trace.states[depth - 1]), native.logits, rtol=0, atol=0)

    def test_generation_respects_native_end_turn_stopping_token(self):
        self.assertIn(65508, self.backend.stop_token_ids)
        # Inject a real Huginn end_turn ID into the tiny-vocabulary smoke model.
        # Generation must stop before attempting to embed another token.
        with patch("ep_loop.inference.sample_token", return_value=torch.tensor([65508])):
            response = generate(self.backend, [1, 20, 30], 2, 3, 1., 1., torch.Generator())
        self.assertTrue(response.reached_eos)
        self.assertEqual(response.token_ids, [65508])

    def test_native_kl_orientation_uniform_first_check_and_actual_stop(self):
        b = self.backend
        ids = torch.tensor([[1, 20, 30]])
        with torch.no_grad():
            torch.manual_seed(3)
            trace = b.trace(ids, 4, use_cache=True, endpoint="native", exit_threshold=1e9)
            self.assertEqual(trace.exits.tolist(), [[1, 1, 1]])
            self.assertEqual(trace.executed_token_loops, 3)
            self.assertTrue(torch.count_nonzero(trace.states[-1]) == 0)
            # Now deepen the next token beyond previous exits: this requires
            # latest-m4 lookup and cannot work with the student's shallow KV.
            next_trace = b.trace(torch.tensor([[40]]), 4, trace.cache, True, "native", 1e-12)
            self.assertEqual(next_trace.exits.item(), 4)
            self.assertEqual(next_trace.cache.get_seq_length(), 4)
            self.assertEqual(next_trace.executed_token_loops, 4)
        evaluator = b.module.KLExitEvaluator(b.model, 5e-4)
        evaluator.init(torch.zeros(1, 1, 32))
        q1 = torch.randn(1, 1, 260)
        q2 = torch.randn(1, 1, 260)
        class Output:
            def __init__(self, logits): self.logits = logits
        with patch.object(b.model, "predict_from_latents", side_effect=[Output(q1), Output(q2)]):
            _, _, first = evaluator.check(b.model, None, {})
            _, _, second = evaluator.check(b.model, None, {})
        uniform = torch.full((1, 260), 1 / 260.)
        torch.testing.assert_close(first, (uniform * (uniform.log() - q1[:, -1].log_softmax(-1))).sum(-1))
        previous, current = q1[:, -1].log_softmax(-1), q2[:, -1].log_softmax(-1)
        torch.testing.assert_close(second, (previous.exp() * (previous - current)).sum(-1))

    def test_teacher_prefix_trace_matches_native_adaptive_generation(self):
        b = self.backend
        ids = torch.tensor([[1, 20, 30, 40, 50]])
        native_evaluator = b.module.KLExitEvaluator(b.model, .04)
        recorded, current = [], []
        original_init, original_check = native_evaluator.init, native_evaluator.check
        def init(latents):
            nonlocal current
            current = []
            recorded.append(current)
            return original_init(latents)
        def check(model, latents, aux):
            out = original_check(model, latents, aux)
            current.append((out[1].logits[:, -1].clone(), out[1].latent_states[:, -1].clone()))
            return out
        native_evaluator.init, native_evaluator.check = init, check
        generation = GenerationConfig(max_new_tokens=3, do_sample=False, eos_token_id=2,
                                      bos_token_id=1, pad_token_id=0, return_dict_in_generate=True)
        with torch.no_grad(), patch.object(b.model, "_sample_next_token", side_effect=[
            torch.tensor([[40]]), torch.tensor([[50]]), torch.tensor([[60]])]):
            torch.manual_seed(87)
            native = b.model.generate_with_adaptive_compute(
                ids[:, :3], generation, max_new_tokens=3, num_steps=4,
                cache_lookup_strategy="latest-m4", exit_evaluator=native_evaluator)
            torch.manual_seed(87)
            trace = b.response_trace(ids, 4, 2, "native", .04)
            self.assertEqual(trace.exits.tolist(), [len(x) for x in recorded])
            self.assertEqual(trace.exits.tolist(), [x[0][0] for x in native.scores])
            for pos, stages in enumerate(recorded):
                for depth, (logits, head_state) in enumerate(stages):
                    torch.testing.assert_close(trace.states[depth][pos:pos + 1], head_state, rtol=0, atol=0)
                    # Full-prompt versus one-row LM-head GEMMs can round differently.
                    torch.testing.assert_close(b.head(trace.states[depth][pos:pos + 1]), logits, rtol=2e-5, atol=5e-7)
        self.assertEqual(trace.executed_token_loops,
                         3 * len(recorded[0]) + sum(len(x) for x in recorded[1:]))

    def test_reused_native_js_produces_same_stage_plan(self):
        b = self.backend
        with torch.no_grad():
            trace = b.response_trace(torch.tensor([[1, 20, 30, 40, 50]]), 4, 2, "native", 1e-12)
            logq = b.head(torch.stack(trace.states)).log_softmax(-1)
        first = make_stage_plan(logq, trace.exits, 2, "js", 0.)
        reused = make_stage_plan_from_js(trace.adjacent_js, trace.exits, 2, "js", 0.)
        torch.testing.assert_close(first.adjacent_js, reused.adjacent_js, rtol=1e-4, atol=1e-7)
        self.assertEqual(first.teacher_depths.tolist(), reused.teacher_depths.tolist())

    def test_chunk_backbone_checkpoint_gradient_equivalence_and_teacher_isolation(self):
        algorithm = smoke_config(load_config(ROOT / "configs/huginn_d_native_js.json"))["algorithm"]
        algorithm.update(exit_threshold=1e-12, js_threshold=0.)
        frozen = HuginnBackend(copy.deepcopy(self.backend.model).requires_grad_(False), "none")
        students = [HuginnBackend(copy.deepcopy(self.backend.model), "none") for _ in range(2)]
        losses = []
        for index, student in enumerate(students):
            student.model.train()
            student.model.gradient_checkpointing = index == 0
            objective = DistillationObjective(student, frozen, algorithm, 2 if index == 0 else 128, index == 0)
            torch.manual_seed(911)
            loss, info = objective(torch.tensor([[1, 20, 30, 40, 50, 60, 2]]), 3)
            loss.backward()
            losses.append(loss.detach())
            self.assertGreater(info["teacher_executed_token_loops"], 0)
            for name in ("prelude", "core_block", "coda"):
                grad = student.model.transformer[name][0].attn.Wqkv.weight.grad
                self.assertTrue(torch.isfinite(grad).all(), name)
                self.assertGreater(float(grad.norm()), 0, name)
        torch.testing.assert_close(*losses, rtol=1e-5, atol=1e-7)
        for (name, a), (_, b) in zip(students[0].model.named_parameters(), students[1].model.named_parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=4e-4, atol=1e-6, msg=name)
        self.assertTrue(all(p.grad is None for p in frozen.model.parameters()))

    def test_saved_tied_weights_reload_without_silently_accepting_missing_weights(self):
        student = HuginnBackend(copy.deepcopy(self.backend.model), "none")
        with tempfile.TemporaryDirectory() as folder:
            student.model.save_pretrained(folder)
            restored = HuginnBackend(copy.deepcopy(student.model), "none")
            with torch.no_grad():
                restored.model.transformer.wte.weight.add_(1)
            read_checkpoint_weights(restored.model, folder)
            for key, value in student.model.state_dict().items():
                torch.testing.assert_close(value, restored.model.state_dict()[key], rtol=0, atol=0)
            from safetensors.torch import load_file, save_file
            path = Path(folder) / "model.safetensors"
            weights = load_file(path)
            del weights["transformer.adapter.weight"]
            save_file(weights, path)
            with self.assertRaises(RuntimeError):
                read_checkpoint_weights(restored.model, folder)

    def test_family_configs_context_and_filter_identity(self):
        configs = [p for p in (ROOT / "configs").glob("huginn_[abcd]_*.json") if "dynamic" not in p.name]
        self.assertEqual(len(configs), 9)
        for path in configs:
            config = validate_config(load_config(path))
            self.assertEqual(config["algorithm"]["teacher_max_loops"], 32)
            self.assertIn(config["algorithm"]["student_loops"], [8, 16])
            self.assertEqual(config["data"]["max_prompt_tokens"] + config["rollout"]["max_new_tokens"], 4096)
            self.assertEqual(filter_policy(config)["cache_lookup_strategy"], "latest-m4")
            self.assertEqual(filter_policy(config)["exit_threshold"], 5e-4)

    def test_sharded_tied_checkpoint_resume_matches_all_parameters(self):
        student = HuginnBackend(copy.deepcopy(self.backend.model), "none")
        with tempfile.TemporaryDirectory() as folder:
            student.model.save_pretrained(folder, max_shard_size="40KB")
            self.assertTrue((Path(folder) / "model.safetensors.index.json").is_file())
            restored = HuginnBackend(copy.deepcopy(student.model), "none")
            with torch.no_grad():
                for parameter in restored.model.parameters(): parameter.add_(.25)
            read_checkpoint_weights(restored.model, folder)
            for key, value in student.model.state_dict().items():
                torch.testing.assert_close(value, restored.model.state_dict()[key], rtol=0, atol=0)

    def test_reference_32_loop_teacher_supports_8_and_16_loop_students(self):
        ids = torch.tensor([[1, 20, 30, 40]])
        for budget in (8, 16):
            backend = HuginnBackend(copy.deepcopy(self.backend.model), "none")
            backend.model.train()
            algorithm = load_config(ROOT / "configs/huginn_d_current_native_js.json")["algorithm"]
            algorithm.update(student_loops=budget, teacher_endpoint="fixed", js_threshold=0.)
            loss, info = DistillationObjective(backend, backend, algorithm, 2, True)(ids, 3)
            loss.backward()
            self.assertEqual(info["teacher_executed_token_loops"], 3 * 32)
            self.assertEqual(info["student_executed_token_loops"], 3 * budget)
            self.assertEqual(int(info["stage_hist"][budget]), 1)
            self.assertEqual(int(info["mapping_hist"][-1, 31]), 1)
            self.assertGreater(float(backend.model.transformer.core_block[0].attn.Wqkv.weight.grad.norm()), 0)


if __name__ == "__main__":
    unittest.main()
