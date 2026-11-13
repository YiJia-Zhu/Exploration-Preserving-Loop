"""Check objective math, tail mass, and teacher isolation without loading Ouro."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from ep_loop.runtime import LoRALinear, aggregate_log_probs, preserve_kl, reverse_kl, teacher_mode


class ObjectiveTests(unittest.TestCase):
    def test_reverse_kl_gradient_and_teacher_detach(self):
        s = torch.tensor([[0.2,-0.8,1.4]],dtype=torch.float64,requires_grad=True)
        t = torch.tensor([[1.0,0.3,-0.4]],dtype=torch.float64,requires_grad=True)
        loss = reverse_kl(s,t)
        loss.backward()
        p, logp, logq = s.detach().softmax(-1), s.detach().log_softmax(-1), t.detach().log_softmax(-1)
        expected = p * (logp-logq-loss.detach())
        torch.testing.assert_close(s.grad,expected,atol=1e-12,rtol=1e-12)
        self.assertIsNone(t.grad)

    def test_preservation_keeps_absolute_candidate_mass(self):
        # Same ratio among candidates, different total mass must incur a loss.
        s = torch.tensor([[0.01,0.02,0.97]],dtype=torch.float64).log().requires_grad_()
        t = torch.tensor([[0.2,0.4,0.4]],dtype=torch.float64).log().requires_grad_()
        agg = aggregate_log_probs(s,[0,1]).exp()
        torch.testing.assert_close(agg.sum(-1),torch.ones(1,dtype=torch.float64))
        self.assertGreater(float(preserve_kl(s,t,[0,1])),0.5)
        preserve_kl(s,t,[0,1]).backward()
        self.assertLess(float(s.grad[0,0]),0)
        self.assertIsNone(t.grad)

    def test_zero_loss_when_equal(self):
        x = torch.tensor([[1.0,-1.0,0.5]],dtype=torch.float64)
        self.assertAlmostEqual(float(reverse_kl(x,x)),0.0,places=12)
        self.assertAlmostEqual(float(preserve_kl(x,x,[0])),0.0,places=12)

    def test_lora_teacher_is_fixed_and_context_restores(self):
        base = torch.nn.Linear(4,3,bias=False).requires_grad_(False)
        layer = LoRALinear(base,rank=2)
        x = torch.randn(2,4)
        initial = base(x).detach().clone()
        torch.testing.assert_close(layer(x),initial,rtol=0,atol=0)
        with torch.no_grad():
            layer.b.add_(0.5)
        self.assertGreater(float((layer(x)-initial).abs().max()),0)
        with teacher_mode(layer):
            torch.testing.assert_close(layer(x),initial,rtol=0,atol=0)
        self.assertTrue(layer.enabled)


if __name__ == "__main__":
    unittest.main()
