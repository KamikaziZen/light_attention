import torch as T
import torch.nn.functional as F

from unittest import TestCase

from light_attention.nn.functional.drop_matmul import drop_matmul


class TestDropMatmul(TestCase):

    def setUp(self):
        T.random.manual_seed(42)
        self.prob = 0.66
        self.lhs = T.randn((3, 4))
        self.rhs = T.randn((4, 5))

    def test_forward(self):
        T.random.manual_seed(42)
        ys_base = F.dropout(self.lhs, self.prob) @ self.rhs

        T.random.manual_seed(42)
        ys_curr = drop_matmul(self.lhs, self.rhs, self.prob)

        self.assertTrue(T.allclose(ys_base, ys_curr))

    def test_backward(self):
        # Essentially, this is drop_matmul function.
        def fn(lhs, rhs, prob):
            return F.dropout(lhs, prob) @ rhs

        # Simple function to remove routine.
        def calc_grad(fn, lhs, rhs, prob):
            T.random.manual_seed(42)
            lhs = lhs.clone().requires_grad_()
            rhs = rhs.clone().requires_grad_()
            ys = fn(lhs, rhs, prob)
            ys.backward(T.ones_like(ys))
            return lhs.grad, rhs.grad

        base_grads = calc_grad(fn, self.lhs, self.rhs, self.prob)
        curr_grads = calc_grad(drop_matmul, self.lhs, self.rhs, self.prob)

        for base_grad, curr_grad in zip(base_grads, curr_grads):
            self.assertTrue(T.allclose(base_grad, curr_grad))