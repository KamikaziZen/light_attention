import torch as T
import torch.nn.functional as F

from unittest import TestCase

from light_attention.nn.functional.light_softmax import light_softmax


class TestLightSoftmax(TestCase):

    def test_forward(self):
        T.random.manual_seed(42)
        xs = T.randn((3, 4))
        origin_ys = F.softmax(xs.clone(), dim=-1)
        light_ys = light_softmax(xs.clone())
        self.assertTrue(T.allclose(origin_ys, light_ys))

    def test_backward(self):
        T.random.manual_seed(42)
        xs = T.randn((3, 4))

        origin_xs = xs.clone().requires_grad_()
        origin_ys = F.softmax(origin_xs, dim=-1)
        origin_ys.backward(T.ones_like(origin_xs))
        origin_dxs = origin_xs.grad

        light_xs = xs.clone().requires_grad_()
        light_ys = light_softmax(light_xs)
        light_ys.backward(T.ones_like(light_xs))
        light_dxs = light_xs.grad

        self.assertTrue(T.allclose(origin_dxs, light_dxs))