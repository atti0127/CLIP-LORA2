import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from loralib.layers import HydraLinearLoRA, PlainMultiheadAttentionLoRA
from loralib.utils import (
    disabled_adapters,
    get_adapter_dir,
    get_lora_parameters,
    get_router_temperature,
    hydra_regularization_losses,
    is_adapter_parameter,
    load_lora,
    mark_only_lora_as_trainable,
    save_lora,
    set_hydra_temperature,
)


def make_attention():
    torch.manual_seed(7)
    return nn.MultiheadAttention(embed_dim=16, num_heads=4, dropout=0.)


class HydraTest(unittest.TestCase):
    def test_memory_efficient_hydra_matches_original_contraction(self):
        torch.manual_seed(11)
        linear = nn.Linear(16, 12)
        hydra = HydraLinearLoRA(
            linear, r=3, num_experts=6, dropout_rate=0.)
        with torch.no_grad():
            hydra.hydra_B.normal_()
        x = torch.randn(5, 2, 16, requires_grad=True)

        output = hydra(x)
        shared_features = F.linear(x, hydra.hydra_A)
        router_weights = F.softmax(
            F.linear(shared_features, hydra.hydra_router)
            / hydra.router_temperature,
            dim=-1)
        expert_outputs = torch.einsum(
            '...r,eor->...eo', shared_features, hydra.hydra_B)
        reference = (
            F.linear(x, hydra.weight, hydra.bias)
            + hydra.scaling * torch.einsum(
                '...e,...eo->...o', router_weights, expert_outputs))
        torch.testing.assert_close(output, reference)

        parameters = [x, hydra.hydra_A, hydra.hydra_B, hydra.hydra_router]
        output_gradients = torch.autograd.grad(
            output.square().mean(), parameters, retain_graph=True)
        reference_gradients = torch.autograd.grad(
            reference.square().mean(), parameters)
        for output_gradient, reference_gradient in zip(
                output_gradients, reference_gradients):
            torch.testing.assert_close(output_gradient, reference_gradient)

    def test_hydra_linear_starts_as_frozen_linear(self):
        linear = nn.Linear(16, 16)
        hydra = HydraLinearLoRA(linear, r=2, num_experts=3, dropout_rate=0.)
        x = torch.randn(5, 2, 16)
        torch.testing.assert_close(hydra(x), linear(x))

    def test_hydra_attention_starts_as_original_lora_attention(self):
        original = make_attention()
        lora = PlainMultiheadAttentionLoRA(
            original, enable_lora=['q', 'k', 'v'], r=2, dropout_rate=0.,
            adaptation='lora')
        hydra = PlainMultiheadAttentionLoRA(
            original, enable_lora=['q', 'k', 'v'], r=2, dropout_rate=0.,
            adaptation='hydra', num_experts=3)
        x = torch.randn(6, 2, 16)
        torch.testing.assert_close(
            hydra(x, x, x, need_weights=False)[0],
            lora(x, x, x, need_weights=False)[0])

    def test_hydra_regularization_losses_receive_gradients(self):
        layer = PlainMultiheadAttentionLoRA(
            make_attention(), enable_lora=['q', 'k', 'v'], r=2,
            adaptation='hydra', num_experts=3)
        with torch.no_grad():
            for module in layer.modules():
                if isinstance(module, HydraLinearLoRA):
                    module.hydra_B.normal_()
        x = torch.randn(6, 2, 16)
        layer(x, x, x, need_weights=False)
        diversity, balance = hydra_regularization_losses([layer])
        (diversity + balance).backward()

        self.assertGreaterEqual(diversity.item(), 0.)
        self.assertGreaterEqual(balance.item(), 0.)
        self.assertIsNotNone(layer.q_proj.hydra_B.grad)
        self.assertGreater(layer.q_proj.hydra_B.grad.abs().sum().item(), 0.)
        self.assertIsNotNone(layer.q_proj.hydra_router.grad)
        self.assertGreater(layer.q_proj.hydra_router.grad.abs().sum().item(), 0.)

    def test_router_temperature_schedule(self):
        args = SimpleNamespace(
            router_temperature=1.0,
            router_temperature_end=0.1,
            router_temperature_schedule='cosine',
        )
        self.assertEqual(get_router_temperature(args, 0, 11), 1.0)
        self.assertAlmostEqual(get_router_temperature(args, 5, 11), 0.55)
        self.assertAlmostEqual(get_router_temperature(args, 10, 11), 0.1)

        layer = PlainMultiheadAttentionLoRA(
            make_attention(), enable_lora=['q', 'k', 'v'], r=2,
            adaptation='hydra', num_experts=3)
        set_hydra_temperature([layer], 0.55)
        self.assertEqual(layer.q_proj.router_temperature, 0.55)
        self.assertEqual(layer.k_proj.router_temperature, 0.55)
        self.assertEqual(layer.v_proj.router_temperature, 0.55)

    def test_only_adapter_parameters_are_trainable(self):
        model = nn.Sequential(
            PlainMultiheadAttentionLoRA(
                make_attention(), enable_lora=['q', 'k', 'v'], r=2,
                adaptation='hydra', num_experts=3))
        mark_only_lora_as_trainable(model)
        trainable_names = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(trainable_names)
        self.assertTrue(all(is_adapter_parameter(name) for name in trainable_names))
        self.assertEqual(
            len(get_lora_parameters(model)),
            len(trainable_names))

    def test_disabled_adapters_restores_frozen_attention_path(self):
        original = make_attention()
        layer = PlainMultiheadAttentionLoRA(
            original, enable_lora=['q', 'k', 'v'], r=2, dropout_rate=0.,
            adaptation='hydra', num_experts=3)
        with torch.no_grad():
            for module in layer.modules():
                if isinstance(module, HydraLinearLoRA):
                    module.hydra_B.normal_()
        x = torch.randn(6, 2, 16)

        adapted = layer(x, x, x, need_weights=False)[0]
        with disabled_adapters([layer]):
            frozen = layer(x, x, x, need_weights=False)[0]
        restored = layer(x, x, x, need_weights=False)[0]
        reference = original(x, x, x, need_weights=False)[0]

        torch.testing.assert_close(frozen, reference)
        torch.testing.assert_close(restored, adapted)
        self.assertGreater((adapted - frozen).abs().sum().item(), 0.)

    def test_experts_receive_gradients(self):
        layer = PlainMultiheadAttentionLoRA(
            make_attention(), enable_lora=['q', 'k', 'v'], r=2,
            adaptation='hydra', num_experts=3)
        mark_only_lora_as_trainable(layer)
        optimizer = torch.optim.SGD(get_lora_parameters(layer), lr=0.1)
        x = torch.randn(6, 2, 16)
        layer(x, x, x, need_weights=False)[0].square().mean().backward()

        self.assertIsNotNone(layer.q_proj.hydra_B.grad)
        self.assertGreater(layer.q_proj.hydra_B.grad.abs().sum().item(), 0)
        self.assertFalse(torch.equal(
            layer.q_proj.hydra_B.grad[0],
            layer.q_proj.hydra_B.grad[1]))

        optimizer.step()
        optimizer.zero_grad()
        layer(x, x, x, need_weights=False)[0].square().mean().backward()
        self.assertGreater(layer.q_proj.hydra_router.grad.abs().sum().item(), 0)
        self.assertGreater(layer.q_proj.hydra_A.grad.abs().sum().item(), 0)

    def test_hydra_checkpoint_round_trip(self):
        args = SimpleNamespace(
            adaptation='hydra',
            alpha=1,
            backbone='ViT-B/16',
            dataset='dtd',
            encoder='both',
            filename='adapter',
            num_experts=3,
            params=['q', 'k', 'v'],
            position='all',
            r=2,
            router_temperature=0.1,
            router_temperature_end=0.1,
            router_temperature_schedule='fixed',
            hydra_diversity_weight=0.01,
            hydra_balance_weight=0.01,
            image_anchor_weight=0.5,
            text_anchor_weight=1.0,
            seed=1,
            shots=4,
        )
        self.assertEqual(get_adapter_dir(args), 'hydra_regularized')
        source = PlainMultiheadAttentionLoRA(
            make_attention(), enable_lora=args.params, r=args.r,
            adaptation=args.adaptation, num_experts=args.num_experts,
            router_temperature=args.router_temperature)
        target = PlainMultiheadAttentionLoRA(
            make_attention(), enable_lora=args.params, r=args.r,
            adaptation=args.adaptation, num_experts=args.num_experts,
            router_temperature=args.router_temperature)
        with torch.no_grad():
            for name, parameter in source.named_parameters():
                if is_adapter_parameter(name):
                    parameter.add_(torch.randn_like(parameter))

        with tempfile.TemporaryDirectory() as directory:
            args.save_path = directory
            save_lora(args, [source])
            load_lora(args, [target])

        for name, parameter in source.named_parameters():
            if is_adapter_parameter(name):
                torch.testing.assert_close(
                    parameter, dict(target.named_parameters())[name])


if __name__ == '__main__':
    unittest.main()
