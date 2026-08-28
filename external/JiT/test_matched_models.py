import pytest
import torch

from matched_models import (
    EndpointLatentObjective,
    JiTEndpointObjective,
    trainable_parameter_count,
)
from train_matched import current_lr


def tiny_model() -> EndpointLatentObjective:
    return EndpointLatentObjective(
        image_size=8,
        patch_size=4,
        latent_dim=4,
        hidden_dim=8,
        depth=2,
        num_heads=2,
        dropout=0.0,
        mlp_layers=1,
        mlp_ratio=2.0,
        decoder_hidden_dim=8,
        num_classes=3,
        gradient_checkpointing=False,
    )


def tiny_jit_endpoint() -> JiTEndpointObjective:
    return JiTEndpointObjective(
        image_size=8,
        patch_size=4,
        hidden_size=48,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        bottleneck_dim=16,
        in_context_len=2,
        in_context_start=1,
        num_classes=3,
        label_drop_prob=0.0,
        gradient_checkpointing=False,
    )


def test_integration_grids_are_positive_fp32_and_sum_to_one():
    for steps in (1, 2, 4, 8, 16):
        grid = EndpointLatentObjective.sample_step_sizes(7, steps, torch.device("cpu"))
        assert grid.dtype == torch.float32
        assert torch.all(grid > 0)
        torch.testing.assert_close(grid.sum(-1), torch.ones(7))


def test_optimizer_step_curriculum_matches_action_reference():
    assert [EndpointLatentObjective.unroll_steps_at(step) for step in range(1, 7)] == [
        1,
        2,
        1,
        2,
        1,
        2,
    ]
    cycle = [EndpointLatentObjective.unroll_steps_at(step) for step in range(2001, 2021)]
    assert cycle == [2] * 16 + [4] * 3 + [8]


def test_target_is_not_an_input_to_latent_generation():
    model = tiny_model().eval()
    labels = torch.tensor([0, 1])
    noise = torch.randn(2, model.num_tokens, model.latent_dim)
    steps = torch.full((2, 2), 0.5)
    first, first_result = model.predict(
        labels, optimizer_step=2, force_steps=2, noise=noise, step_sizes=steps
    )
    second, second_result = model.predict(
        labels, optimizer_step=2, force_steps=2, noise=noise, step_sizes=steps
    )
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first_result.endpoint, second_result.endpoint)


def test_terminal_image_loss_reaches_decoder_and_field():
    model = tiny_model().train()
    images = torch.randn(2, 3, 8, 8)
    labels = torch.tensor([0, 1])
    metrics = model(images, labels, optimizer_step=2, force_steps=2)
    assert torch.isfinite(metrics["loss"])
    metrics["loss"].backward()
    decoder_grad = sum(
        parameter.grad.abs().sum()
        for parameter in model.decoder.parameters()
        if parameter.grad is not None
    )
    field_grad = sum(
        parameter.grad.abs().sum()
        for parameter in model.field.parameters()
        if parameter.grad is not None
    )
    assert decoder_grad > 0
    assert field_grad > 0


def test_sampling_shape_and_noise_sensitivity():
    model = tiny_model().eval()
    labels = torch.tensor([0, 1])
    first = model.sample(labels, num_steps=2)
    second = model.sample(labels, num_steps=2)
    assert first.shape == (2, 3, 8, 8)
    assert not torch.equal(first, second)


def test_jit_endpoint_has_no_image_input_stem_and_exact_full_parameter_count():
    model = JiTEndpointObjective()
    assert not hasattr(model.net, "x_embedder")
    assert trainable_parameter_count(model) == 131_516_928


def test_jit_endpoint_target_is_not_an_input_and_noise_is_not_mutated():
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    model = tiny_jit_endpoint().to(device).eval()
    labels = torch.tensor([0, 1], device=device)
    noise = torch.randn(2, model.num_tokens, model.latent_dim, device=device)
    original = noise.clone()
    step_sizes = torch.full((2, 2), 0.5, device=device)
    first, first_result = model.predict(
        labels,
        optimizer_step=2,
        force_steps=2,
        noise=noise,
        step_sizes=step_sizes,
    )
    second, second_result = model.predict(
        labels,
        optimizer_step=2,
        force_steps=2,
        noise=noise,
        step_sizes=step_sizes,
    )
    torch.testing.assert_close(noise, original)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first_result.endpoint, second_result.endpoint)


def test_jit_endpoint_terminal_loss_reaches_decoder_then_field():
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    model = tiny_jit_endpoint().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    images = torch.randn(2, 3, 8, 8, device=device)
    labels = torch.tensor([0, 1], device=device)

    first = model(images, labels, optimizer_step=1, force_steps=1)["loss"]
    first.backward()
    decoder_grad = model.decoder.weight.grad.abs().sum()
    velocity_head_grad = model.net.final_layer.linear.weight.grad.abs().sum()
    assert decoder_grad > 0
    assert velocity_head_grad > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second = model(images, labels, optimizer_step=2, force_steps=2)["loss"]
    second.backward()
    block_modulation_grad = sum(
        parameter.grad.abs().sum()
        for block in model.net.blocks
        for parameter in block.adaLN_modulation.parameters()
        if parameter.grad is not None
    )
    assert block_modulation_grad > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    third = model(images, labels, optimizer_step=3, force_steps=1)["loss"]
    third.backward()
    block_grad = sum(
        parameter.grad.abs().sum()
        for parameter in model.net.blocks.parameters()
        if parameter.grad is not None
    )
    time_grad = sum(
        parameter.grad.abs().sum()
        for parameter in model.net.t_embedder.parameters()
        if parameter.grad is not None
    )
    label_grad = model.net.y_embedder.embedding_table.weight.grad.abs().sum()
    assert block_grad > 0
    assert time_grad > 0
    assert label_grad > 0


def test_jit_endpoint_sampling_shape_and_noise_sensitivity():
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    model = tiny_jit_endpoint().to(device).eval()
    labels = torch.tensor([0, 1], device=device)
    first = model.sample(labels, num_steps=2)
    second = model.sample(labels, num_steps=2)
    assert first.shape == (2, 3, 8, 8)
    assert not torch.equal(first, second)


def test_action_lr_schedule_has_exact_warmup_peak_and_floor():
    kwargs = dict(
        base_lr=3e-5,
        effective_batch=1024,
        updates_per_epoch=1,
        warmup_epochs=0,
        schedule="action_warmup_cosine",
        min_lr=3e-6,
        warmup_steps=3000,
        warmup_start_factor=0.1,
        total_steps=240000,
    )
    assert current_lr(optimizer_step=0, **kwargs) == pytest.approx(3e-6)
    assert current_lr(optimizer_step=3000, **kwargs) == pytest.approx(3e-5)
    assert current_lr(optimizer_step=240000, **kwargs) == pytest.approx(3e-6)
    assert 3e-6 < current_lr(optimizer_step=120000, **kwargs) < 3e-5
