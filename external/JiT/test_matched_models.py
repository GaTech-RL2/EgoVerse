import torch

from matched_models import EndpointLatentObjective


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
