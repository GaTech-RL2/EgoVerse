"""The device the trainer actually runs on has to reach ``HPTModel``.

``HPT`` builds its policy on CPU and leaves both devices unset; Lightning
learns the real one and ``ModelWrapper.on_fit_start`` / ``on_validation_start``
write it to the algo. If that write stops at the algo,
``HPTModel.compute_loss`` builds its loss accumulators on the unset device
while ``HPT.compute_losses`` accumulates on the trainer's, and the run dies
with "Expected all tensors to be on the same device".
"""

from __future__ import annotations

import torch

from egomimic.algo.hpt import HPT, HPTModel


def _algo_with_policy(construction_device: torch.device):
    """An algo holding a policy, without running either ``__init__``.

    Building a real HPT means a full trunk, stems and heads; only the device
    wiring is under test here, so both objects are created bare, exactly as
    ``test_infra_port_wiring`` does for the stem registration.
    """
    algo = HPT.__new__(HPT)
    algo.nets = torch.nn.ModuleDict()
    algo.device = construction_device
    policy = HPTModel.__new__(HPTModel)
    torch.nn.Module.__init__(policy)
    policy.device = construction_device
    algo.nets["policy"] = policy
    return algo, policy


def test_assigning_the_algo_device_reaches_the_policy():
    algo, policy = _algo_with_policy(torch.device("cuda"))

    algo.device = torch.device("cpu")  # what ModelWrapper.on_fit_start does

    assert policy.device == torch.device("cpu")


def test_algo_without_a_policy_still_takes_a_device():
    """The algo sets its device in ``__init__`` before the policy is built."""
    algo = HPT.__new__(HPT)
    algo.nets = torch.nn.ModuleDict()

    algo.device = torch.device("cpu")

    assert algo.device == torch.device("cpu")
