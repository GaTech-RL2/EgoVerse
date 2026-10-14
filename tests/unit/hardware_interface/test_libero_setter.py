"""Only an exact native normalization of an already-unit quaternion is allowed."""

from types import SimpleNamespace

import numpy as np
import pytest

from astra_reversal.hardware_interface.reset_validation import validate_setter


def fixture(kind=0):
    nq = 7 if kind == 0 else 4
    model = SimpleNamespace(nq=nq, nv=6, na=1, njnt=1, jnt_type=[kind], jnt_qposadr=[0])
    state = np.arange(1 + nq + model.nv + model.na, dtype=float)
    start = 4 if kind == 0 else 1
    state[start : start + 4] = [0.5, 0.5, 0.5, np.nextafter(0.5, 1.0)]
    return model, state, start


def normalized(state, start):
    result = state.copy()
    result[start : start + 4] = [0.5] * 4
    return result


def native_stub(quaternion):
    # An explicit injected native output lets these tests check the acceptance
    # boundary independently of NumPy's or MuJoCo's normalization arithmetic.
    quaternion[:] = [0.5] * 4


@pytest.mark.parametrize("kind", [0, 1])
def test_exact_native_normalization_and_input_immutability(kind):
    model, state, start = fixture(kind)
    original = state.copy()
    actual = normalized(state, start)
    receipt = validate_setter(state, actual, model, normalize=native_stub)
    assert receipt["match"] == "native_unit_quaternion_normalization"
    assert receipt["normalized_joints"] == [0]
    assert np.array_equal(state, original)
    assert np.array_equal(actual, normalized(state, start))


def test_exact_match_does_not_normalize():
    model, state, _ = fixture()
    receipt = validate_setter(
        state, state.copy(), model, normalize=lambda q: pytest.fail()
    )
    assert receipt["match"] == "exact"


@pytest.mark.parametrize("index", [0, 1, 2, 3, 8, 12, 14])
def test_non_quaternion_changes_rejected_even_at_one_ulp(index):
    model, state, start = fixture()
    actual = normalized(state, start)
    actual[index] = np.nextafter(actual[index], np.inf)
    with pytest.raises(ValueError, match="official_state_setter_mismatch"):
        validate_setter(state, actual, model, normalize=native_stub)


def test_arbitrary_quaternion_difference_is_not_a_tolerance_match():
    model, state, start = fixture()
    actual = normalized(state, start)
    actual[start] = np.nextafter(actual[start], np.inf)
    with pytest.raises(ValueError, match="official_state_setter_mismatch"):
        validate_setter(state, actual, model, normalize=native_stub)


@pytest.mark.parametrize("bad", [1.01, float("nan"), float("inf")])
def test_nonunit_or_nonfinite_quaternions_rejected(bad):
    model, state, start = fixture()
    actual = normalized(state, start)
    state[start] = bad
    with pytest.raises(ValueError, match="official_state_setter_mismatch"):
        validate_setter(state, actual, model, normalize=native_stub)


def test_shape_mismatch_rejected():
    model, state, _ = fixture()
    with pytest.raises(ValueError, match="official_state_setter_mismatch"):
        validate_setter(state, state[:-1], model, normalize=native_stub)
