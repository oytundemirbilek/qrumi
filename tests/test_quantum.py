"""Test graph dataset classes."""

from __future__ import annotations

import copy
import os

import numpy as np
import pytest
from numpy.typing import NDArray

from qrumi.errors import QuantumDecisionError
from qrumi.quantum import (
    QuantumBit,
    QuantumDecision,
    QuantumDecisionNplayers,
    QuantumEntanglenment,
    QuantumRegister,
    QuantumState,
)

GOLD_STANDARD_PATH = os.path.join(os.path.dirname(__file__), "expected")
DEVICE = "cpu"
GAMMA = np.pi / 2.0


def strategy_mapping(strategy: str) -> QuantumDecision:
    """Create a QuantumDecision object based on the provided code of the strategy."""
    if strategy == "C":
        return QuantumDecision(0, 0)
    if strategy == "D":
        return QuantumDecision(np.pi, 0)
    if strategy == "Q":
        return QuantumDecision(0, np.pi / 2.0)
    else:
        raise ValueError("Unknown strategy.")


def test_quantum_bit() -> None:
    """Test if quantum bit e.g., |0> is represented as expected."""
    with pytest.raises(QuantumDecisionError):
        QuantumBit(0.0, 0.0)

    with pytest.raises(QuantumDecisionError):
        QuantumBit(1.0, 1.0)

    with pytest.raises(QuantumDecisionError):
        QuantumBit(0.3, 0.4)

    qubit = QuantumBit(0.6, 0.8)
    measured = qubit.measure()
    assert measured in {0, 1}

    qubit = QuantumBit(1.0, 0.0)  # qubit = |0> = [1, 0]
    measured = qubit.measure()
    assert measured == 0
    assert (qubit.state == np.array([1.0, 0.0])).all()

    qubit_other = copy.copy(qubit)  # qubits = |00> = [1,0,0,0]
    joint_result = qubit.kronecker(qubit_other)
    assert (joint_result == np.array([1.0, 0.0, 0.0, 0.0])).all(), joint_result

    joint_result = qubit.kronecker(joint_result)
    assert (
        joint_result == np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ).all(), joint_result  # qubits = |000> = [1,0,0,0,0,0,0,0]

    qubit_1 = QuantumBit(0.0, 1.0)  # qubit = |1> = [0, 1]
    joint_result = qubit_1.kronecker(qubit)  # qubits = |10> = [0,0,1,0]
    assert (joint_result == np.array([0.0, 0.0, 1.0, 0.0])).all(), joint_result


def test_quantum_register() -> None:
    """Test if quantum register e.g., |00> is represented as expected."""
    qubit_0 = QuantumBit(1.0, 0.0)  # qubit = |0> = [1, 0]
    qubit_1 = QuantumBit(0.0, 1.0)  # qubit = |1> = [0, 1]
    qubit_00 = QuantumBit(1.0, 0.0)
    qubit_000 = QuantumBit(1.0, 0.0)

    qreg = QuantumRegister([qubit_0, qubit_00])  # qubits = |00>
    joint_state = qreg.calculate_joint_state()
    assert (joint_state == np.array([1.0, 0.0, 0.0, 0.0])).all(), joint_state

    qreg = QuantumRegister([qubit_1, qubit_00])  # qubits = |10>
    joint_state = qreg.calculate_joint_state()
    assert (joint_state == np.array([0.0, 0.0, 1.0, 0.0])).all(), joint_state

    qreg = QuantumRegister([qubit_1, qubit_1])  # qubits = |11>
    joint_state = qreg.calculate_joint_state()
    assert (joint_state == np.array([0.0, 0.0, 0.0, 1.0])).all(), joint_state

    qreg = QuantumRegister([qubit_0, qubit_00, qubit_000])  # qubits = |000>
    joint_state = qreg.calculate_joint_state()
    assert (
        joint_state == np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ).all(), joint_state


def test_quantum_decision_2players() -> None:
    """Test if the model can be iterated - cpu based."""
    with pytest.raises(QuantumDecisionError):
        QuantumDecision(-np.pi, 0)

    with pytest.raises(QuantumDecisionError):
        QuantumDecision(0, 2 * np.pi)

    cooperate = QuantumDecision(0, 0)
    assert np.isclose(
        cooperate.calculate_unitary_matrix(),
        np.array([[1 + 0j, 0], [0, 1 + 0j]]),
    ).all()

    defect = QuantumDecision(np.pi, 0)
    assert np.isclose(
        defect.calculate_unitary_matrix(),
        np.array([[0, 1 + 0j], [-1 + 0j, 0]]),
    ).all()

    quantum = QuantumDecision(0, np.pi / 2)
    assert np.isclose(
        quantum.calculate_unitary_matrix(),
        np.array([[0 + 1j, 0], [0, 0 - 1j]]),
    ).all()


def test_quantum_decision_nplayers() -> None:
    """Test if the model can be iterated - cpu based."""
    with pytest.raises(QuantumDecisionError):
        QuantumDecisionNplayers(0, 2 * np.pi, 0)

    with pytest.raises(QuantumDecisionError):
        QuantumDecisionNplayers(0, 0, -np.pi)

    cooperate = QuantumDecisionNplayers(0, 0, 0)
    assert np.isclose(
        cooperate.calculate_unitary_matrix(),
        np.array([[1 + 0j, 0], [0, 1 + 0j]]),
    ).all()

    defect = QuantumDecisionNplayers(0, 0, np.pi)
    assert np.isclose(
        defect.calculate_unitary_matrix(), np.array([[0, 0 + 1j], [0 + 1j, 0]])
    ).all()

    quantum = QuantumDecisionNplayers(np.pi / 2, 0, 0)
    assert np.isclose(
        quantum.calculate_unitary_matrix(),
        np.array([[0 + 1j, 0], [0, 0 - 1j]]),
    ).all()


def test_quantum_entanglement() -> None:
    """Test if the model can be iterated - cpu based."""
    with pytest.raises(QuantumDecisionError):
        QuantumEntanglenment(np.pi)
    # no entanglement
    qent = QuantumEntanglenment(0.0)
    ent_mat = qent.calculate_entanglement_matrix()
    assert ent_mat.shape == (4, 4), ent_mat
    assert np.isclose(
        ent_mat, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    ).all(), ent_mat

    # max entanglement
    qent = QuantumEntanglenment(np.pi / 2.0)
    ent_mat = qent.calculate_entanglement_matrix()
    assert ent_mat.shape == (4, 4), ent_mat
    assert np.isclose(
        ent_mat,
        np.array(
            [
                [0.70710678, 0, 0, 0.70710678j],
                [0, 0.70710678, -0.70710678j, 0],
                [0, -0.70710678j, 0.70710678, 0],
                [0.70710678j, 0, 0, 0.70710678],
            ]
        ),
    ).all(), ent_mat


@pytest.mark.parametrize(
    argnames=[
        "strategy_a",
        "strategy_b",
        "gamma",
        "expected_state",
        "expected_payoff",
    ],
    argvalues=[
        ("C", "C", 0.0, np.array([1, 0, 0, 0]), 3),  # |CC> wo quantum entanglement
        ("C", "D", 0.0, np.array([0, -1, 0, 0]), 0),  # |CD> wo quantum entanglement
        ("D", "D", 0.0, np.array([0, 0, 0, 1]), 1),  # |DD> wo quantum entanglement
        ("D", "C", GAMMA, np.array([0, 0, -1, 0]), 5),  # |DC> with quantum entanglement
        ("Q", "D", GAMMA, np.array([0, 0, 1, 0]), 5),  # |QD> with quantum entanglement
        ("D", "Q", GAMMA, np.array([0, 1, 0, 0]), 0),  # |DQ> with quantum entanglement
        ("Q", "Q", GAMMA, np.array([-1, 0, 0, 0]), 3),  # |QQ> with quantum entanglement
        ("C", "Q", GAMMA, np.array([0, 0, 0, 1]), 1),  # |CQ> with quantum entanglement
        ("Q", "C", GAMMA, np.array([0, 0, 0, 1]), 1),  # |CQ> with quantum entanglement
    ],
)
def test_quantum_state(
    strategy_a: str,
    strategy_b: str,
    gamma: float,
    expected_state: NDArray,
    expected_payoff: float,
) -> None:
    """Test if the model can be iterated - cpu based."""
    decision_a = strategy_mapping(strategy_a)
    decision_b = strategy_mapping(strategy_b)
    qstate = QuantumState(gamma, [decision_a, decision_b])

    # Check initial state
    assert qstate.qstate_vec.shape == (4,), qstate.qstate_vec
    if gamma == 0.0:
        assert np.isclose(
            qstate.qstate_vec, np.array([1, 0, 0, 0])
        ).all(), qstate.qstate_vec
    else:
        assert np.isclose(
            qstate.qstate_vec, np.array([0.70710678, 0, 0, 0.70710678j])
        ).all(), qstate.qstate_vec

    qstate_vec = qstate.calculate_quantum_state()
    assert qstate_vec.shape == (4,), qstate_vec
    assert np.isclose(qstate_vec, expected_state).all(), qstate_vec
    payoff_a = qstate.calculate_expected_payoff(np.array([3, 0, 5, 1]))
    assert payoff_a == expected_payoff, payoff_a
