"""Test graph dataset classes."""

from __future__ import annotations

import copy
import os

import numpy as np
import pytest

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
    joint_result = qubit.tensordot(qubit_other)
    assert (joint_result == np.array([1.0, 0.0, 0.0, 0.0])).all(), joint_result

    joint_result = qubit.tensordot(joint_result)
    assert (
        joint_result == np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ).all(), joint_result  # qubits = |000> = [1,0,0,0,0,0,0,0]

    qubit_1 = QuantumBit(0.0, 1.0)  # qubit = |1> = [0, 1]
    joint_result = qubit_1.tensordot(qubit)  # qubits = |10> = [0,0,1,0]
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

    # max entanglement
    qent = QuantumEntanglenment(np.pi / 2.0)
    ent_mat = qent.calculate_entanglement_matrix()
    assert ent_mat.shape == (4, 4), ent_mat


def test_quantum_state() -> None:
    """Test if the model can be iterated - cpu based."""
    cooperate1 = QuantumDecision(0, 0)
    cooperate2 = QuantumDecision(0, 0)
    qstate = QuantumState(0.0, [cooperate1, cooperate2])
    qstate_vec = qstate.calculate_quantum_state()
    assert qstate_vec.shape == (4,), qstate_vec

    defect1 = QuantumDecision(np.pi, 0)
    qstate = QuantumState(0.0, [cooperate1, defect1])
    qstate_vec = qstate.calculate_quantum_state()
    assert qstate_vec.shape == (4,), qstate_vec
