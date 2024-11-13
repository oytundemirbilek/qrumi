"""Module to represent quantum methodologies."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from qrumi.errors import QuantumDecisionError


class QuantumBit:
    """Class to represent a qubit."""

    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta
        self.state = np.array([self.alpha, self.beta])
        self.check_qubit()

    def check_qubit(self) -> None:
        """Check input parameters if they satisfy the alpha^2 + beta^2 = 1."""
        validity = True
        if self.alpha**2 + self.beta**2 != 1:
            validity = False

        if not validity:
            raise QuantumDecisionError(
                "The input parameters do not meet the requirements."
            )

    def measure(self) -> int:
        """Measure a quantum bit to project it into the classical bits (return either 0 or 1)."""
        prob_bit = np.random.rand()
        measured = 0 if prob_bit <= self.alpha**2 else 1
        return measured

    def kronecker(self, other: QuantumBit | NDArray) -> NDArray:
        """Do a matrix multiplication for joint representation in Hilbert space with the other qubit."""
        other_state = other.state if isinstance(other, QuantumBit) else other
        return np.kron(self.state, other_state)


class QuantumRegister:
    """Class to represent multiple qubits in a register written as: |00>."""

    def __init__(self, qubits: list[QuantumBit]):
        self.qubits = qubits
        self.joint_state: NDArray

    def calculate_joint_state(self) -> NDArray:
        """Calculate the joint state of given qubits in Hilbert space."""
        hilbert_product = self.qubits[-1].state
        for idx in range(2, len(self.qubits) + 1):
            next_qubit = self.qubits[-idx]
            hilbert_product = next_qubit.kronecker(hilbert_product)
        self.joint_state = hilbert_product
        return hilbert_product


class QuantumDecision:
    """Class to represent a quantum decision as a unitary operator."""

    def __init__(self, theta: float, fi: float):
        self.fi = fi
        self.theta = theta
        self.unitary_matrix: NDArray | None = None
        self.check_input_parameters()

    def check_input_parameters(self) -> None:
        """Check input parameters for their validity and return True if there is no violation."""
        validity = True
        if self.fi > np.pi:
            validity = False
        if self.fi < -np.pi:
            validity = False

        if self.theta > np.pi:
            validity = False
        if self.theta < 0:
            validity = False

        if not validity:
            raise QuantumDecisionError(
                "The input parameters do not meet the requirements."
            )

    def calculate_unitary_matrix(self) -> NDArray:
        """Calculate a unitary matrix based on the given parameters and return as numpy array."""
        if self.unitary_matrix is None:
            x11 = np.cos(self.theta / 2.0) * np.exp(1.0j * self.fi)
            x12 = np.sin(self.theta / 2.0)
            x21 = -1.0 * np.sin(self.theta / 2.0)
            x22 = np.cos(self.theta / 2.0) * np.exp(-1.0j * self.fi)
            self.unitary_matrix = np.round(np.array([[x11, x12], [x21, x22]]), 10)
        return self.unitary_matrix

    def kronecker(self, other: QuantumDecision | NDArray) -> NDArray:
        """Do a matrix multiplication for joint representation in Hilbert space with the other qubit."""
        if self.unitary_matrix is None:
            self.unitary_matrix = self.calculate_unitary_matrix()
        if isinstance(other, QuantumDecision):
            if other.unitary_matrix is None:
                other.unitary_matrix = other.calculate_unitary_matrix()
            other_unitary = other.unitary_matrix
        else:
            other_unitary = other
        return np.kron(self.unitary_matrix, other_unitary)


class QuantumDecisionNplayers(QuantumDecision):
    """Class to represent a quantum decision as a unitary operator."""

    def __init__(self, alpha: float, beta: float, theta: float):
        self.beta = beta
        super().__init__(theta, alpha)

    def check_input_parameters(self) -> None:
        """Check input parameters for their validity and return True if there is no violation."""
        validity = True
        super().check_input_parameters()

        if self.beta > np.pi:
            validity = False
        if self.beta < -np.pi:
            validity = False

        if not validity:
            raise QuantumDecisionError(
                "The input parameters do not meet the requirements."
            )

    def calculate_unitary_matrix(self) -> NDArray:
        """Calculate a unitary matrix based on the given parameters and return as numpy array."""
        if self.unitary_matrix is None:
            x11 = np.cos(self.theta / 2.0) * np.exp(1.0j * self.fi)
            x12 = np.sin(self.theta / 2.0) * 1.0j * np.exp(1.0j * self.beta)
            x21 = np.sin(self.theta / 2.0) * 1.0j * np.exp(-1.0j * self.beta)
            x22 = np.cos(self.theta / 2.0) * np.exp(-1.0j * self.fi)
            self.unitary_matrix = np.array([[x11, x12], [x21, x22]])
        return self.unitary_matrix


class QuantumEntanglenment:
    """Class to represent a quantum entanglement (often notated as J)."""

    def __init__(self, gamma: float):
        self.gamma = gamma
        self.unitary: NDArray | None = None
        self.check_input_parameters()

    def check_input_parameters(self) -> None:
        """Check input parameters for their validity and return True if there is no violation."""
        validity = True
        if self.gamma > np.pi / 2.0:
            validity = False
        if self.gamma < 0:
            validity = False

        if not validity:
            raise QuantumDecisionError(
                "The input parameters do not meet the requirements."
            )

    def calculate_entanglement_matrix(self) -> NDArray:
        """Calculate a entanglement matrix based on the given parameters and return as numpy array."""
        if self.unitary is None:
            defect = QuantumDecision(np.pi, 0.0)
            unitary_d = defect.calculate_unitary_matrix()
            ident = np.identity(2)
            ident_kron = np.kron(ident, ident)
            unitary_kron = np.kron(unitary_d, unitary_d)
            coef = ident_kron * np.cos(self.gamma / 2) + 1.0j * unitary_kron * np.sin(
                self.gamma / 2
            )
            self.unitary = np.round(coef, 10)
        return self.unitary

    def calculate_hermitian_adjoint(self) -> NDArray:
        """Calculate Hermitian transpose of the entanglement matrix - a conjugate transpose."""
        if self.unitary is None:
            self.unitary = self.calculate_entanglement_matrix()
        # Following should be equivalent to conjugate transpose, also known as the Hermitian transpose:
        return self.unitary.conj().T.copy()


class QuantumState:
    """Class to represent a quantum state within the game (often notated as psi)."""

    def __init__(self, gamma: float, decisions: list[QuantumDecision]):
        self.gamma = gamma
        self.decisions = decisions
        self.entanglement = QuantumEntanglenment(self.gamma)
        ent_mat = self.entanglement.calculate_entanglement_matrix()
        self.qstate_vec: NDArray
        init_qbits = self.get_initial_qubits()
        self.qstate_vec = np.matmul(ent_mat, init_qbits)

    def get_initial_qubits(self) -> NDArray:
        """Create and calculate the tensor product of the initial qubits |00>."""
        qubits = [QuantumBit(1.0, 0.0) for _ in range(len(self.decisions))]
        qreg = QuantumRegister(qubits)
        return qreg.calculate_joint_state()

    def calculate_kronecker_unitaries(self) -> NDArray:
        """Calculate the tensor product of unitary operators from given quantum decisions."""
        curr_product = self.decisions[0].calculate_unitary_matrix()
        for idx in range(1, len(self.decisions)):
            next_decision = self.decisions[idx].calculate_unitary_matrix()
            curr_product = np.kron(curr_product, next_decision)
        return curr_product

    def calculate_quantum_state(self) -> NDArray:
        """Calculate quantum state as a vector, after the quantum gate formula applies."""
        ent_mat = self.entanglement.calculate_entanglement_matrix()
        ent_mat_h = self.entanglement.calculate_hermitian_adjoint()
        unitary_product = self.calculate_kronecker_unitaries()
        init_qbits = self.get_initial_qubits()

        prod1 = np.matmul(ent_mat_h, unitary_product)
        prod2 = np.matmul(prod1, ent_mat)
        self.qstate_vec = np.matmul(prod2, init_qbits)
        return self.qstate_vec

    @staticmethod
    def get_all_possible_decision_qubits() -> NDArray:
        """Create and calculate the tensor product for the all  qubits |00>."""
        qreg00 = QuantumRegister([QuantumBit(1.0, 0.0), QuantumBit(1.0, 0.0)])
        qreg01 = QuantumRegister([QuantumBit(1.0, 0.0), QuantumBit(0.0, 1.0)])
        qreg10 = QuantumRegister([QuantumBit(0.0, 1.0), QuantumBit(1.0, 0.0)])
        qreg11 = QuantumRegister([QuantumBit(0.0, 1.0), QuantumBit(0.0, 1.0)])
        qregs = [qreg00, qreg01, qreg10, qreg11]
        return np.array([qreg.calculate_joint_state() for qreg in qregs])

    def calculate_expected_payoff(self, payoff_matrix: NDArray) -> float:
        """Calculate the payoff for first agent in current state from a given payoff matrix."""
        payoffs = payoff_matrix.flatten()
        qubits = self.get_all_possible_decision_qubits()
        total_payoff = 0.0
        # Disable ruff check here to support python<3.10
        for qubit, payoff in zip(qubits, payoffs):  # noqa: B905
            dotprod = np.dot(self.qstate_vec, qubit)
            total_payoff += payoff * dotprod**2
        return np.round(total_payoff, 5)
