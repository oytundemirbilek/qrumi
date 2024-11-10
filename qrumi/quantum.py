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
        if self.alpha**2 + self.beta**2 != 1.0:
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

    def tensordot(self, other: QuantumBit | NDArray) -> NDArray:
        """Do a matrix multiplication for joint representation in Hilbert space with the other qubit."""
        other_state = other.state if isinstance(other, QuantumBit) else other
        return np.tensordot(self.state, other_state, axes=0).flatten()


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
            hilbert_product = next_qubit.tensordot(hilbert_product)
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
        if self.theta < 0.0:
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
            self.unitary_matrix = np.array([[x11, x12], [x21, x22]])
        return self.unitary_matrix

    def tensordot(self, other: QuantumDecision | NDArray) -> NDArray:
        """Do a matrix multiplication for joint representation in Hilbert space with the other qubit."""
        if self.unitary_matrix is None:
            self.unitary_matrix = self.calculate_unitary_matrix()
        if isinstance(other, QuantumDecision):
            if other.unitary_matrix is None:
                other.unitary_matrix = other.calculate_unitary_matrix()
            other_unitary = other.unitary_matrix
        else:
            other_unitary = other
        return np.tensordot(self.unitary_matrix, other_unitary, axes=0).reshape(
            self.unitary_matrix.shape[0] ** 2, self.unitary_matrix.shape[1] ** 2
        )


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
        if self.gamma < 0.0:
            validity = False

        if not validity:
            raise QuantumDecisionError(
                "The input parameters do not meet the requirements."
            )

    def calculate_entanglement_matrix(self) -> NDArray:
        """Calculate a entanglement matrix based on the given parameters and return as numpy array."""
        defect = QuantumDecision(np.pi, 0.0)
        unitary_d = defect.calculate_unitary_matrix()
        unitary_j = np.exp(
            1.0j * self.gamma * np.tensordot(unitary_d, unitary_d, axes=0)
        )
        self.unitary = unitary_j.reshape(
            unitary_d.shape[0] ** 2, unitary_d.shape[1] ** 2
        )
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

    def get_initial_qubits(self) -> NDArray:
        """Create and calculate the tensor product of the initial qubits |00>."""
        qubits = [QuantumBit(1.0, 0.0) for _ in range(len(self.decisions))]
        qreg = QuantumRegister(qubits)
        return qreg.calculate_joint_state()

    def calculate_tensordot_unitaries(self) -> NDArray:
        """Calculate the tensor product of unitary operators from given quantum decisions."""
        unitary_product = self.decisions[-1]
        for idx in range(2, len(self.decisions) + 1):
            next_decision = self.decisions[-idx]
            unitary_product = next_decision.tensordot(unitary_product)
        if isinstance(unitary_product, QuantumDecision):
            unitary_product = unitary_product.calculate_unitary_matrix()
        return unitary_product

    def calculate_quantum_state(self) -> NDArray:
        """Calculate quantum state as a vector, after the quantum gate formula applies."""
        ent_mat = self.entanglement.calculate_entanglement_matrix()
        ent_mat_h = self.entanglement.calculate_hermitian_adjoint()
        unitary_product = self.calculate_tensordot_unitaries()
        init_qbits = self.get_initial_qubits()

        prod1 = np.matmul(ent_mat_h, unitary_product)
        prod2 = np.matmul(prod1, ent_mat)
        result = np.matmul(prod2, init_qbits)
        return result

    # def calculate_payoff_for_agent(self, payoff_matrix: NDArray) -> float:
    #     """"""
    #     return 0.0
