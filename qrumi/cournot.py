"""Module to simulate a newsvendor problem with quantum entanglement."""

# Model setup from Prisoner’s dilemma on competing retailers’ investment in
# green supply chain management paper.

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Retailer:
    """Class represents a retailer and its related functions."""

    def __init__(
        self,
        unit_cost: float,
        coop_cost: float,
        coop_effect: float,
    ):
        self.unit_cost = unit_cost  # c_x/c_y
        self.coop_cost = coop_cost  # g_x/g_y
        self.coop_effect = coop_effect  # G

        # To be calculated:
        self.demand = 0.0  # q_x/q_y
        self.retail_price = 0.0  # p_x/p_y
        self.profit = 0  # pi_x/pi_y

    def calculate_optimal_retail_price(self, beta: float) -> float:
        """Calculate optimal retail price for the retailer, based on current condition."""
        nom = (
            1
            + self.unit_cost
            + self.coop_cost
            - beta
            + self.coop_effect * (1 - beta**2)
        )
        denom = 2 - beta
        self.retail_price = nom / denom
        return self.retail_price

    def calculate_demand(self, beta: float, other_price: float) -> float:
        """Calculate expected demand for the retailer."""
        self.demand = (
            1 / (1 + beta)
            - 1 / (1 - beta**2) * self.retail_price
            + beta * other_price * 1 / (1 - beta**2)
            + self.coop_effect
        )
        return self.demand

    def calculate_profit(self) -> float:
        """Calculate profit."""
        self.profit = self.demand * (
            self.retail_price - self.unit_cost - self.coop_cost
        )
        return self.profit


class InvestmentModel:
    """Class that represents the situation between the retailers, manufacturer and market."""

    def __init__(
        self,
        beta: float = 0.5,
        unit_cost: float = 1.0,
        coop_effects: tuple[float, float] = (1.0, 1.0),
        coop_costs: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        """
        Initialize class.

        Parameters
        ----------
        beta: float between 0 and 1.
            degree of differentiation between the retailers. 0 means two retailers are independent.
            1 means products sold by the two retailers are fully substitutable.
        """
        self.beta = beta
        self.unit_cost = unit_cost
        self.coop_effect_x, self.coop_effect_y = coop_effects
        self.coop_cost_x, self.coop_cost_y = coop_costs

    def calculate_profits(self, scenario: str) -> tuple[float, float]:
        """Calculate profits for the provided scenario with the given parameters of the model."""
        if scenario == "CC":
            self.retailer_x = Retailer(
                self.unit_cost, self.coop_cost_x, self.coop_effect_x
            )
            self.retailer_y = Retailer(
                self.unit_cost, self.coop_cost_y, self.coop_effect_y
            )
        elif scenario == "CD":
            self.retailer_x = Retailer(
                self.unit_cost, self.coop_cost_x, self.coop_effect_x
            )
            self.retailer_y = Retailer(self.unit_cost, 0.0, self.coop_effect_y)
        elif scenario == "DC":
            self.retailer_x = Retailer(self.unit_cost, 0.0, self.coop_effect_x)
            self.retailer_y = Retailer(
                self.unit_cost, self.coop_cost_y, self.coop_effect_y
            )
        elif scenario == "DD":
            self.retailer_x = Retailer(self.unit_cost, 0.0, 0.0)
            self.retailer_y = Retailer(self.unit_cost, 0.0, 0.0)
        else:
            raise ValueError(
                f"scenario should be either of: CC, DC, CD, DD. Got: {scenario}"
            )

        self.price_x = self.retailer_x.calculate_optimal_retail_price(self.beta)
        self.price_y = self.retailer_y.calculate_optimal_retail_price(self.beta)
        # print("Prices: ", self.price_x, self.price_y)

        self.demand_x = self.retailer_x.calculate_demand(self.beta, self.price_y)
        self.demand_y = self.retailer_y.calculate_demand(self.beta, self.price_x)
        # print("Demand: ", self.demand_x, self.demand_y)

        self.profit_x = self.retailer_x.calculate_profit()
        self.profit_y = self.retailer_y.calculate_profit()

        return self.profit_x, self.profit_y

    def prepare_payoff_matrix(self) -> tuple[NDArray, NDArray]:
        """Calculate expected profits for each scenario and collect them in a matrix."""
        # print("-------- Both Cooperate --------")
        x11, y11 = self.calculate_profits("CC")
        # print("-------- 1st Cooperate 2nd Deceit--------")
        x12, y12 = self.calculate_profits("CD")
        # print("-------- 1st Deceit 2nd Cooperate--------")
        x21, y21 = self.calculate_profits("DC")
        # print("-------- Both Deceit --------")
        x22, y22 = self.calculate_profits("DD")

        return np.array([x11, x12, x21, x22]), np.array([y11, y12, y21, y22])



if __name__ == "__main__":
    from qrumi.quantum import QuantumDecision, QuantumState

    basic_model = InvestmentModel()
    payoff_x, payoff_y = basic_model.prepare_payoff_matrix()
    # print(payoff_x)
    # print(payoff_y)
    payoff_x = np.array([[3,0],[5,1]])

    coop = QuantumDecision(0, 0)
    defect = QuantumDecision(np.pi, 0)
    quantum = QuantumDecision(0, np.pi / 2.0)
    decs = [coop, defect, quantum]
    labels = ["Coop", "Defect", "Quantum"]
    for dec_x, lab_x in zip(decs, labels):
        for dec_y, lab_y in zip(decs, labels):
            qstate = QuantumState(np.pi / 2.0, [dec_x, dec_y])
            qstate.calculate_quantum_state()
            payoff_c = qstate.calculate_expected_payoff(payoff_x)
            print(f"{lab_x}-{lab_y} payoff:", payoff_c)

    # print("Total payoff:", payoff_q + payoff_c + payoff_d)
