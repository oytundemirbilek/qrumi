"""Module to simulate Cournot duopoly competition model."""

import numpy as np


class CournotModel:
    """Class to represent the demand curve parameters for the Cournot model."""

    def __init__(
        self,
        baseline_demand: float,
        baseline_cost: float,
        demand_slope: float,
        demand_boost: float,
        cost_increase: float,
    ) -> None:
        self.baseline_demand = baseline_demand  # a
        self.baseline_cost = baseline_cost  # c
        self.demand_slope = demand_slope  # b
        self.demand_boost = demand_boost  # d0
        self.cost_increase = cost_increase  # k0
        if not self.verify_dilemma():
            raise UserWarning(
                "These parameters are not going to cause a Prisoner's Dilemma."
            )

    def verify_dilemma(self) -> bool:
        """Verify if the selected parameters result in a Prisoner's Dilemma.

        Returns
        -------
        bool
        """
        pd = True
        if (
            self.demand_boost > self.cost_increase / 2
            and self.demand_boost < self.cost_increase
        ):
            pd = True
        else:
            return False
        return pd


class Firm:
    """Class to represent a firm and its decisions."""

    def __init__(self, cournot_model: CournotModel) -> None:
        self.model = cournot_model

    def get_demand_boost(self, theta: float, theta_other: float) -> float:
        """Calculate and return the potential demand boost effect.

        Parameters
        ----------
        theta : float
            An angle represents the firm's decision in quantum game.
        theta_other : float
            An angle represents the other firm's decision in quantum game.

        Returns
        -------
        float
            Demand boost when the decision theta and theta_other is made.
        """
        return self.model.demand_boost * (
            np.cos(theta / 2) ** 2 + np.cos(theta_other / 2) ** 2
        )

    def get_cost_increase(self, theta: float) -> float:
        """Calculate and return the potential cost increase.

        Parameters
        ----------
        theta : float
            An angle represents the firm's decision in quantum game.

        Returns
        -------
        float
            Cost increase when the decision theta is made.
        """
        return self.model.cost_increase * np.cos(theta / 2) ** 2

    def calculate_best_quantity(self, theta: float, theta_other: float) -> float:
        """Calculate the best response of how much quantity to produce for the Cournot firm.

        Parameters
        ----------
        theta : float
            An angle represents the firm's decision in quantum game.
        theta_other : float
            An angle represents the other firm's decision in quantum game.

        Returns
        -------
        float
            Quantity to produce when the decision theta and theta_other is made.
        """
        denom = 3 * self.model.demand_slope
        nom = (
            self.model.baseline_demand
            + self.get_demand_boost(theta, theta_other)
            - self.model.baseline_cost
            - self.get_cost_increase(theta)
        )
        return nom / denom

    def calculate_best_profit(self, theta: float, theta_other: float) -> float:
        """Calculate the profit to be made when the best response quantity is produced.

        Parameters
        ----------
        theta : float
            An angle represents the firm's decision in quantum game.
        theta_other : float
            An angle represents the other firm's decision in quantum game.

        Returns
        -------
        float
            Best profit to make when the decision theta and theta_other is made.
        """
        denom = 9 * self.model.demand_slope
        nom = (
            self.model.baseline_demand
            + self.get_demand_boost(theta, theta_other)
            - self.model.baseline_cost
            - self.get_cost_increase(theta)
        ) ** 2
        return nom / denom


class CournotDuopoly:
    """Class to represent a Cournot system with two firms."""

    def __init__(self, cournot_model: CournotModel | None = None) -> None:
        self.cournot_model = cournot_model
        if self.cournot_model is None:
            self.cournot_model = CournotModel(100, 20, 1.5, 20, 30)
        self.coop_theta = 0.0
        self.defect_theta = np.pi
        self.firm1 = Firm(self.cournot_model)
        self.firm2 = Firm(self.cournot_model)

    def calculate_payoff_matrix(self, firm: str = "1") -> np.ndarray:
        """Calculate the payoff matrix for the defined Cournot model.

        Returns
        -------
        np.ndarray
        """
        cc = self.firm1.calculate_best_profit(self.coop_theta, self.coop_theta)
        cd = self.firm1.calculate_best_profit(self.coop_theta, self.defect_theta)
        dc = self.firm1.calculate_best_profit(self.defect_theta, self.coop_theta)
        dd = self.firm1.calculate_best_profit(self.defect_theta, self.defect_theta)
        payoff_mat = np.array([[cc, cd], [dc, dd]])
        self.verify_dilemma(payoff_mat)
        return payoff_mat

    @staticmethod
    def verify_dilemma(payoff_mat: np.ndarray) -> bool:
        """Verify if the given payoff matrix result in a Prisoner's Dilemma.

        Returns
        -------
        bool
        """
        t = payoff_mat[1, 0]
        r = payoff_mat[0, 0]
        p = payoff_mat[1, 1]
        s = payoff_mat[0, 1]
        return t > r > p > s
