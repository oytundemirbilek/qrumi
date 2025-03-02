"""_summary_"""

import numpy as np


class CournotModel:
    """"""

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

    def verify_dilemma(self) -> bool:
        """_summary_

        Returns
        -------
        bool
            _description_
        """
        return True


class Firm:
    """_summary_"""

    def __init__(self, cournot_model: CournotModel) -> None:
        """_summary_

        Parameters
        ----------
        baseline_demand : float
            _description_
        baseline_cost : float
            _description_
        """
        self.model = cournot_model

    def get_demand_boost(self, theta: float, theta_other: float) -> float:
        """"""
        return self.model.demand_boost * (
            np.cos(theta / 2) ** 2 + np.cos(theta_other / 2) ** 2
        )

    def get_cost_increase(self, theta: float) -> float:
        """"""
        return self.model.cost_increase * np.cos(theta / 2) ** 2

    def calculate_best_quantity(self, theta: float, theta_other: float) -> float:
        """"""
        denom = 3 * self.model.demand_slope
        nom = (
            self.model.baseline_demand
            + self.get_demand_boost(theta, theta_other)
            - self.model.baseline_cost
            - self.get_cost_increase(theta)
        )
        return nom / denom

    def calculate_best_profit(self, theta: float, theta_other: float) -> float:
        """"""
        denom = 9 * self.model.demand_slope
        nom = (
            self.model.baseline_demand
            + self.get_demand_boost(theta, theta_other)
            - self.model.baseline_cost
            - self.get_cost_increase(theta)
        ) ** 2
        return nom / denom


class CournotDuopoly:
    """_summary_"""

    def __init__(self, cournot_model: CournotModel | None = None) -> None:
        """_summary_"""
        self.cournot_model = cournot_model
        if self.cournot_model is None:
            self.cournot_model = CournotModel(100, 20, 1.5, 20, 30)
        self.coop_theta = 0.0
        self.defect_theta = np.pi
        self.firm1 = Firm(self.cournot_model)
        self.firm2 = Firm(self.cournot_model)

    def calculate_payoff_matrix(self, firm: str = "1") -> np.ndarray:
        """_summary_

        Returns
        -------
        np.ndarray
            _description_
        """
        cc = self.firm1.calculate_best_profit(self.coop_theta, self.coop_theta)
        cd = self.firm1.calculate_best_profit(self.coop_theta, self.defect_theta)
        dc = self.firm1.calculate_best_profit(self.defect_theta, self.coop_theta)
        dd = self.firm1.calculate_best_profit(self.defect_theta, self.defect_theta)
        return np.array([[cc, cd], [dc, dd]])

    def verify_dilemma(self) -> bool:
        """_summary_

        Returns
        -------
        bool
            _description_
        """
        return True
