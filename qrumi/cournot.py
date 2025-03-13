"""Module to simulate Cournot duopoly competition model."""

import warnings

import numpy as np


class CournotParameters:
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
            warnings.warn(
                "These parameters are not going to cause a Prisoner's Dilemma.",
                stacklevel=1,
            )

    def __str__(self) -> str:
        """Return string representation of the class."""
        return str(self.__dict__)

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

    def __init__(self, cournot_params: CournotParameters) -> None:
        self.params = cournot_params

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
        return self.params.demand_boost * (
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
        return self.params.cost_increase * np.cos(theta / 2) ** 2

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
        denom = 3 * self.params.demand_slope
        nom = (
            self.params.baseline_demand
            + self.get_demand_boost(theta, theta_other)
            - self.params.baseline_cost
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
        denom = 9 * self.params.demand_slope
        nom = (
            self.params.baseline_demand
            + self.get_demand_boost(theta, theta_other)
            - self.params.baseline_cost
            - self.get_cost_increase(theta)
        ) ** 2
        return nom / denom

    def calculate_market_price(self, theta: float, theta_other: float) -> float:
        """Calculate the market price for the Cournot system.

        Parameters
        ----------
        theta : float
            An angle represents the firm's decision in quantum game.
        theta_other : float
            An angle represents the other firm's decision in quantum game.

        Returns
        -------
        float
            Calculated market price for the Cournot system.
        """
        return (
            self.params.baseline_demand
            + self.get_demand_boost(theta, theta_other)
            - self.params.demand_slope
            * (
                self.calculate_best_quantity(theta, theta_other)
                + self.calculate_best_quantity(theta_other, theta)
            )
        )


class CournotDuopoly:
    """Class to represent a Cournot system with two firms."""

    def __init__(self, cournot_params: CournotParameters | None = None) -> None:
        self.cournot_params = cournot_params
        if self.cournot_params is None:
            self.cournot_params = CournotParameters(100, 20, 1.5, 20, 30)
        self.coop_theta = 0.0
        self.defect_theta = np.pi
        self.firm1 = Firm(self.cournot_params)
        self.firm2 = Firm(self.cournot_params)

        if not self.verify_quantites():
            warnings.warn("These quantites are not valid.", stacklevel=1)

        if not self.verify_prices():
            warnings.warn("These prices are not valid.", stacklevel=1)

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
        if not self.verify_dilemma(payoff_mat):
            warnings.warn("These payoffs are not a Prisoner's Dilemma.", stacklevel=1)
        return payoff_mat

    def verify_quantites(self) -> bool:
        """Check whether the best response quantity is valid (>0) for each scenario.

        Returns
        -------
        bool
        """
        cc = self.firm1.calculate_best_quantity(self.coop_theta, self.coop_theta)
        cd = self.firm1.calculate_best_quantity(self.coop_theta, self.defect_theta)
        dc = self.firm1.calculate_best_quantity(self.defect_theta, self.coop_theta)
        dd = self.firm1.calculate_best_quantity(self.defect_theta, self.defect_theta)
        return cc > 0 and cd > 0 and dc > 0 and dd > 0

    def verify_prices(self) -> bool:
        """Check whether the market price is valid (>0) for each scenario.

        Returns
        -------
        bool
        """
        cc = self.firm1.calculate_market_price(self.coop_theta, self.coop_theta)
        cd = self.firm1.calculate_market_price(self.coop_theta, self.defect_theta)
        dc = self.firm1.calculate_market_price(self.defect_theta, self.coop_theta)
        dd = self.firm1.calculate_market_price(self.defect_theta, self.defect_theta)
        return cc > 0 and cd > 0 and dc > 0 and dd > 0

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
