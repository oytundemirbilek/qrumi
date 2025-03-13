"""Module to define various experiment setup and run them."""

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import Normalize

from qrumi.cournot import CournotDuopoly, CournotParameters
from qrumi.quantum import QuantumDecision, QuantumState


class Experiment:
    """Class to represent the experimentation of the methods."""

    def __init__(self, gamma: float = np.pi / 2.0) -> None:
        self.gamma = gamma
        self.gamma_default = gamma
        self.decisions_x = self.define_decisions()
        self.decisions_y = self.define_decisions()
        self.cournot = CournotDuopoly()
        self.payoff_mat: np.ndarray | None = None

    @staticmethod
    def define_decisions(classical: bool = False) -> dict[str, QuantumDecision]:
        """Define and return possible decisions for the game in quantum-form.

        Parameters
        ----------
        classical : bool, optional
            Whether to limit decisions with classical-only, by default False

        Returns
        -------
        dict[str, QuantumDecision]
            A dictionary where the key is the short-form of the decision.
        """
        main_decisions = {
            "C": QuantumDecision(0, 0),
            "D": QuantumDecision(np.pi, 0),
        }
        if not classical:
            main_decisions.update({"Q": QuantumDecision(0, np.pi / 2.0)})
        return main_decisions

    def calculate_outcome_matrix(self) -> np.ndarray:
        """Calculate the payoffs from each possible scenario and put them into a matrix.

        Returns
        -------
        np.ndarray
        """
        self.payoff_mat = self.cournot.calculate_payoff_matrix()
        # print(payoff_mat)
        # payoff_mat = np.array([[3, 0], [5, 1]])

        res_mat = []
        for key_x, dec_x in self.decisions_x.items():
            res = []
            for key_y, dec_y in self.decisions_y.items():
                qstate = QuantumState(self.gamma, [dec_x, dec_y])
                qstate.calculate_quantum_state()
                payoff = qstate.calculate_expected_payoff(self.payoff_mat)
                print(f"{key_x} - {key_y}: {payoff}")
                res.append(payoff)
            res_mat.append(res)

        return np.array(res_mat).real

    def calculate_min_expected_outcome(self) -> float:
        """Get the minimum expected outcome from all possible scenarios.

        Returns
        -------
        float
        """
        res_payoffs = self.calculate_outcome_matrix()
        return float(np.max(np.min(res_payoffs, axis=0)))

    def experiment_gammas(self, gammas: np.ndarray | None = None) -> pd.DataFrame:
        """Run an experiment for different gammas to observe quantum entanglement effect.

        Parameters
        ----------
        gammas : np.ndarray | None, optional
            A set of gammas will be used to run the experiment, by default None.

        Returns
        -------
        pd.DataFrame
            Results and gammas.
        """
        if gammas is None:
            gammas = np.arange(0, (np.pi / 2) + 0.001, np.pi / 60)

        results = []
        for gamma in gammas:
            self.gamma = float(gamma)
            print(rf"Gamma: {self.gamma / np.pi} pi")
            min_pay = exp.calculate_min_expected_outcome()
            results.append(min_pay)
        print(self.payoff_mat)
        return pd.DataFrame({"MinimumPayoffs": results, "Gammas": gammas / np.pi})

    def visualize_outcome_matrix(
        self, outcomes: np.ndarray, ax: Axes | None = None
    ) -> Axes:
        """Visualize an outcome matrix as a heatmap.

        Parameters
        ----------
        outcomes : np.ndarray
            The input outcomes in matrix form.
        ax : Axes | None, optional
            Add the plot into the provided axes, by default None

        Returns
        -------
        Axes
            The axes with the visualization added.
        """
        labels = ["Q", "D", "C"]
        axes = sns.heatmap(
            outcomes,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            linewidths=1.0,
            cmap="RdYlGn",
            annot=True,
            square=True,
            cbar=False,
            vmin=-1000,
            vmax=1000,
            # robust=True,
            fmt="g",
        )
        if self.gamma != 0.0:  # noqa: PLR2004
            denom = np.pi / self.gamma
            axes.set_title(rf"X's payoffs, $\gamma=\pi/{denom}$")
        else:
            axes.set_title(r"X's payoffs, $\gamma=0$")
        axes.set_xlabel("X's decisions")
        axes.set_ylabel("Y's decisions")
        return axes

    @staticmethod
    def visualize_gamma_experiment(
        results: pd.DataFrame,
        title: str | None = None,
        show: bool = True,
    ) -> None:
        """Visualize the results of the gamme experiment.

        Parameters
        ----------
        results : np.ndarray
            Results from the gamma experiment.
        """
        _fig, ax = plt.subplots(1, 1)
        # fig.set_size_inches((16, 9))
        sns.lineplot(results, ax=ax, x="Gammas", y="MinimumPayoffs", hue="Experiment")
        sns.scatterplot(
            results,
            ax=ax,
            x="Gammas",
            y="MinimumPayoffs",
            hue="Experiment",
            legend=False,
        )
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel("Minimum expected payoff")
        ax.xaxis.set_major_formatter(tck.FormatStrFormatter(r"%g$\pi$"))
        ax.xaxis.set_major_locator(tck.MultipleLocator(base=0.05))
        plt.tight_layout()
        if show:
            plt.show()
        elif title is not None:
            plt.savefig(f"figures/{title}.png", dpi=300)
        else:
            plt.savefig("figures/gamma_experiment.png", dpi=300)
        plt.close()

    def visualize_multiple_outcomes(self) -> None:
        """Visualize multiple outcome matrices in the same figure as subplots."""
        gammas = np.arange(0, (np.pi / 2) + 0.001, np.pi / 4)

        fig, ax = plt.subplots(1, len(gammas))
        # fig.set_size_inches((16, 9))

        for i, gamma in enumerate(gammas):
            self.gamma = float(gamma)
            outcomes = self.calculate_outcome_matrix()
            exp.visualize_outcome_matrix(outcomes, ax[i])
        fig.colorbar(
            plt.cm.ScalarMappable(cmap="RdYlGn", norm=Normalize(vmin=-1000, vmax=1000)),
            ax=ax,
            location="right",
            fraction=0.02,
            pad=0.1,
        )
        plt.show()
        # plt.savefig("./figure.png", dpi=300)

    def experiment_demand_slope(self, show: bool = True) -> None:
        """Run an experiment with multiple set of Cournot parameters."""
        all_results = []

        cournot_model = CournotParameters(100, 20, 1.5, 20, 30)
        print(cournot_model)
        self.cournot = CournotDuopoly(cournot_model)
        results = self.experiment_gammas()
        results["Experiment"] = "Default demand slope (b=1.5)"
        all_results.append(results)

        cournot_model = CournotParameters(100, 20, 2.5, 29, 30)
        print(cournot_model)
        self.cournot = CournotDuopoly(cournot_model)
        results = self.experiment_gammas()
        results["Experiment"] = "High demand slope (b=2.5)"
        all_results.append(results)

        cournot_model = CournotParameters(100, 20, 0.75, 16, 30)
        print(cournot_model)
        self.cournot = CournotDuopoly(cournot_model)
        results = self.experiment_gammas()
        results["Experiment"] = "Low demand slope (b=0.75)"
        all_results.append(results)

        result_df = pd.concat(all_results)
        self.visualize_gamma_experiment(result_df, title="demand_slopes", show=show)

    def experiment_demand_boost(self, show: bool = True) -> None:
        """Run an experiment with multiple set of Cournot parameters."""
        all_results = []

        cournot_model = CournotParameters(100, 20, 1.5, 16, 30)
        print(cournot_model)
        self.cournot = CournotDuopoly(cournot_model)
        results = self.experiment_gammas()
        results["Experiment"] = r"Low demand boost ($d_0$=16)"
        all_results.append(results)

        cournot_model = CournotParameters(100, 20, 1.5, 20, 30)
        print(cournot_model)
        self.cournot = CournotDuopoly(cournot_model)
        results = self.experiment_gammas()
        results["Experiment"] = r"Default demand boost ($d_0$=20)"
        all_results.append(results)

        cournot_model = CournotParameters(100, 20, 1.5, 29, 30)
        print(cournot_model)
        self.cournot = CournotDuopoly(cournot_model)
        results = self.experiment_gammas()
        results["Experiment"] = r"High demand boost ($d_0$=29)"
        all_results.append(results)

        result_df = pd.concat(all_results)
        self.visualize_gamma_experiment(result_df, title="demand_boosts", show=show)

    def experiment_baseline_cost(self, show: bool = True) -> None:
        """Run an experiment with multiple set of Cournot parameters."""
        all_results = []

        cournot_model = CournotParameters(100, 20, 1.5, 20, 30)
        print(cournot_model)
        self.cournot = CournotDuopoly(cournot_model)
        results = self.experiment_gammas()
        results["Experiment"] = "Low baseline cost (c=20)"
        all_results.append(results)

        cournot_model = CournotParameters(100, 50, 1.5, 20, 30)
        print(cournot_model)
        self.cournot = CournotDuopoly(cournot_model)
        results = self.experiment_gammas()
        results["Experiment"] = "High baseline cost (c=50)"
        all_results.append(results)

        cournot_model = CournotParameters(100, 80, 1.5, 20, 30)
        print(cournot_model)
        self.cournot = CournotDuopoly(cournot_model)
        results = self.experiment_gammas()
        results["Experiment"] = "Extreme baseline cost (c=80)"
        all_results.append(results)

        result_df = pd.concat(all_results)
        self.visualize_gamma_experiment(result_df, title="baseline_costs", show=show)

    def experiment_cournot(self, show: bool = True) -> None:
        """Run an experiment with multiple set of Cournot parameters."""
        print("Demand slope experiment")
        self.experiment_demand_slope(show)
        print("Demand boost experiment")
        self.experiment_demand_boost(show)
        print("Baseline cost experiment")
        self.experiment_baseline_cost(show)


if __name__ == "__main__":
    exp = Experiment(np.pi / 2)
    exp.experiment_cournot(False)
