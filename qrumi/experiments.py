"""Module to define various experiment setup and run them."""

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import Normalize

from qrumi.cournot import CournotDuopoly, CournotModel
from qrumi.quantum import QuantumDecision, QuantumState


class Experiment:
    """Class to represent the experimentation of the methods."""

    def __init__(self, gamma: float = np.pi / 2.0) -> None:
        self.gamma = gamma
        self.gamma_default = gamma
        self.decisions_x = self.define_decisions()
        self.decisions_y = self.define_decisions()
        self.cournot = CournotDuopoly()

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
        payoff_mat = self.cournot.calculate_payoff_matrix()
        # print(payoff_mat)
        # payoff_mat = np.array([[3, 0], [5, 1]])

        res_mat = []
        for _key_x, dec_x in self.decisions_x.items():
            res = []
            for _key_y, dec_y in self.decisions_y.items():
                qstate = QuantumState(self.gamma, [dec_x, dec_y])
                qstate.calculate_quantum_state()
                payoff = qstate.calculate_expected_payoff(payoff_mat)
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

    def experiment_gammas(
        self, gammas: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run an experiment for different gammas to observe quantum entanglement effect.

        Parameters
        ----------
        gammas : np.ndarray | None, optional
            A set of gammas will be used to run the experiment, by default None.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Results and gammas.
        """
        if gammas is None:
            gammas = np.arange(0, (np.pi / 2) + 0.001, np.pi / 60)

        results = []
        for gamma in gammas:
            self.gamma = gamma
            min_pay = exp.calculate_min_expected_outcome()
            results.append(min_pay)
        return np.array(results), gammas

    def experiment_cournot(self) -> None:
        """Run an experiment with multiple set of Cournot parameters."""
        cournot_model = CournotModel(100, 20, 1.5, 20, 30)
        self.cournot = CournotDuopoly(cournot_model)
        results, gammas = self.experiment_gammas()
        self.visualize_gamma_experiment(results, gammas)

        cournot_model = CournotModel(100, 20, 1.5, 29, 30)
        self.cournot = CournotDuopoly(cournot_model)
        results, gammas = self.experiment_gammas()
        self.visualize_gamma_experiment(results, gammas)

        cournot_model = CournotModel(100, 20, 1.5, 16, 30)
        self.cournot = CournotDuopoly(cournot_model)
        results, gammas = self.experiment_gammas()
        self.visualize_gamma_experiment(results, gammas)

        cournot_model = CournotModel(20, 5, 1.5, 2, 3)
        self.cournot = CournotDuopoly(cournot_model)
        results, gammas = self.experiment_gammas()
        self.visualize_gamma_experiment(results, gammas)

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
    def visualize_gamma_experiment(results: np.ndarray, gammas: np.ndarray) -> None:
        """Visualize the results of the gamme experiment.

        Parameters
        ----------
        results : np.ndarray
            Results from the gamma experiment.
        gammas : np.ndarray
            Gammas from the gamma experiment.
        """
        _fig, ax = plt.subplots(1, 1)
        # fig.set_size_inches((16, 9))
        sns.lineplot(x=gammas / np.pi, y=results, ax=ax)
        sns.scatterplot(x=gammas / np.pi, y=results, ax=ax)
        # ax.set_title(
        #     "Effect of quantum entanglement\non minimum expected payoff for retailer X"
        # )
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel("Minimum expected payoff")
        ax.xaxis.set_major_formatter(tck.FormatStrFormatter(r"%g$\pi$"))
        ax.xaxis.set_major_locator(tck.MultipleLocator(base=0.05))
        plt.show()
        plt.tight_layout()
        # plt.savefig("min_payoff.png", dpi=300)
        plt.close()

    def visualize_multiple_outcomes(self) -> None:
        """Visualize multiple outcome matrices in the same figure as subplots."""
        gammas = np.arange(0, (np.pi / 2) + 0.001, np.pi / 4)

        fig, ax = plt.subplots(1, len(gammas))
        # fig.set_size_inches((16, 9))

        for i, gamma in enumerate(gammas):
            self.gamma = gamma
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


if __name__ == "__main__":
    exp = Experiment(np.pi / 2)
    exp.experiment_cournot()
