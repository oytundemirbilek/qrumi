""""""

import matplotlib.ticker as tck
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize

from qrumi.cournot import CournotDuopoly, CournotModel
from qrumi.quantum import QuantumDecision, QuantumState


class Experiment:
    """"""

    def __init__(self, gamma: float = np.pi / 2.0) -> None:
        self.gamma = gamma
        self.gamma_default = gamma
        self.decisions_x = self.define_decisions()
        self.decisions_y = self.define_decisions()
        self.cournot = CournotDuopoly()

    def define_decisions(self, classical: bool = False) -> dict[str, QuantumDecision]:
        """"""
        main_decisions = {
            "C": QuantumDecision(0, 0),
            "D": QuantumDecision(np.pi, 0),
        }
        if not classical:
            main_decisions.update({"Q": QuantumDecision(0, np.pi / 2.0)})
        return main_decisions

    def calculate_min_expected_outcome(self) -> float:
        """"""
        res_payoffs = self.calculate_outcome_matrix()
        return float(np.max(np.min(res_payoffs, axis=0)))

    def calculate_outcome_matrix(self) -> np.ndarray:
        """_summary_

        Returns
        -------
        np.ndarray
            _description_
        """
        payoff_mat = self.cournot.calculate_payoff_matrix()
        print(payoff_mat)
        # payoff_mat = np.array([[3, 0], [5, 1]])

        res_mat = []
        for key_x, dec_x in self.decisions_x.items():
            res = []
            for key_y, dec_y in self.decisions_y.items():
                qstate = QuantumState(self.gamma, [dec_x, dec_y])
                qstate.calculate_quantum_state()
                payoff = qstate.calculate_expected_payoff(payoff_mat)
                res.append(payoff)
            res_mat.append(res)

        return np.array(res_mat).real

    def experiment_gammas(
        self, gammas: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """"""
        if gammas is None:
            gammas = np.arange(0, (np.pi / 2) + 0.001, np.pi / 60)

        results = []
        for gamma in gammas:
            self.gamma = gamma
            min_pay = exp.calculate_min_expected_outcome()
            results.append(min_pay)
        return np.array(results), gammas

    def experiment_cournot(self):
        """_summary_"""
        # CournotModel(100, 20, 1.5, 20, 30)
        self.cournot = CournotDuopoly()
        results, gammas = self.experiment_gammas()
        self.visualize_gamma_experiment(results, gammas)

        cournot_model = CournotModel(100, 20, 1.5, 29, 30)
        self.cournot = CournotDuopoly(cournot_model)
        results, gammas = self.experiment_gammas()
        self.visualize_gamma_experiment(results, gammas)

        cournot_model = CournotModel(100, 20, 1.5, 16, 30)
        self.cournot = CournotDuopoly()
        self.experiment_gammas()
        self.visualize_gamma_experiment(results, gammas)

    def visualize_outcome_matrix(
        self, outcomes: np.ndarray, ax: Axes | None = None
    ) -> Axes:
        """"""
        labels = ["Q", "D", "C"]
        ax = sns.heatmap(
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
        if self.gamma != 0.0:
            denom = np.pi / self.gamma
            ax.set_title(rf"X's payoffs, $\gamma=\pi/{denom}$")
        else:
            ax.set_title(rf"X's payoffs, $\gamma=0$")
        ax.set_xlabel("X's decisions")
        ax.set_ylabel("Y's decisions")
        return ax

    def visualize_gamma_experiment(
        self, results: np.ndarray, gammas: np.ndarray
    ) -> None:

        fig, ax = plt.subplots(1, 1)
        # fig.set_size_inches((16, 9))
        sns.lineplot(x=gammas / np.pi, y=results, ax=ax)
        sns.scatterplot(x=gammas / np.pi, y=results, ax=ax)
        # ax.set_title(
        #     "Effect of quantum entanglement\non minimum expected payoff for retailer X"
        # )
        ax.set_xlabel("$\gamma$")
        ax.set_ylabel("Minimum expected payoff")
        ax.xaxis.set_major_formatter(tck.FormatStrFormatter("%g$\pi$"))
        ax.xaxis.set_major_locator(tck.MultipleLocator(base=0.05))
        plt.show()
        plt.tight_layout()
        # plt.savefig("min_payoff.png", dpi=300)
        plt.close()

    def visualize_multiple_outcomes(self) -> None:
        gammas = np.arange(0, (np.pi / 2) + 0.001, np.pi / 4)

        fig, ax = plt.subplots(1, len(gammas))
        # fig.set_size_inches((16, 9))

        for i, gamma in enumerate(gammas):
            self.gamma = gamma
            outcomes = self.calculate_outcome_matrix()
            exp.visualize_outcome_matrix(outcomes, ax[i])
        cbar_ax = fig.colorbar(
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
