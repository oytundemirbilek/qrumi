""""""

import matplotlib
from matplotlib.axes import Axes
import matplotlib.colorbar
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize

from qrumi.cournot import InvestmentModel
from qrumi.quantum import QuantumDecision, QuantumState


class Experiment:
    """"""

    def __init__(self, beta: float = 0.5, gamma: float = np.pi / 2.0) -> None:
        self.beta = beta
        self.gamma = gamma
        self.decisions = self.define_decisions()

    def define_decisions(self) -> dict[str, QuantumDecision]:
        """"""
        return {
            "C": QuantumDecision(0, 0),
            "D": QuantumDecision(np.pi, 0),
            "Q": QuantumDecision(0, np.pi / 2.0),
        }

    def calculate_model_per_player(self, player: str = "x") -> np.ndarray:
        """"""
        basic_model = InvestmentModel(self.beta)
        payoff_x, payoff_y = basic_model.prepare_payoff_matrix()
        if player == "x":
            payoff_mat = payoff_x
        elif player == "y":
            payoff_mat = payoff_y
        else:
            raise ValueError("player should be x or y.")
        # payoff_mat = np.array([[3, 0], [5, 1]])

        res_mat = []
        for key_x, dec_x in self.decisions.items():
            res = []
            for key_y, dec_y in self.decisions.items():
                qstate = QuantumState(self.gamma, [dec_x, dec_y])
                qstate.calculate_quantum_state()
                payoff = qstate.calculate_expected_payoff(payoff_mat)
                print(f"{key_x}-{key_y} payoff:", payoff)
                res.append(payoff)
            res_mat.append(res)

        return np.array(res_mat)

    def visualize_decisions(self, fig: Figure, ax: Axes) -> Axes:
        """"""
        labels = ["Q", "D", "C"]
        # print("-------- Player Cooperates --------")
        payoffs = self.calculate_model_per_player().real
        print(payoffs)

        ax = sns.heatmap(
            payoffs,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cmap="RdYlGn",
            annot=True,
            square=True,
            cbar=False,
            vmin=-1,
            vmax=1,
        )
        if self.gamma != 0.0:
            denom = np.pi / self.gamma
            ax.set_title(rf"X's payoffs $\beta={self.beta}, \gamma=\pi/{denom}$")
        else:
            ax.set_title(rf"X's payoffs $\beta={self.beta}, \gamma=0$")
        ax.set_xlabel("X's decisions")
        ax.set_ylabel("Y's decisions")
        return ax

    def run(self) -> None:
        """"""
        self.visualize_decisions()


class ExperimentSet:
    """"""

    def __init__(
        self, gammas: np.ndarray | None = None, betas: np.ndarray | None = None
    ):
        if gammas is None:
            gammas = np.arange(0, (np.pi / 2) + 0.001, np.pi / 4)
        if betas is None:
            betas = np.arange(0, 1, 0.49)
        self.gammas = gammas
        self.betas = betas
        print(self.betas)
        print(self.gammas)

    def calculate_payoffs_per_strategy(self, experiment: Experiment) -> np.ndarray:
        """"""
        payoffs_c = experiment.calculate_model_per_player("x")
        payoffs_d = experiment.calculate_model_per_player("y")
        # payoffs_q = experiment.calculate_model_per_player("Q")
        return np.array([payoffs_c, payoffs_d])

    def visualize_results(self, payoffs: np.ndarray) -> None:
        """"""
        sns.lineplot(payoffs)
        plt.show()

    def experiment_betas(self, gamma: float = np.pi / 2) -> np.ndarray:
        """"""
        results = []
        fig, ax = plt.subplots(1, 3)
        fig.set_size_inches((16, 9))
        fig.set_dpi(300)
        for i, beta in enumerate(self.betas):
            exp = Experiment(beta=beta, gamma=gamma)
            # payoffs = self.calculate_payoffs_per_strategy(exp)
            exp.visualize_decisions(fig, ax[i])
            # results.append(payoffs)
        cbar_ax = fig.colorbar(
            plt.cm.ScalarMappable(cmap="RdYlGn", norm=Normalize(vmin=-1, vmax=1)),
            ax=ax,
            location="right",
            fraction=0.02,
            pad=0.1,
        )
        plt.savefig("./figure.png")
        return np.array(results)

    def experiment_gammas(self, beta: float = 0.5) -> np.ndarray:
        """"""
        results = []
        fig, ax = plt.subplots(1, 3)
        fig.set_size_inches((16, 9))
        fig.set_dpi(300)
        for i, gamma in enumerate(self.gammas):
            exp = Experiment(beta=beta, gamma=gamma)
            # payoffs = self.calculate_payoffs_per_strategy(exp)
            exp.visualize_decisions(fig, ax[i])
            # results.append(payoffs)
        cbar_ax = fig.colorbar(
            plt.cm.ScalarMappable(cmap="RdYlGn", norm=Normalize(vmin=-1, vmax=1)),
            ax=ax,
            location="right",
            fraction=0.02,
            pad=0.1,
        )
        # plt.show()
        plt.savefig("./figure.png")
        return np.array(results)

    def run(self) -> None:
        """"""
        results = self.experiment_betas()
        # self.visualize_results(res)

        # results = self.experiment_gammas()
        # self.visualize_results(res)


if __name__ == "__main__":
    # exp = Experiment(0.99, np.pi / 2)
    # exp.run()
    exp_set = ExperimentSet()
    exp_set.run()
