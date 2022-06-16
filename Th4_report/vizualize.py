from . import problem as prb
from . import algorithm as alg

import numpy as np

import matplotlib.pyplot as plt  # type: ignore
from matplotlib.ticker import ScalarFormatter  # type: ignore
import japanize_matplotlib  # type: ignore
import seaborn as sns  # type: ignore

sns.set_style("whitegrid")

ls = [0, 1, 5, 10]
fontsize = 40
alpha = 0.8
cm = plt.cm.get_cmap("tab20")
linewidth = 8


def set_ax(ax):
    ax.tick_params(axis="x", labelsize=fontsize // 2)
    ax.tick_params(axis="y", labelsize=fontsize // 2)
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.xaxis.offsetText.set_fontsize(fontsize // 2)
    ax.yaxis.offsetText.set_fontsize(fontsize // 2)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


def set_graph(fontsize=40):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)
    set_ax(ax)
    return fig, ax


def F123(fig_num: int, prob: prb.problem, x0: np.ndarray):
    fig, ax = set_graph()
    for i, l in enumerate(ls):
        prob.set_lambda(l)
        if fig_num == 1:
            ret = alg.SD_fixed(prob, 1 / prob.get_L()).solve(x0)
        elif fig_num == 2:
            ret = alg.SD_LSearch(prob, (1, 0.5, 0.9)).solve(x0)
        elif fig_num == 3:
            ret = alg.Nesterov(prob, 1 / prob.get_L()).solve(x0)
        ax.plot(
            [prob.f(r)[0][0] for r in ret],
            linewidth=linewidth,
            linestyle="solid",
            label=f"$\\lambda={l}$",
            alpha=alpha,
            color=cm(i / 10),
        )

    ax.set_xlabel("$k$", fontsize=fontsize)
    ax.set_ylabel("$f(w_k)$", fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    fig.savefig(f"fig{fig_num}.pdf")
    plt.show()


def F45(fig_num: int, prob: prb.problem, x0: np.ndarray):
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    if fig_num == 4:
        crt = "grad"
        y_ax = "$f(w_k)$"
    if fig_num == 5:
        crt = "f"
        y_ax = "$f(w_k)-f(w^*)$"

    for i in range(2):
        for j in range(2):
            prob.set_lambda(ls[2 * i + j])
            if ls[2 * i + j] > 0:
                f_opt = prob.f(-np.linalg.solve(prob.P, prob.q))[0][0]
            else:
                f_opt = 0

            ax = axes[i, j]
            ret = alg.SD_fixed(prob, 1 / prob.get_L()).solve(x0, criteria=crt)
            ax.plot(
                [prob.f(r)[0][0] - f_opt for r in ret],
                linewidth=linewidth,
                label="Fixed step",
                alpha=alpha,
                color=cm(0),
            )
            ret = alg.Nesterov(prob, 1 / prob.get_L()).solve(x0, criteria=crt)
            m = prob.f(ret[0])[0][0] - f_opt
            f_v = []
            for r in ret:
                m = min(prob.f(r)[0][0] - f_opt, m)
                f_v.append(m)
            ax.plot(
                [v for v in f_v],
                linewidth=linewidth,
                label="Nesterov",
                alpha=alpha,
                color=cm(1 / 10),
            )
            ret = alg.Polyak(prob, 1 / prob.get_L(), 0.5).solve(x0, criteria=crt)
            ax.plot(
                [prob.f(r)[0][0] - f_opt for r in ret],
                linewidth=linewidth,
                label=f"Polyak",
                alpha=alpha,
                color=cm(3 / 10),
            )
            ret = alg.SD_LSearch(prob, (0.5, 0.5, 0.9)).solve(x0, criteria=crt)
            ax.plot(
                [prob.f(r)[0][0] - f_opt for r in ret],
                linewidth=linewidth,
                label="Backtracking",
                alpha=alpha,
                color=cm(2 / 10),
            )
            set_ax(ax)

            ax.set_yscale("log")
            ax.set_xlabel("$k$", fontsize=fontsize - 10)
            ax.set_ylabel(y_ax, fontsize=fontsize - 10)
            ax.set_title(f"$\\lambda={ls[2*i+j]}$", fontsize=fontsize - 10)
            if i == 0 and j == 1:
                ax.legend(fontsize=fontsize - 15, loc="upper right")

    fig.savefig(f"fig{fig_num}.pdf")
    plt.show()


def F6(fig_num: int, prob: prb.problem, x0: np.ndarray):
    fig, ax = set_graph()
    lf = []
    ll = []
    ln = []
    lp = []
    kappa = []

    for i in range(19):
        l = (i + 2) / 2
        prob.set_lambda(l)
        kappa.append(prob.get_cond())
        lf.append(len(alg.SD_fixed(prob, 1 / prob.get_L()).solve(x0)))
        ln.append(len(alg.Nesterov(prob, 1 / prob.get_L()).solve(x0)))
        ll.append(len(alg.SD_LSearch(prob, (0.5, 0.5, 0.9)).solve(x0)))
        lp.append(len(alg.Polyak(prob, 1 / prob.get_L(), 0.5).solve(x0)))

    ax.set_xlabel("Condition Number $\\kappa$", fontsize=fontsize - 10)
    ax.set_ylabel("Iteration Step $k$", fontsize=fontsize - 10)

    ax.plot(
        kappa,
        lf,
        linewidth=linewidth,
        label="Fixed step",
        alpha=alpha,
        color=cm(0),
    )
    ax.plot(
        kappa,
        ln,
        linewidth=linewidth,
        label="Nesterov",
        alpha=alpha,
        color=cm(1 / 10),
    )
    ax.plot(
        kappa,
        lp,
        linewidth=linewidth,
        label="Polyak",
        alpha=alpha,
        color=cm(3 / 10),
    )
    ax.plot(
        kappa,
        ll,
        linewidth=linewidth,
        label="Backtracking",
        alpha=alpha,
        color=cm(2 / 10),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=fontsize)
    fig.savefig(f"fig{fig_num}.pdf")
    plt.show()
