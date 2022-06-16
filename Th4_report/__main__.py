import argparse
import numpy as np

from . import problem as prb
from . import vizualize as viz


def gen_prob(m, n):
    A = np.random.randn(m, n)
    b = np.random.randn(m, 1)
    return prb.problem(A, b)


def main():
    np.random.seed(seed=42)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fig",
        help="figure number",
        default=5,
        type=int,
    )
    args = parser.parse_args()
    fig_num = args.fig

    m = 90
    n = 100
    problem_instance = gen_prob(m, n)
    x0 = np.random.randn(n, 1) + 10

    if fig_num in [1, 2, 3]:
        viz.F123(fig_num, problem_instance, x0)

    if fig_num in [4, 5]:
        viz.F45(fig_num, problem_instance, x0)

    if fig_num in [6]:
        viz.F6(fig_num, problem_instance, x0)


if __name__ == "__main__":
    main()
