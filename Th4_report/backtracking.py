import numpy as np
from .problem import problem


class bt_parameter:
    """backtracking法のパラメータ

    backtracking法のパラメータを保持するクラス

    Attribute:
       alpha: initial step
       c: Almijo condition parameter
       rho: backtrack parameter
    """

    def __init__(self, alpha: float, c: float, rho: float) -> None:
        assert all(
            (0 < alpha, 0 < rho < 1, 0 < c < 1)
        ), "0 < alpha, 0 < rho < 1, 0 < c < 1となる必要があります。"
        self.alpha = alpha
        self.c = c
        self.rho = rho


class bt_point:
    """backtracking法の現在の点

    実行中のbacktracking法の現在の点に関する情報を保持し、再計算を防ぐ
    """

    def __init__(self, prob: problem, x: np.ndarray, alpha: float) -> None:
        self.x = x
        self.fx = prob.f(x)
        self.d = -prob.grad(x)
        self.dfd = -self.d.T @ self.d
        self.alpha = alpha


class backtracking:
    """backtracking法"""

    def __init__(self, prob: problem, params: tuple) -> None:
        self.params = bt_parameter(*params)
        self.prob = prob

    def judge_Armijo(self, cur_pt: bt_point) -> bool:
        return (
            self.prob.f(cur_pt.x + cur_pt.alpha * cur_pt.d)
            <= cur_pt.fx + self.params.c * cur_pt.alpha * cur_pt.dfd
        )

    def backtrack(self, cur_pt: bt_point) -> None:
        while not self.judge_Armijo(cur_pt):
            cur_pt.alpha = self.params.rho * cur_pt.alpha

    def next_x(self, x: np.ndarray) -> np.ndarray:
        cur_pt = bt_point(self.prob, x, self.params.alpha)
        self.backtrack(cur_pt)
        return cur_pt.x + cur_pt.alpha * cur_pt.d
