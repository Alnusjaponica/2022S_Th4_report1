import copy
import numpy as np
from . import backtracking as bt
import abc
from .problem import problem


class iterative_method(abc.ABC):
    """反復法の枠組みとなる抽象クラス

    Attributes:
        prob(problem): 解きたい問題
    """

    def __init__(self, prob: problem) -> None:
        self.prob = prob

    def init_param(self, x: np.ndarray):
        pass

    def clear_param(self):
        pass

    @abc.abstractmethod
    def update(self, x: np.ndarray):
        pass

    def criteria_grad(self, x: np.ndarray, f_opt: float) -> float:
        return np.linalg.norm(self.prob.grad(x))

    def criteria_f(self, x: np.ndarray, f_opt: float) -> float:
        return np.linalg.norm(self.prob.f(x) - f_opt)

    def solve(
        self,
        x0: np.ndarray,
        max_itr: int = 1000000,
        eps: float = 1e-4,
        criteria: str = "grad",
    ) -> list[np.ndarray]:
        """最適解を計算

        与えられた初期解から反復法によって最適解を計算する関数

        Args:
            x0 (np.array): 初期解
            max_itr(int, optional): 最大反復回数 (default=300000)
            eps(float, optional, default=1e-4), criteria(str, default="grad"):
                停止条件に関するパラメータ
                - criteria="grad"の場合 |∇f(w)| < epsで停止。
                - criteria="f"の場合 |f(w)| < epsで停止

        Returns:
            np.array: 停止条件を満たすproblemの近似解
        """
        x = copy.deepcopy(x0)
        if criteria == "f":
            judge = self.criteria_f
            f_opt = self.prob.f(-np.linalg.solve(self.prob.P, self.prob.q))
        elif criteria == "grad":
            judge = self.criteria_grad
            f_opt = float("inf")
        self.init_param(x)
        ret = []
        for _ in range(max_itr):
            ret.append(x)
            x = self.update(x)
            if judge(x, f_opt) < eps:
                ret.append(x)
                self.clear_param()
                return ret
        self.clear_param()
        assert False, "収束しませんでした"


class SD_fixed(iterative_method):
    """固定ステップ幅の最急降下法

    Attributes:
        prob(problem): 解きたい問題
        alpha(float): ステップ幅
    """

    def __init__(self, prob: problem, alpha: float) -> None:
        super().__init__(prob)
        self.alpha = alpha

    def update(self, x: np.ndarray) -> np.ndarray:
        return x - self.alpha * self.prob.grad(x)


class SD_LSearch(iterative_method):
    """backtracking法による直線探索付きの最急降下法

    Almijoの条件を満たす点をbacktracking法をもちいて直線探索する最急降下法

    Attributes:
        prob(problem): 解きたい問題
        param(tuple): Almijoの条件に基づくbacktracking法のパラメータ(alpha: float, c: float, rho: float)
    """

    def __init__(self, prob: problem, param) -> None:
        super().__init__(prob)
        self.bt_engine = bt.backtracking(prob, param)

    def update(self, x: np.ndarray) -> np.ndarray:
        return self.bt_engine.next_x(x)


class Nesterov(iterative_method):
    """Nesterovの加速法

    Attributes:
        prob(problem): 解きたい問題
        alpha(float):
    """

    def __init__(self, prob: problem, alpha: float) -> None:
        super().__init__(prob)
        self.alpha = alpha
        self.y = None
        self.i = None

    def init_param(self, x: np.ndarray) -> None:
        self.y = copy.deepcopy(x)
        self.i = 0

    def clear_param(self) -> None:
        self.y = None
        self.i = None

    def update(self, x: np.ndarray) -> np.ndarray:
        x_ = self.y - self.alpha * self.prob.grad(self.y)
        self.i += 1
        self.beta = self.i / (self.i + 3)
        self.y = x_ + self.beta * (x_ - x)
        return x_


class Polyak(iterative_method):
    """Polyakの加速法

    Attributes:
        prob(problem): 解きたい問題
        alpha(float):
        beta(float):
    """

    def __init__(self, prob: problem, alpha: float, beta: float) -> None:
        super().__init__(prob)
        self.alpha = alpha
        self.beta = beta
        self.x_ = None

    def init_param(self, x: np.ndarray) -> None:
        self.x_ = np.zeros_like(x)

    def clear_param(self) -> None:
        self.x_ = None

    def update(self, x: np.ndarray) -> np.ndarray:
        ret = x - self.alpha * self.prob.grad(x) + self.beta * (x - self.x_)
        self.x_ = copy.deepcopy(x)
        return ret
