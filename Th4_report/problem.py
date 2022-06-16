#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse.linalg as spl  # type: ignore


class problem:
    """問題クラス

    具体的な問題のパラメータを保持し、目的関数の2次までの微分情報を返すクラス

    Attributes:
        P():

    P, q, r: f(w) = 1/2 w^T P w + q^T w + r = |b-Aw|^2
    lam: f(w) = 1/2 w^T P w + q^T w + r = |b-Aw|^2 + lam |w|
    L: ∇f(w)のLipschitz定数
    """

    def __init__(self, A: np.ndarray, b: np.ndarray, lam: float = 0) -> None:
        assert len(b) == A.shape[0], "A, bの次元が対応していません."
        self.n = A.shape[1]
        self.lam = lam
        self.P = 2 * (A.T @ A)
        self.L0, _ = spl.eigsh(self.P, 1, which="LM")
        self.P += 2 * self.lam * np.eye(self.n)
        self.q = -2 * (A.T @ b)
        self.r = b.T @ b

    def get_L(self) -> float:
        return self.L0 + self.lam * 2

    def get_cond(self) -> float:
        Lmin, _ = spl.eigsh(self.P, 1, which="SM")
        return self.get_L() / Lmin

    def set_lambda(self, lam: float) -> None:
        if lam != self.lam:
            self.P += 2 * (lam - self.lam) * np.eye(self.n)
            self.lam = lam

    def f(self, w: np.ndarray) -> np.ndarray:
        return 1 / 2 * w.T @ self.P @ w + self.q.T @ w + self.r

    def grad(self, w: np.ndarray) -> np.ndarray:
        return self.P @ w + self.q

    def hessian(self) -> np.ndarray:
        return self.P
