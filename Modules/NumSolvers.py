import math
from typing import Optional, Tuple
import torch
from scipy.sparse.linalg import LinearOperator
from tqdm.auto import tqdm

def joint_soft_thresholding(
    g: torch.Tensor,
    lamda: float,
    num_gradient_dims: int = 2,
) -> torch.Tensor:

    g = torch.reshape(g, (num_gradient_dims, -1))

    r = torch.sqrt(torch.sum(g**2, axis=0))

    r = torch.fmax(r, torch.tensor((lamda,)))

    factor = (r - lamda) / r

    g = factor * g

    return g.ravel()


def admm(
    Q: LinearOperator,
    b: torch.Tensor,
    G: LinearOperator,
    x: torch.Tensor,
    lamda: float,
    rho: float,
    num_gradient_dims: int = 2,
    maxiter_admm: int = 30,
    maxiter_cg: int = 5,
) -> torch.Tensor:

    z = torch.zeros(G.shape[0], dtype=x.dtype)
    u = torch.zeros(G.shape[0], dtype=x.dtype)

    for iteration in tqdm(range(maxiter_admm)):

        x = conjugate_gradient(
            Q + rho * G.H @ G,
            b=b + rho * G.H @ (z - u),
            x=x,
            maxiter=maxiter_cg,
            disable_progressbar=True,
        )

        z = joint_soft_thresholding(
            G @ x + u,
            lamda / rho,
            num_gradient_dims,
        )

        u = u + G @ x - z

    return x


def conjugate_gradient(
    Q: LinearOperator,
    b: torch.Tensor,
    x: torch.Tensor,
    maxiter: int = 10,
    disable_progressbar: bool = False,
) -> torch.Tensor:

    g_prev = torch.ones(x.shape, dtype=x.dtype)
    p_prev = torch.zeros(x.shape, dtype=x.dtype)

    for iteration in tqdm(
        range(maxiter),
        disable=disable_progressbar,
    ):

        g = Q @ x - b

        conjugacy_factor = g @ g / (g_prev @ g_prev)
        p = g + conjugacy_factor * p_prev
        stepsize = g @ g / (p @ (Q @ p))
        x = x - stepsize * p

        g_prev = g.clone()
        p_prev = p.clone()

    return x