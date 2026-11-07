import torch 
import math
import torch.nn as nn
import torchkbnufft as tkbn
from Modules.Regularizer import Regularizer
from Modules.recon_ops import IdentityOperator
from Modules.NumSolvers import conjugate_gradient 

class DeepADMM(nn.Module):
    def __init__(self, shape, rho_init=1e-4, maxiter_admm=1, maxiter_cg=20,
                 kernel_size=(3, 3, 3), features=32, num_of_resblocks=6,
                 scale_factor=1):
        super().__init__()
        self.rho = nn.Parameter(torch.tensor(rho_init, dtype=torch.float32))
        self.regularizer = Regularizer(
            shape, kernel_size=kernel_size, features=features,
            num_of_resblocks=num_of_resblocks, scale_factor=scale_factor,
        )
        self.I = IdentityOperator(shape)
        self.maxiter_admm = maxiter_admm
        self.maxiter_cg = maxiter_cg

    def _step(self, x, z, u, Q, b, maxiter_cg):
        x = conjugate_gradient(Q + self.rho * self.I, b=b + self.rho * (z - u),
                               x=x, maxiter=maxiter_cg, disable_progressbar=True)
        z = self.regularizer(x + u)
        u = u + x - z
        return x, z, u

    def forward(self, Q, b, x, z=None, maxiter_admm=None, maxiter_cg=None,
                disable_progressbar=True):
        # warm start: x = data-fitting solution (the CG result), z = g(x)
        n_admm = self.maxiter_admm if maxiter_admm is None else maxiter_admm
        n_cg = self.maxiter_cg if maxiter_cg is None else maxiter_cg
        if z is None:
            z = self.regularizer(x)
        u = torch.zeros(self.I.shape[0], dtype=x.dtype)
        for _ in range(n_admm):
            x, z, u = self._step(x, z, u, Q, b, n_cg)
        return x

def _admm_iter(xs, zs, ops, batch, h, bidx, b_f):
    'one dual-free ADMM iteration (checkpointable); returns NEW lists'
    xs_new = [None] * len(batch)
    for i in range(len(batch)):
        xs_new[i] = conjugate_gradient(ops[i][0] + RHO * I_op, b_f[i] + RHO * zs[i],
                                       xs[i], CG_ITERS, True)
    zin = torch.zeros(F_ALL, n, n, n, dtype=torch.complex64, device=device)
    for f in range(F_ALL):
        if f == h: continue                          # TEMPORAL MASK: held-out slot zero
        i = bidx.get(f)
        v = xs_new[i] if i is not None else x_st[f]  # batch frames / frozen context
        zin[f] = torch.view_as_complex(v.reshape(-1, 2)).reshape(n, n, n)
    y = model(torch.view_as_real(zin.permute(1, 2, 3, 0).reshape(-1)).reshape(-1))
    zall = torch.view_as_complex(y.reshape(-1, 2)).reshape(n, n, n, F_ALL).permute(3, 0, 1, 2)
    zs_new = [None] * len(batch)
    for i in range(len(batch)):
        zs_new[i] = torch.view_as_real(zall[batch[i]].reshape(-1)).reshape(-1)
    return xs_new, zs_new