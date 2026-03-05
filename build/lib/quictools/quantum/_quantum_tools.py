import platform as _platform
from collections import namedtuple as _namedtuple
from warnings import warn as _warn

import numpy as _np
from scipy.linalg import expm as _expm, sqrtm as _sqrtm
from scipy.stats import unitary_group as _uni
import quictools as _quictools
from quictools.spins import (
    ang_mom as _ang_mom,
    scs_from_unit_vector as _scs_from_unit_vector,
)

import ctypes as _ctypes
from pathlib import Path as _Path

# from importlib.resources import path

_system = _platform.system()
if _system == "Linux":
    p = _Path(__file__).parent.parent / "data" / "libcg.so"
elif _system == "Windows":
    p = _Path(__file__).parent.parent / "data" / "libcg.dll"
else:
    raise RuntimeError(f"Platform <<{_system}>> is not supported")
clebsch_gordan = _ctypes.cdll.LoadLibrary(p.as_posix()).clebsch_gordan
clebsch_gordan.argtypes = [_ctypes.c_double] * 6
clebsch_gordan.restype = _ctypes.c_double
# double clebsch_gordan(double j1, double j2, double j, double m1, double m2, double m)


def small_rot(sd, dim):
    if sd == 0:
        return _np.eye(dim)
    V = _uni.rvs(dim)
    D = _np.diag(_np.exp(1j * _np.random.randn(dim) * sd))
    return V @ D @ (V.T.conj())


def density_fidelity(rho1, rho2):
    d11, d12 = rho1.shape
    d21, d22 = rho2.shape
    assert (
        d11 == d12
    ), f"First density matrix must be square: has dimensions ({d11:d},{d12:d})"
    assert (
        d21 == d22
    ), f"Second density matrix must be square: has dimensions ({d21:d},{d22:d})"
    assert (
        d11 == d22
    ), f"Density matrices have different dimensions: have dimensions DIM(rho1)={d11:d}, DIM(rho2)={d22:d}"
    sq_rho1 = _sqrtm(rho1)
    return _np.abs(_np.trace(_sqrtm(sq_rho1 @ rho2 @ sq_rho1))) ** 2


def evolve(
    state, U, op, complex=False, steps=100, sd_bias=0, sd_err=0, sd_impure=0, n_impure=0
):

    # noise_dict elements n_pure, sd_bias, sd_err, sd_pure
    result = _namedtuple("SimResult", ["expect", "fidelity", "purity"])

    dtype = _np.complex128 if complex else _np.float64
    domain_filter = (lambda x: x) if complex else _np.real

    # op can be list of ops
    if type(op) == _np.ndarray:
        op = [op]
        N_op = 1
    elif type(op) == list:
        N_op = len(op)
    else:
        ValueError("op must be numpy.ndarray or list of numpy.ndarray")

    if (sd_bias == 0) & (sd_err == 0) & (sd_impure == 0) & (n_impure == 0):

        expect = _np.zeros((N_op, steps), dtype=dtype)

        state_temp = state.copy()
        for i in range(steps):
            for j in range(N_op):
                expect[j, i] = domain_filter(state_temp.T.conj() @ op[j] @ state_temp)[
                    0, 0
                ]
            state_temp = U @ state_temp
        return result(expect, _np.ones(steps), _np.ones(steps))

    # TODO error correct dims
    # add exact expectation and domain of operator
    expect = _np.zeros((N_op, steps), dtype=dtype)
    fidelity = _np.zeros(steps)
    purity = _np.zeros(steps)

    dim = U.shape[0]

    U_bias = U @ small_rot(sd_bias, dim)
    rho = state @ state.T.conj()
    state_temp = state.copy()
    for i in range(steps):
        for j in range(N_op):
            expect[j, i] = domain_filter(_np.dot(op[j].flatten().conj(), rho.flatten()))
        fidelity[i] = min(1, density_fidelity(rho, state_temp @ (state_temp.T.conj())))
        purity[i] = min(1, _np.linalg.norm(rho) ** 2)

        rho_temp = _np.zeros_like(rho)
        for _ in range(n_impure):
            U_err = small_rot(sd_impure, dim)
            rho_temp += U_err @ rho @ (U_err.T.conj())
        U_temp = U_bias @ small_rot(sd_err, dim)
        rho = U_temp @ (rho_temp / n_impure) @ (U_temp.T.conj())
        state_temp = U @ state_temp

    return result(expect, fidelity, purity)


def op_expect(op, state, complex=False):
    # domain to allow expectation values of complex objects
    # put state in correct shape or determine if density

    result = (state.T.conj() @ op @ state)[0, 0]
    if complex:
        return result
    else:
        return _np.real(result)
