import platform as _platform
import numpy as _np

import ctypes as _ctypes
from pathlib import Path as _Path

from quictools.stats import (rand_hermitian as _rand_hermitian,
                             rand_haar_state as _rand_haar_state, 
                             rand_haar_uni as _rand_haar_uni)

_system = _platform.system()
if _system == "Linux":
    p = _Path(__file__).parent.parent/'data'/'libcg.so'
elif _system == "Windows":
   p = _Path(__file__).parent.parent/'data'/'libcg.dll'
else:
    raise RuntimeError(f"Platform <<{_system}>> is not supported")
clebsch_gordan = _ctypes.cdll.LoadLibrary(p.as_posix()).clebsch_gordan
clebsch_gordan.argtypes = [_ctypes.c_double]*6
clebsch_gordan.restype = _ctypes.c_double
# double clebsch_gordan(double j1, double j2, double j, double m1, double m2, double m)

def spin_norm(spin):
    # norm of spin matrix is J*(J+1)*(2*J+1)/3
    return _np.sqrt(spin*(spin+1)*(2*spin+1)/3)

def ang_mom(spin=7.5,convention="Standard"):
    
    # norm of spin matrix is J*(J+1)*(2*J+1)/3
    m = _np.arange(spin-1,-spin-1,-1)
    v = _np.sqrt( spin*(spin+1) - m*(m+1)  )

    jx = (_np.diag(v,1) + _np.diag(v,-1))/2

    if convention=="Standard":
        jy = (_np.diag(v,1) + _np.diag(-v,-1))/(2j)
        jz = _np.diag(_np.arange(spin,-spin-1,-1))
    elif convention=="Reversed":
        jy = (_np.diag(-v,1) + _np.diag(v,-1))/(2j)
        jz = _np.diag(_np.arange(-spin,spin+1,1))
    else:
        raise ValueError("Invalid convention")

    return jx,jy,jz

def scs_from_unit_vector(n,spin=7.5,convention="Standard"):
    # arguments
    #     n (3,1) double {mustBeUnitVector}
    #     options.J (1,1) double {mustBeHalfInteger,mustBeNonnegative} = 7.5
    #     options.convention (1,1) string {mustBeValidConvention} = "Standard"
    # end

    # generates spin coherent state from unit vector n in subspace with angular momentum J
    
    dim = int(2*spin+1)
    
    nn = dim-1 # dimension - 1 (not to be confused with n)
    
    phi = _np.arctan2(n[1],n[0])  # azimuthal angle
    
    if convention=="Standard":
        s = 1
        ind = nn
    elif convention=="Reversed":
        s = -1
        ind = 0
    else:
        raise ValueError("Invalid convention")
    

    scs = _np.zeros((dim,1),dtype=_np.complex128)
    if n[2]==1:
        scs[nn-ind] = 1
        return scs
    if n[2]==-1:
        scs[ind] = 1
        return scs
    

    p = (s*n[2]+1)/2      # stands in for altitude angle
    
    # working in log space so funciton works for large spin
    base = nn*_np.log(p)/2
    step = (_np.log(1-p)-_np.log(p))/2
    
    r = _np.arange(0,nn+1).reshape(-1,1)
    phase = _np.exp(1j*phi*r)
    
    # base + step*r
    # each step multiply by sqrt((1-p)/p)  % multiply by phase later
    
    # if not special case, proceed to calculate binomial
    scs[0] = base
    
    for ii in range(1,nn+1):
        # scs[ii-1] = scs[ii-2] + step + _np.log( (nn-ii+2)/(ii-1) )/2
        scs[ii] = scs[ii-1] + step + _np.log( (nn-ii+1)/ii )/2
    
    scs = _np.exp(scs)*phase
    scs = scs / _np.linalg.norm(scs)
    return scs

def rand_hermitian(spin,ensemble="Unitary"):
    dim = int(2*spin+1)
    return _rand_hermitian(dim,ensemble=ensemble)

def rand_haar_state(spin):
    dim = int(2*spin+1)
    return _rand_haar_state(dim)

def rand_haar_uni(spin=7.5,ensemble="Unitary",domain="Complex"):
    dim = int(2*spin+1)
    return _rand_haar_uni(dim,ensemble=ensemble,domain=domain)
    