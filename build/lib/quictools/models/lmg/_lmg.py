import numpy as _np
from scipy.linalg import expm as _expm
from quictools.spins import ang_mom as _ang_mom


def trot_LMG_U(s,τ,spin=7.5,lin='z',nonlin='x',convention="Standard"):
   
    [jx,jy,jz] = _ang_mom(spin=spin,convention=convention)
    ops = {'x':jx,'y':jy,'z':jz}
    
    lin_op = ops[lin]
    nonlin_op = ops[nonlin]

    return _expm(1j*(1-s)*τ*lin_op)@_expm(1j*s*τ*(nonlin_op@nonlin_op)/(2*spin))

def LMG_U(s,τ,spin=7.5,lin='z',nonlin='x',convention="Standard"):
   
    [jx,jy,jz] = _ang_mom(spin=spin,convention=convention)
    ops = {'x':jx,'y':jy,'z':jz}
    
    lin_op = ops[lin]
    nonlin_op = ops[nonlin]

    return _expm( 1j*(1-s)*τ*lin_op + 1j*s*τ*(nonlin_op@nonlin_op)/(2*spin) )

def LMG_H(s,spin=7.5,lin='z',nonlin='x',convention="Standard"):
   
    [jx,jy,jz] = _ang_mom(spin=spin,convention=convention)
    ops = {'x':jx,'y':jy,'z':jz}
    
    lin_op = ops[lin]
    nonlin_op = ops[nonlin]

    return 1j*(1-s)*lin_op + 1j*s*(nonlin_op@nonlin_op)/(2*spin)
