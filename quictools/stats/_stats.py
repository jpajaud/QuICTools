import numpy as _np
from scipy.special import loggamma as _loggamma

def mean_vector_length(d,sigma=1):
    return sigma*(2**.5)*_np.exp(_loggamma((d+1)/2)-_loggamma(d/2))


def rand_hermitian(dim,ensemble="Unitary"):
    if ensemble=="Unitary":
        H = _np.random.randn(dim,dim)+1j*_np.random.randn(dim,dim)
        H += H.T.conj()
    elif ensemble=="Orthogonal":
        H = _np.random.randn(dim,dim)
        H += H.T
    else:
        raise ValueError("Invalid ensemble")
    return H/2

def rand_haar_state(dim):
    state = _np.random.randn(dim,1)+1j*_np.random.randn(dim,1)
    state /= _np.linalg.norm(state)
    return state

def rand_haar_uni(dim,ensemble="Unitary",domain="Complex"):
    # This algorithm follows the algorithm outlined in 
    #   Diaconis P., Shahshahani M. "The Subgroup Algorithm for Generating Uniform Random Variables" (2009)
    #
    # Draw random Unitary from circular orthogonal, unitary, or symplectic ensembles
    #

    if domain=="Complex":
        dtype = _np.complex128
    elif domain=="Real":
        dtype = _np.float64
    else:
        raise ValueError("Invalid domain")

    U = _np.eye(dim,dtype=dtype)

    for ii in range(2,dim+1):

        v = _np.random.randn(ii,1).astype(dtype)
        if domain=="Complex":
            v += 1j*_np.random.randn(ii,1)

        # Householder transform
        v /= _np.linalg.norm(v)
        phi = v[-1]/_np.abs(v[-1])
        v *= phi.conj()
        v[-1] = v[-1]-1 # v is now dx
        v /= _np.linalg.norm(v)

        R = phi*(_np.eye(ii,dtype=dtype)-2*(v@v.T.conj()))

        U[:ii,:ii] = R@U[:ii,:ii]

    if ensemble=="Unitary":
        return U
    elif ensemble=="Orthogonal":
        return U.T@U
    elif ensemble=="Symplectic":
        if dim%2 == 1:
            raise ValueError("Symplectic matrix must have even dimension")
        rot = _np.zeros((dim,dim),dtype=dtype)
        for jj in range(1,dim//2+1):
            rot[2*jj-1,2*jj-2] = 1
            rot[2*jj-2,2*jj-1] = -1
            # rot(2*jj,2*jj-1) = 1;
            # rot(2*jj-1,2*jj) = -1;
        return rot@U.T@rot.T@U
    else:
        ValueError("Invalid ensemble")