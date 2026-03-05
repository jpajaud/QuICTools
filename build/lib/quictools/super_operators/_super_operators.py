import numpy as _np

def op2vec(op):
    return op.flatten(order='F').reshape(-1,1)

def op2dvec(op):
    return op.flatten(order='F').reshape(1,-1).conj()

def vec2op(vec):
    d2 = vec.size
    dim = int(_np.sqrt(d2)//1)
    if dim**2 != d2:
        raise ValueError('Vector must have dimension that is perfect square')
    return vec.reshape(dim,dim,order='F')

def dvec2op(dvec):
    d2 = dvec.size
    dim = int(_np.sqrt(d2)//1)
    if dim**2 != d2:
        raise ValueError('Vector must have dimension that is perfect square')
    return dvec.reshape(dim,dim,order='F').conj()

def AxAd(A):
    # A is (dim,dim)
    # M is (dim^2,dim^2)
    dim = A.shape[0]
    M = _np.zeros((dim**2,dim**2),dtype=A.dtype)
    for ii in range(dim**2):
        for jj in range(dim**2):
            M[jj,ii] = A[int(jj//dim),int(ii//dim)].conj()*A[jj%dim,ii%dim]
    return M
