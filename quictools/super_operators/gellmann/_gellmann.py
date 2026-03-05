import numpy as _np

def gen_basis(d):
    #
    # gellmann.gen_basis(dim)
    # Generates basis vectors of Gell-Mann basis as array of orthogonal
    # Hermitian matrices
    #    
    # Arguments:
    #     dim   : dimension of Hilbert space
    # Output:
    #     basis : 3d array of basis elements of shape (dim^2,dim,dim) 
    #                    (output from gellmann.gen_basis(dim))
    # 
    basis = _np.zeros((d*d,d,d),dtype=_np.complex128)
    basis[0,:,:] = _np.eye(d)/_np.sqrt(d)
    for i in range(1,d):
        basis[i,:,:] = _np.diag(_np.hstack((_np.ones(i), -i, _np.zeros(d-1-i))))/_np.sqrt((i+1)*i)
    ind = d
    for i in range(1,d):
        for j in range(i):
            basis[ind,i,j] = 1/_np.sqrt(2)
            basis[ind,j,i] = 1/_np.sqrt(2)
            basis[ind+1,i,j] = 1j/_np.sqrt(2)
            basis[ind+1,j,i] = -1j/_np.sqrt(2)
            ind += 2
    return basis

def gen_basis_super(d):
    #
    # gellmann.gen_basis_super(dim)
    # Generates basis vectors of Gell-Mann basis as array of orthogonal
    # Hermitian matrices
    #    
    # Arguments:
    #     dim   : dimension of Hilbert space
    # Output:
    #     basis : 3d array of basis elements of shape (dim^2,dim^2) 
    #                    (output from gellmann.gen_basis_super(dim))
    # 
    basis = _np.zeros((d*d,d*d),dtype=_np.complex128)
    basis[:,0] = _np.eye(d).flatten(order='F')/_np.sqrt(d)
    for i in range(1,d):
        basis[:,i] = _np.diag(_np.hstack((_np.ones(i), -i, _np.zeros(d-1-i)))).flatten(order='F')/_np.sqrt((i+1)*i)
    ind = d
    for i in range(1,d):
        for j in range(i):
            basis_i = _np.zeros((d,d),dtype=_np.complex128)
            basis_i[i,j] = 1/_np.sqrt(2)
            basis_i[j,i] = 1/_np.sqrt(2)
            basis[:,ind] = basis_i.flatten(order='F')
            basis_i = _np.zeros((d,d),dtype=_np.complex128)
            basis_i[i,j] = 1j/_np.sqrt(2)
            basis_i[j,i] = -1j/_np.sqrt(2)
            basis[:,ind+1] = basis_i.flatten(order='F')
            ind += 2
    return basis

def compose(r,basis=None):
    #
    # gellmann.compose(r,basis)
    # Converts vector in gellmann basis to operator
    #    
    # Arguments:
    #     r     : vector (real for Hermitian matrix)
    #     basis : 3d array of basis elements of shape (dim^2,dim,dim) 
    #                    (output from gellmann.gen_basis(dim))
    # Output:
    #     A     : operator (Hermitian for real vector)
    # 

    d = r.size
    d = _np.sqrt(d)
    assert d//1 == d, 'Dimension is not perfect square'
    d = int(d)
    if basis is None:
        basis = gen_basis(d)
    else:
        _,_,d2 = basis.shape
        assert d2==d,'Dimension of basis does not match input'
    
    return _np.einsum('i,ijk->jk',r,basis)
    
def compose_super(r,basis=None):
    #
    # gellmann.compose_super(r,basis)
    # Converts vector in gellmann basis to operator
    #    
    # Arguments:
    #     r     : vector (real for Hermitian matrix)
    #     basis : matrix of basis elements of shape (dim^2,dim^2)
    #                    (output from gellmann.gen_basis_super(dim))
    # Output:
    #     A     : operator (Hermitian for real vector)
    # 
    d = r.size
    d = _np.sqrt(d)
    assert d//1 == d, 'Dimension is not perfect square'
    d = int(d)
    if basis is None:
        basis = gen_basis_super(d)
    else:
        _,d2 = basis.shape
        assert d2==(d**2),'Dimension of basis does not match input'

    A_vec = basis@r
    return A_vec.reshape(d,d,order='F')

def decompose(A,basis=None):
    #
    # gellmann.decompose(A,basis)
    # Converts operator to vector in gellmann basis
    #    
    # Arguments:
    #     A     : operator (Hermitian matrix for real vector)
    #     basis : 3d array of basis elements of shape (dim^2,dim,dim) 
    #                    (output from gellmann.gen_basis(dim))
    # Output:
    #     r     : vector (real for Hermitian operator)
    # 

    d,d1 = A.shape
    assert d==d1,'Input operator must be square'
    if basis is None:
        basis = gen_basis(d)
    else:
        _,_,d2 = basis.shape
        assert d2==d,'Dimension of basis does not match operator'
    r = _np.zeros((d**2,1),dtype=_np.complex128)
    for i in range(d**2):
        r[i] = (A.T * basis[i]).sum()
    return r

def decompose_super(A,basis=None):
    #
    # gellmann.decompose_super(A,basis)
    # Converts operator to vector in gellmann basis
    #    
    # Arguments:
    #     A     : operator (Hermitian matrix for real vector)
    #     basis : matrix of basis elements of shape (dim^2,dim^2) 
    #                    (output from gellmann.gen_basis_super(dim))
    # Output:
    #     r     : vector (real for Hermitian operator)
    # 

    d,d1 = A.shape # # assume square
    assert d==d1,'Input operator must be square'
    if basis is None:
        basis = gen_basis_super(d)
    else:
        _,d2 = basis.shape
        assert d2==d**2,'Dimension of basis does not match operator'
    
    return basis.T.conj()@A.flatten(order='F')
