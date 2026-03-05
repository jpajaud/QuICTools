import numpy as _np
from scipy.linalg import expm as _expm
from quictools.spins import ang_mom as _ang_mom

# class kicked_pSpin:

#     def __init__(self,α,Λ,p,spin=7.5,lin='x',nonlin='z',convention="Standard"):
#         self.


def kicked_pSpin_U(α,Λ,p,spin=7.5,lin='x',nonlin='z',convention="Standard"):
    
    [jx,jy,jz] = _ang_mom(spin=spin,convention=convention)
    ops = {'x':jx,'y':jy,'z':jz}
    
    lin_op = ops[lin]
    nonlin_op = ops[nonlin]
    return _expm(1j*α*lin_op)@_expm(1j*Λ*(_np.linalg.matrix_power(nonlin_op,p))/(p*(spin**(p-1))))

def __perm_parity(perm):
    # TODO can return 0 if duplicate arguments
    N = len(perm)
    perm_check = [1 for i in range(N)]
    for i in range(N):
        perm_check[perm[i]] = 0
    if sum(perm_check)>0:
        return 0
    i = perm[0]
    s = 1
    while i!=0:
        i = perm[i]
        s *= -1
    return s

def APM(α,Λ,p,N=100,steps=100,lin='x',nonlin='z'):
    # this will return the orbits as and array of 3-vectors for chaotic and nonchaotic starts

    # starting points
    points = _np.zeros((N,3))
    phi = _np.pi * (3. - _np.sqrt(5.))  # golden angle in radians
    for i in range(N):
        y = 1 - (i / float(N - 1)) * 2  # y goes from 1 to -1
        radius = _np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = _np.cos(theta) * radius
        z = _np.sin(theta) * radius
        points[i] = _np.array([x, y, z])

    alt_ind = {0,1,2}
    inds = {'x':0,'y':1,'z':2}
    i_lin = inds[lin]
    jk_lin = list(alt_ind-{i_lin})
    i_nonlin = inds[nonlin]
    jk_nonlin = list(alt_ind-{i_nonlin})


    # points 
    sin_α = _np.sin(α)
    cos_α = _np.cos(α)
    
    # populate orbits
    orbits = _np.zeros((3,steps,N))
    orbits[:,0,:] = points.T
    s1 = __perm_parity((i_nonlin,*jk_nonlin))
    s2 = __perm_parity((i_lin,*jk_lin))

    for i in range(steps-1):

        # first rotate about nonlin by Λ*orbits[:,nonlin]
        # then rotate about lin by α
        θ = Λ*orbits[i_nonlin,i,:]**(p-1)
        cos_θ = _np.cos(θ)
        sin_θ = _np.sin(θ)

        # nonlinear rotation
        orbits[jk_nonlin[0],i+1,:] = cos_θ*orbits[jk_nonlin[0],i,:] - s1 * sin_θ*orbits[jk_nonlin[1],i,:]
        orbits[jk_nonlin[1],i+1,:] = cos_θ*orbits[jk_nonlin[1],i,:] + s1 * sin_θ*orbits[jk_nonlin[0],i,:]
        orbits[i_nonlin,i+1,:] = orbits[i_nonlin,i,:]

        # linear rotation
        temp_lin0 = cos_α*orbits[jk_lin[0],i+1,:] - s2 * sin_α*orbits[jk_lin[1],i+1,:]
        temp_lin1 = cos_α*orbits[jk_lin[1],i+1,:] + s2 * sin_α*orbits[jk_lin[0],i+1,:]
        orbits[jk_lin[0],i+1,:] = temp_lin0
        orbits[jk_lin[1],i+1,:] = temp_lin1


        norms = _np.linalg.norm(orbits[:,i+1,:],axis=0).reshape(1,-1)
        orbits[:,i+1,:] /= norms
    
    return orbits

def APM_spherical(α,Λ,p,shape=(20,20),steps=100,lin='x',nonlin='z'):
    # this will return the orbits as and array of 3-vectors for chaotic and nonchaotic starts

    # starting points

    assert type(shape)==tuple, 'Shape must be tuple for spherical states'
    φ = _np.linspace(0,2*_np.pi,shape[0])
    θ = _np.linspace(0.001,_np.pi-.001,shape[1])
    φs,θs = _np.meshgrid(φ,θ)
        
    points = _np.zeros((3,shape[1],shape[0]))
    points[0,:,:] = _np.sin(θs)*_np.cos(φs)
    points[1,:,:] = _np.sin(θs)*_np.sin(φs)
    points[2,:,:] = _np.cos(θs)

    alt_ind = {0,1,2}
    inds = {'x':0,'y':1,'z':2}
    i_lin = inds[lin]
    jk_lin = list(alt_ind-{i_lin})
    i_nonlin = inds[nonlin]
    jk_nonlin = list(alt_ind-{i_nonlin})


    # points 
    sin_α = _np.sin(α)
    cos_α = _np.cos(α)
    
    # populate orbits
    orbits = _np.zeros((3,steps,shape[1],shape[0]))
    orbits[:,0,:,:] = points
    s1 = __perm_parity((i_nonlin,*jk_nonlin))
    s2 = __perm_parity((i_lin,*jk_lin))

    for i in range(steps-1):

        # first rotate about nonlin by Λ*orbits[:,nonlin]
        # then rotate about lin by α
        θ = Λ*orbits[i_nonlin,i,:,:]**(p-1)
        cos_θ = _np.cos(θ)
        sin_θ = _np.sin(θ)

        # nonlinear rotation
        orbits[jk_nonlin[0],i+1,:,:] = cos_θ*orbits[jk_nonlin[0],i,:,:] - s1 * sin_θ*orbits[jk_nonlin[1],i,:,:]
        orbits[jk_nonlin[1],i+1,:,:] = cos_θ*orbits[jk_nonlin[1],i,:,:] + s1 * sin_θ*orbits[jk_nonlin[0],i,:,:]
        orbits[i_nonlin,i+1,:,:] = orbits[i_nonlin,i,:,:]

        # linear rotation
        temp_lin0 = cos_α*orbits[jk_lin[0],i+1,:,:] - s2 * sin_α*orbits[jk_lin[1],i+1,:,:]
        temp_lin1 = cos_α*orbits[jk_lin[1],i+1,:,:] + s2 * sin_α*orbits[jk_lin[0],i+1,:,:]
        orbits[jk_lin[0],i+1,:,:] = temp_lin0
        orbits[jk_lin[1],i+1,:,:] = temp_lin1


        # norms = _np.linalg.norm(orbits[:,i+1,:,:],axis=0).reshape(1,-1)
        # orbits[:,i+1,:] /= norms
    
    return orbits

def APM_single(α,Λ,p,x0,steps=100,lin='x',nonlin='z'):
    # this will return the orbits as and array of 3-vectors for single starting point

    
    alt_ind = {0,1,2}
    inds = {'x':0,'y':1,'z':2}
    i_lin = inds[lin]
    jk_lin = list(alt_ind-{i_lin})
    i_nonlin = inds[nonlin]
    jk_nonlin = list(alt_ind-{i_nonlin})


    # points 
    sin_α = _np.sin(α)
    cos_α = _np.cos(α)
    
    # populate orbits
    orbit = _np.zeros((3,steps))
    orbit[:,0] = x0
    s1 = __perm_parity((i_nonlin,*jk_nonlin))
    s2 = __perm_parity((i_lin,*jk_lin))

    for i in range(steps-1):

        # first rotate about nonlin by Λ*orbit[:,nonlin]
        # then rotate about lin by α
        θ = Λ*orbit[i_nonlin,i]**(p-1)
        cos_θ = _np.cos(θ)
        sin_θ = _np.sin(θ)

        # nonlinear rotation
        orbit[jk_nonlin[0],i+1] = cos_θ*orbit[jk_nonlin[0],i] - s1 * sin_θ*orbit[jk_nonlin[1],i]
        orbit[jk_nonlin[1],i+1] = cos_θ*orbit[jk_nonlin[1],i] + s1 * sin_θ*orbit[jk_nonlin[0],i]
        orbit[i_nonlin,i+1] = orbit[i_nonlin,i]

        # linear rotation
        temp_lin0 = cos_α*orbit[jk_lin[0],i+1] - s2 * sin_α*orbit[jk_lin[1],i+1]
        temp_lin1 = cos_α*orbit[jk_lin[1],i+1] + s2 * sin_α*orbit[jk_lin[0],i+1]
        orbit[jk_lin[0],i+1] = temp_lin0
        orbit[jk_lin[1],i+1] = temp_lin1


        norm = _np.linalg.norm(orbit[:,i+1])
        orbit[:,i+1] /= norm
    
    return orbit

