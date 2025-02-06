import numpy as np
import jax.numpy as jnp
import jax

# use cheap fast matrix-exponential, but is unstable for large norm
# (see sum and square)
@jax.jit
def jexpm(A):
    k = 2
    A = A*(2**(-k))

    xs = [2/3*(1+np.sqrt(177))/88,2/3*(1+np.sqrt(177))/352,2/3,
            (-271+29*np.sqrt(177))/(315*2/3),(11*(-1+np.sqrt(177)))/(1260*2/3),
            (11*(-9+np.sqrt(177)))/(5040*2/3),(89-np.sqrt(177))/(5040*4/9)]
    ys = [1,1,(857-58*np.sqrt(177))/(630)]
    x1,x2,x3,x4,x5,x6,x7 = xs
    y0,y1,y2 = ys
    I = jnp.eye(A.shape[0])
    A2 = A@A
    A4 = A2@(x1*A+x2*A2)
    A8 = (x3*A2+A4)@(x4*I+x5*A+x6*A2+x7*A4)
    eig = I+A+y2*A2+A8
    eig = eig@eig
    return eig@eig

def get_nonz(N):
    v0 = np.zeros((N,N))[np.triu_indices(N)]

    xc = np.concatenate([np.arange(len(v0)), np.arange(N,len(v0))[::-1]])
    skewHerm = np.triu(xc.reshape([N,N]),k=0)
    skewHerm -=skewHerm.conj().T
    nonz = skewHerm[np.triu_indices(N,k=1)].flatten()
    return nonz

def get_nonz_herm(N):
    v0 = np.zeros((N,N))[np.triu_indices(N)]

    xc = np.concatenate([np.arange(len(v0)), np.arange(N,len(v0))[::-1]])
    Heff = np.triu(xc.reshape([N,N]),k=0)
    #Heff += Heff.conj().T
    Heff += np.tril(Heff.conj().T,k=-1)
    nonz = Heff[np.triu_indices(N,k=0)].flatten()
    return nonz
def makeRot(v,N):
    xc = jnp.concatenate([v, v[N:][::-1]])
    skewHerm = jnp.triu(xc.reshape([N,N]),k=0)
    skewHerm -=skewHerm.T
    return jax.scipy.linalg.expm(skewHerm)
def makeRot_herm(v,N):
    xc = jnp.concatenate([v, v[N:][::-1]])
    Heff = jnp.triu(xc.reshape([N,N]),k=0)
    Heff += np.tril(Heff.conj().T,k=-1)
    return jax.scipy.linalg.expm(Heff)
