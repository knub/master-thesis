import numpy as np
import scipy as sc
import scipy.stats
import scipy.linalg as la

def rW(kappa,m):
    dim = m-1
    b = dim / (np.sqrt(4*kappa*kappa + dim*dim) + 2*kappa)
    x = (1-b) / (1+b)
    c = kappa*x + dim*np.log(1-x*x)

    done = False
    while not done:
        z = sc.stats.beta.rvs(dim/2,dim/2)
        w = (1 - (1+b)*z) / (1 - (1-b)*z)
        u = sc.stats.uniform.rvs()
        if kappa*w + dim*np.log(1-x*w) - c >= np.log(u):
            done = True
    return w

def sample_tangent_unit(mu):
    mat = np.matrix(mu)

    if mat.shape[1]>mat.shape[0]:
        mat = mat.T

    U,_,_ = la.svd(mat)
    nu = np.matrix(np.random.randn(mat.shape[0])).T

    foo1 = U[:,1:]
    foo2 = nu[1:,:]
    print foo1.shape
    print foo2.shape
    x = np.dot(foo1,foo2)
    x = np.reshape(x, 3)
    print x.shape
    res = x/la.norm(x)
    print res.shape
    return res

def rvMF(n, theta):
    dim = len(theta)
    kappa = np.linalg.norm(theta)
    mu = theta / kappa

    result = []
    for sample in range(0,n):
        w = rW(kappa,dim)
        v = sample_tangent_unit(mu)
        # v = np.random.randn(dim)
        # v = v / np.linalg.norm(v)

        res = np.sqrt(1-w**2)*v + w*mu
        result.append(res)

    return result

n = 1
kappa = 1
direction = np.array([1,-1,1])
direction = direction / np.linalg.norm(direction)

res_sampling = rvMF(n, kappa * direction)
for r in res_sampling:
    print r
    print r[0,0] * r[0,0] + r[0,1] * r[0,1] + r[0,2] * r[0,2]
