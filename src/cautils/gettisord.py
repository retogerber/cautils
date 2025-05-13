import numba
import numpy as np
import scipy

CONNECTIVITY_ROOK = 1
CONNECTIVITY_QUEEN = 2

@numba.jit(nopython=True, parallel=False, cache=True)
def meshgrid_2D(x, y):
    """
    Create a meshgrid for 2D arrays.

    Example:
    x = np.arange(3)
    y = np.arange(4)
    xx, yy = meshgrid_2D(x, y)
    """
    xx = np.empty(shape=(x.shape[0], y.shape[0]), dtype=x.dtype)
    yy = np.empty(shape=(x.shape[0], y.shape[0]), dtype=y.dtype)
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            xx[i,j] = x[i]
            yy[i,j] = y[j]
    return yy, xx

@numba.jit(nopython=True, parallel=False, cache=True)
def get_offset_indices(d: int = 1) -> np.ndarray:
    """
    Get the offset indices for a 2D grid.

    The offsets are in the range [-d, d] for both dimensions.
    
    Example:
    d = 1
    offset_indices = get_offset_indices(d)
    """
    potential_offsets = meshgrid_2D(np.arange(np.floor(-d),np.ceil(d)+1),np.arange(np.floor(-d),np.ceil(d)+1))
    squares = np.empty_like(potential_offsets[0], dtype=np.uint8)
    for i in range(potential_offsets[0].shape[0]):
        for j in range(potential_offsets[0].shape[1]):
            # don't use the center point
            if potential_offsets[0][i,j]==0 and potential_offsets[1][i,j]==0:
                squares[i,j] = d**2+1
            else:   
                squares[i,j] = potential_offsets[0][i,j]**2 + potential_offsets[1][i,j]**2
    to_keep = squares<=d**2
    offset_indices = np.empty((np.sum(to_keep),2), dtype=np.int32)
    c = 0
    for i in range(potential_offsets[0].shape[0]):
        for j in range(potential_offsets[0].shape[1]):
            if to_keep[i,j]:
                offset_indices[c][0] = potential_offsets[0][i,j]
                offset_indices[c][1] = potential_offsets[1][i,j]
                c += 1
    return offset_indices
# get_offset_indices(1)

@numba.jit(nopython=True, parallel=False, cache=True)
def weight_from_dims(n1, n2, wrap1=False,wrap2=False, offset_indices=None) -> np.ndarray:
    """
    Create a weight matrix for a 2D grid.

    The weight matrix is a square matrix of size n1*n2 x n1*n2.
    The weights are 1 for the neighbors and 0 for the rest.
    The neighbors are defined by the offset indices.
    
    Example:
    n1 = 3
    n2 = 4
    wrap1 = True
    wrap2 = False
    offset_indices = None
    W = weight_from_dims(n1, n2, wrap1, wrap2, offset_indices)
    """
    if offset_indices is None:
        offset_indices = get_offset_indices(1)
    n = n1*n2
    W = np.zeros((n,n), dtype=np.uint8)
    for x1 in range(n1):
        for y1 in range(n2):
            for ox,oy in offset_indices:
                x2 = x1+ox
                y2 = y1+oy
                # print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
                x2w = x2%n1
                y2w = y2%n2
                if x2w != x2 and not wrap1:
                    continue
                if y2w != y2 and not wrap2:
                    continue
                W[x1*n2+y1,x2w*n2+y2w] = 1
                # print(W[0,:].reshape(n1,n2))
    return W

@numba.jit(nopython=True, parallel=False, cache=True)
def subset_weights(W, to_keep):
    return W[to_keep][:, to_keep]

@numba.jit(nopython=True, parallel=False, cache=True)
def compute_G(x,W):
    """
    Compute G statistic for a 2D grid.
    
    Example:
    x = np.random.rand(3,4)
    W = weight_from_dims(3, 4)
    G = compute_G(x, W)
    """
    den = 0
    nume = 0
    x_lin = x.flatten()
    for i in range(len(x_lin)):
        for j in range(len(x_lin)):
            nume += x_lin[i]*x_lin[j]*W[i,j]
            den += x_lin[i]*x_lin[j]
    return nume/den


@numba.jit(nopython=True, parallel=False, cache=True)
def compute_coefficients(n, S1, S2, W2):
    B0 = (n**2 - 3*n + 3)*S1 - n*S2 + 3*W2
    B1 = -((n**2 - n)*S1 - 2*n*S2 + 3*W2)
    B2 = -(2*n*S1 - (n + 3)*S2 + 6*W2)
    B3 = 4*(n - 1)*S1 - 2*(n + 1)*S2 + 8*W2
    B4 = S1 - S2 + W2
    return B0, B1, B2, B3, B4

@numba.jit(nopython=True, parallel=False, cache=True)
def compute_S1(Wmat):
    return np.sum((Wmat+Wmat.T)**2)/2

@numba.jit(nopython=True, parallel=False, cache=True)
def compute_S2(Wmat):
    # np.sum((np.sum(Wmat,axis=0)+np.sum(Wmat,axis=1))**2)
    S2 = 0
    for i in range(Wmat.shape[0]):
        S2 += (np.sum(Wmat[i,:]) + np.sum(Wmat[:,i]))**2
    return S2

@numba.jit(nopython=True, parallel=False, cache=True)
def compute_ms(x):
    m1 = 0
    m2 = 0
    m3 = 0
    m4 = 0
    for xi in x:
        m1 += xi
        m2 += xi**2
        m3 += xi**3
        m4 += xi**4 
    return m1, m2, m3, m4

def compute_EG_2(x, Wmat):
    n = len(x)
    S1 = compute_S1(Wmat)
    S2 = compute_S2(Wmat)
    W2 = np.sum(Wmat)**2

    B0, B1, B2, B3, B4 = compute_coefficients(n, S1, S2, W2)
    m1, m2, m3, m4 = compute_ms(x)

    den = (m1**2-m2)**2*(n*(n-1)*(n-2)*(n-3))
    num = B0*m2**2 + B1*m4 + B2*m1**2*m2 + B3*m1*m3 + B4*m1**4

    if den == 0:
        return np.nan
    E_G_2 = num/den
    return E_G_2

def compute_Gi(x,W):
    """
    Compute Gi statistic for a 2D grid.

    Example:
    x = np.random.rand(3,4)
    W = weight_from_dims(3, 4)
    Gi, Zi, Pi = compute_Gi(x.flatten(), W)
    """
    Gi=np.empty_like(x, dtype=np.float64)
    Zi=np.empty_like(x, dtype=np.float64)
    Pi=np.empty_like(x, dtype=np.float64)
    n=len(x)
    Yi1 = np.sum(x)/(n-1)
    Yi2 = np.sum(x**2)/(n-1)-Yi1**2
    x_sum = np.sum(x)
    for i in range(len(x)):
        Gi[i] = np.sum(W[i,:]*x)/x_sum
        wi = np.sum(W[i,:])
        E_i = wi/(n-1)
        V_i = (wi*(n-1-wi)*Yi2)/((n-1)**2*(n-2)*Yi1**2)
        Zi[i] = (Gi[i]-E_i)/np.sqrt(V_i)
        Pi[i] = 1.0 - scipy.stats.norm.cdf(np.abs(Zi[i]))
    return Gi, Zi, Pi

def compute_Hi(x,W):
    """
    Compute Hi statistic for a 2D grid.

    Example:
    x = np.random.rand(3,4)
    W = weight_from_dims(3, 4)
    Hi, Zi, Pi = compute_Hi(x.flatten(), W)
    """
    w1 = np.sum(W, axis=1)
    w2 = np.sum(W**2,axis=1)
    # Calculate spatial mean
    xlag = np.matmul(W, x)/w1
    # xlag = (np.matmul(W, x)+x)/(rowsum+1)
    xresid = (x - xlag) ** 2
    h1 = np.mean(xresid)
    h2 = np.mean(xresid**2)
    denom = h1 * w1
    Hi = np.matmul(W, xresid) / denom

    n = len(x)
    term1 = 1/(n - 1)
    term2 = (1/denom)**2
    term3 = h2 - h1**2
    term4 = (n * w2) - (w1**2)
    VarHi = term1 * term2 * term3 * term4
    dof = 2/VarHi
    Zi = (2*Hi)/VarHi
    Pi = 1 - scipy.stats.chi2.cdf(Zi, dof)
    return Hi, Zi, Pi


# x = np.random.rand(100,100)

@numba.jit(nopython=True, parallel=False, cache=True)
def man_pad(x):
    xm = np.empty((x.shape[0]+2,x.shape[1]+2), dtype=x.dtype)
    xm[1:-1,1:-1] = x
    xm[1:-1,0] = x[:,0]
    xm[1:-1,-1] = x[:,-1]
    xm[0,1:-1] = x[0,:]
    xm[-1,1:-1] = x[-1,:]
    xm[0,0] = x[0,0]
    xm[0,-1] = x[0,-1]
    xm[-1,0] = x[-1,0]
    xm[-1,-1] = x[-1,-1]
    return xm

@numba.jit(nopython=True, parallel=True, cache=True)
def G_and_H_numba(x, connectivity=CONNECTIVITY_QUEEN):
    n=np.float64((x.shape[0]*x.shape[1]))
    x_sum = np.float64(np.sum(x))
    x2_sum = np.float64(np.sum(x**2))
    Yi1 = x_sum/(n-1)
    Yi2 = x2_sum/(n-1)-Yi1**2

    xm = man_pad(x)
    if connectivity == CONNECTIVITY_QUEEN:
        xm = xm[:-2,1:-1]+xm[1:-1,:-2]+xm[2:,1:-1]+xm[1:-1,2:]+xm[:-2,:-2]+xm[:-2,2:]+xm[2:,:-2]+xm[2:,2:]
    else:
        xm = xm[:-2,1:-1]+xm[1:-1,:-2]+xm[2:,1:-1]+xm[1:-1,2:]
    Gi = xm/x_sum

    w1=4*connectivity
    Ei=w1/(n-1)
    V_i = (w1*(n-1-w1)*Yi2)/((n-1)**2*(n-2)*Yi1**2)
    GZi = (Gi-Ei)/np.sqrt(V_i)

    # because all weights are equal to one
    w2=w1
    xresid = (x - xm/w1) ** 2
    h1 = np.mean(xresid)
    h2 = np.mean(xresid**2)

    denom = h1 * w1
    xresidm = man_pad(xresid)
    if connectivity == CONNECTIVITY_QUEEN:
        xresidm = xresidm[:-2,1:-1]+xresidm[1:-1,:-2]+xresidm[2:,1:-1]+xresidm[1:-1,2:]+xresidm[:-2,:-2]+xresidm[:-2,2:]+xresidm[2:,:-2]+xresidm[2:,2:]
    else:
        xresidm = xresidm[:-2,1:-1]+xresidm[1:-1,:-2]+xresidm[2:,1:-1]+xresidm[1:-1,2:]

    VarHi = 1/(n - 1) * (1/denom)**2 * (h2 - h1**2) * ((n * w2) - (w1**2))
    HZi = (2*xresidm / denom)/VarHi
    return GZi, HZi, VarHi
G_and_H_numba_serial = numba.jit(G_and_H_numba.py_func, nopython=True, parallel=False, cache=True)

# x = np.random.rand(100,100)
# G,H,_ = G_and_H_numba(x)
# G,H,_ = G_and_H_numba_serial(x)

@numba.jit(nopython=True, parallel=False, cache=True)
def G_and_H(x, parallel=None, connectivity=CONNECTIVITY_QUEEN):
    if parallel is None:
        if numba.get_num_threads() > 1 and x.size > 5000:
            parallel = True
        else:
            parallel = False
    if parallel:
        return G_and_H_numba(x, connectivity=connectivity)
    else:
        return G_and_H_numba_serial(x, connectivity=connectivity)

# x = np.random.rand(100,100)
# G,H,VarHi = G_and_H_numba(x)

def H_PV(x, VarHi):
    dof = 2/VarHi
    return 1 - scipy.stats.chi2.cdf(x, dof)
# H_PV(H, VarHi)

def G_PV(x):
    return  1.0 - scipy.stats.norm.cdf(np.abs(x))
# G_PV(G)

# import timeit
# import time
# iterations=10
# times1 = []
# times2 = []
# times3 = []
# # ns = [50, 100,500, 1000,5000, 10000,20000]
# # ns = [1000,5000,10000,20000]
# # ns = [50,100,200,500,750,1000,1500,2000,2500,5000,7500]
# ns = [10]
# for i in range(15):
#     ns.append(np.round(np.sqrt(ns[-1]**2*2)).astype(int))
# ns = np.array(ns)
# # ns = [10,20,30,40,50,75,100,200,500,750,1000,1500,2000,2500,5000,7500]
# for n in ns:
#     tmp_iterations = np.ceil(np.max([1/(n/np.max(ns)),1])).astype(int)*10
#     print(f"n: {n}, iterations: {tmp_iterations}")
#     x = np.random.rand(n,n)
#     print(f"\tnumba serial ")
#     time.sleep(0.5)
#     ts = np.array([timeit.timeit("G_and_H_numba_serial(x)", number=1, globals=globals()) for i in range(tmp_iterations)])
#     print(f"\t\tlen: {len(ts)}, median: {np.median(ts):.6f}")
#     times1.append(np.median(ts))
#     print(f"\tnumba parallel ")
#     time.sleep(0.5)
#     ts = np.array([timeit.timeit("G_and_H_numba(x)", number=1, globals=globals()) for i in range(tmp_iterations)])
#     print(f"\t\tlen: {len(ts)}, median: {np.median(ts):.6f}")
#     times2.append(np.median(ts))
#     print(f"\tnumba variable")
#     time.sleep(0.5)
#     ts = np.array([timeit.timeit("G_and_H(x)", number=1, globals=globals()) for i in range(tmp_iterations)])
#     print(f"\t\tlen: {len(ts)}, median: {np.median(ts):.6f}")
#     times3.append(np.median(ts))

# # tmp_iterations=10000
# # timeit.timeit("G_and_H(x)", number=tmp_iterations, globals=globals())/tmp_iterations
# # timeit.timeit("G_and_H(x)", number=tmp_iterations, globals=globals())/tmp_iterations

# import matplotlib.pyplot as plt
# fig,ax = plt.subplots(1,2,figsize=(10, 5))
# ax[0].plot(ns**2, times1, label='numba serial')
# ax[0].plot(ns**2, times2, label='numba parallel')
# ax[0].plot(ns**2, times3, label='numba variable', c="black")
# ax[0].set_xlabel('n')
# ax[0].set_ylabel('Time (seconds)')
# ax[0].set_title('Time taken to compute G and H')
# ax[0].legend()
# ax[0].set_xscale('log')
# ax[0].set_yscale('log')   
# ax[1].plot(ns**2, np.array(times2)/np.array(times1), label='numba parallel vs. numba serial')
# ax[1].plot(ns**2, np.array(times3)/np.array(times1), label='numba variable vs. numba serial')
# ax[1].plot(ns**2, np.array(times3)/np.array(times2), label='numba variable vs. numba parallel',c="black")
# ax[1].set_xlabel('n')
# ax[1].set_ylabel('Speedup')
# ax[1].set_title('Speedup of numba vs ne')
# ax[1].set_xscale('log')
# ax[1].legend()
# plt.savefig('G_and_H.png')
