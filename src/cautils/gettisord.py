import numba
import numpy as np
import scipy

CONNECTIVITY_ROOK = 1
CONNECTIVITY_QUEEN = 2

# @numba.jit(nopython=True, parallel=False, cache=False)
# def meshgrid_2D(x, y):
#     """
#     Create a meshgrid for 2D arrays.

#     Example:
#     x = np.arange(3)
#     y = np.arange(4)
#     xx, yy = meshgrid_2D(x, y)
#     """
#     xx = np.empty(shape=(x.shape[0], y.shape[0]), dtype=x.dtype)
#     yy = np.empty(shape=(x.shape[0], y.shape[0]), dtype=y.dtype)
#     for i in range(x.shape[0]):
#         for j in range(y.shape[0]):
#             xx[i,j] = x[i]
#             yy[i,j] = y[j]
#     return yy, xx

# @numba.jit(nopython=True, parallel=False, cache=False)
# def get_offset_indices(d: int = 1) -> np.ndarray:
#     """
#     Get the offset indices for a 2D grid.

#     The offsets are in the range [-d, d] for both dimensions.
    
#     Example:
#     d = 1
#     offset_indices = get_offset_indices(d)
#     """
#     potential_offsets = meshgrid_2D(np.arange(np.floor(-d),np.ceil(d)+1),np.arange(np.floor(-d),np.ceil(d)+1))
#     squares = np.empty_like(potential_offsets[0], dtype=np.uint8)
#     for i in range(potential_offsets[0].shape[0]):
#         for j in range(potential_offsets[0].shape[1]):
#             # don't use the center point
#             if potential_offsets[0][i,j]==0 and potential_offsets[1][i,j]==0:
#                 squares[i,j] = d**2+1
#             else:   
#                 squares[i,j] = potential_offsets[0][i,j]**2 + potential_offsets[1][i,j]**2
#     to_keep = squares<=d**2
#     offset_indices = np.empty((np.sum(to_keep),2), dtype=np.int32)
#     c = 0
#     for i in range(potential_offsets[0].shape[0]):
#         for j in range(potential_offsets[0].shape[1]):
#             if to_keep[i,j]:
#                 offset_indices[c][0] = potential_offsets[0][i,j]
#                 offset_indices[c][1] = potential_offsets[1][i,j]
#                 c += 1
#     return offset_indices
# # get_offset_indices(1)

# @numba.jit(nopython=True, parallel=False, cache=False)
# def weight_from_dims(n1, n2, wrap1=False,wrap2=False, offset_indices=None) -> np.ndarray:
#     """
#     Create a weight matrix for a 2D grid.

#     The weight matrix is a square matrix of size n1*n2 x n1*n2.
#     The weights are 1 for the neighbors and 0 for the rest.
#     The neighbors are defined by the offset indices.
    
#     Example:
#     n1 = 3
#     n2 = 4
#     wrap1 = True
#     wrap2 = False
#     offset_indices = None
#     W = weight_from_dims(n1, n2, wrap1, wrap2, offset_indices)
#     """
#     if offset_indices is None:
#         offset_indices = get_offset_indices(1)
#     n = n1*n2
#     W = np.zeros((n,n), dtype=np.uint8)
#     for x1 in range(n1):
#         for y1 in range(n2):
#             for ox,oy in offset_indices:
#                 x2 = x1+ox
#                 y2 = y1+oy
#                 # print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
#                 x2w = x2%n1
#                 y2w = y2%n2
#                 if x2w != x2 and not wrap1:
#                     continue
#                 if y2w != y2 and not wrap2:
#                     continue
#                 W[x1*n2+y1,x2w*n2+y2w] = 1
#                 # print(W[0,:].reshape(n1,n2))
#     return W

# @numba.jit(nopython=True, parallel=False, cache=False)
# def subset_weights(W, to_keep):
#     return W[to_keep][:, to_keep]

# @numba.jit(nopython=True, parallel=False, cache=False)
# def compute_G(x,W):
#     """
#     Compute G statistic for a 2D grid.
    
#     Example:
#     x = np.random.rand(3,4)
#     W = weight_from_dims(3, 4)
#     G = compute_G(x, W)
#     """
#     den = 0
#     nume = 0
#     x_lin = x.flatten()
#     for i in range(len(x_lin)):
#         for j in range(len(x_lin)):
#             nume += x_lin[i]*x_lin[j]*W[i,j]
#             den += x_lin[i]*x_lin[j]
#     return nume/den


# @numba.jit(nopython=True, parallel=False, cache=False)
# def compute_coefficients(n, S1, S2, W2):
#     B0 = (n**2 - 3*n + 3)*S1 - n*S2 + 3*W2
#     B1 = -((n**2 - n)*S1 - 2*n*S2 + 3*W2)
#     B2 = -(2*n*S1 - (n + 3)*S2 + 6*W2)
#     B3 = 4*(n - 1)*S1 - 2*(n + 1)*S2 + 8*W2
#     B4 = S1 - S2 + W2
#     return B0, B1, B2, B3, B4

# @numba.jit(nopython=True, parallel=False, cache=False)
# def compute_S1(Wmat):
#     return np.sum((Wmat+Wmat.T)**2)/2

# @numba.jit(nopython=True, parallel=False, cache=False)
# def compute_S2(Wmat):
#     # np.sum((np.sum(Wmat,axis=0)+np.sum(Wmat,axis=1))**2)
#     S2 = 0
#     for i in range(Wmat.shape[0]):
#         S2 += (np.sum(Wmat[i,:]) + np.sum(Wmat[:,i]))**2
#     return S2

# @numba.jit(nopython=True, parallel=False, cache=False)
# def compute_ms(x):
#     m1 = 0
#     m2 = 0
#     m3 = 0
#     m4 = 0
#     for xi in x:
#         m1 += xi
#         m2 += xi**2
#         m3 += xi**3
#         m4 += xi**4 
#     return m1, m2, m3, m4

# def compute_EG_2(x, Wmat):
#     n = len(x)
#     S1 = compute_S1(Wmat)
#     S2 = compute_S2(Wmat)
#     W2 = np.sum(Wmat)**2

#     B0, B1, B2, B3, B4 = compute_coefficients(n, S1, S2, W2)
#     m1, m2, m3, m4 = compute_ms(x)

#     den = (m1**2-m2)**2*(n*(n-1)*(n-2)*(n-3))
#     num = B0*m2**2 + B1*m4 + B2*m1**2*m2 + B3*m1*m3 + B4*m1**4

#     if den == 0:
#         return np.nan
#     E_G_2 = num/den
#     return E_G_2

# def compute_Gi(x,W):
#     """
#     Compute Gi statistic for a 2D grid.

#     Example:
#     x = np.random.rand(3,4)
#     W = weight_from_dims(3, 4)
#     Gi, Zi, Pi = compute_Gi(x.flatten(), W)
#     """
#     Gi=np.empty_like(x, dtype=np.float64)
#     Zi=np.empty_like(x, dtype=np.float64)
#     Pi=np.empty_like(x, dtype=np.float64)
#     n=len(x)
#     Yi1 = np.sum(x)/(n-1)
#     Yi2 = np.sum(x**2)/(n-1)-Yi1**2
#     x_sum = np.sum(x)
#     for i in range(len(x)):
#         Gi[i] = np.sum(W[i,:]*x)/x_sum
#         wi = np.sum(W[i,:])
#         E_i = wi/(n-1)
#         V_i = (wi*(n-1-wi)*Yi2)/((n-1)**2*(n-2)*Yi1**2)
#         Zi[i] = (Gi[i]-E_i)/np.sqrt(V_i)
#         Pi[i] = 1.0 - scipy.stats.norm.cdf(np.abs(Zi[i]))
#     return Gi, Zi, Pi

# def compute_Hi(x,W):
#     """
#     Compute Hi statistic for a 2D grid.

#     Example:
#     x = np.random.rand(3,4)
#     W = weight_from_dims(3, 4)
#     Hi, Zi, Pi = compute_Hi(x.flatten(), W)
#     """
#     w1 = np.sum(W, axis=1)
#     w2 = np.sum(W**2,axis=1)
#     # Calculate spatial mean
#     xlag = np.matmul(W, x)/w1
#     # xlag = (np.matmul(W, x)+x)/(rowsum+1)
#     xresid = (x - xlag) ** 2
#     h1 = np.mean(xresid)
#     h2 = np.mean(xresid**2)
#     denom = h1 * w1
#     Hi = np.matmul(W, xresid) / denom

#     n = len(x)
#     term1 = 1/(n - 1)
#     term2 = (1/denom)**2
#     term3 = h2 - h1**2
#     term4 = (n * w2) - (w1**2)
#     VarHi = term1 * term2 * term3 * term4
#     dof = 2/VarHi
#     Zi = (2*Hi)/VarHi
#     Pi = 1 - scipy.stats.chi2.cdf(Zi, dof)
#     return Hi, Zi, Pi


# x = np.random.rand(100,100)

@numba.jit(nopython=True, parallel=False, cache=False)
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

# @numba.jit(nopython=True, parallel=True, cache=False)
# def G_and_H_numba(x, connectivity=CONNECTIVITY_QUEEN):
#     n=np.float64((x.shape[0]*x.shape[1]))
#     x_sum = np.float64(np.sum(x))
#     x2_sum = np.float64(np.sum(x**2))
#     Yi1 = x_sum/(n-1)
#     Yi2 = x2_sum/(n-1)-Yi1**2
#     w1=4*connectivity

#     # neighborhood means
#     xp = man_pad(x)
#     if connectivity == CONNECTIVITY_QUEEN:
#         xnm = (xp[:-2,1:-1]+xp[1:-1,:-2]+xp[2:,1:-1]+xp[1:-1,2:]+xp[:-2,:-2]+xp[:-2,2:]+xp[2:,:-2]+xp[2:,2:])/w1
#     else:
#         xnm = (xp[:-2,1:-1]+xp[1:-1,:-2]+xp[2:,1:-1]+xp[1:-1,2:])/w1
#     Gi = (xnm*w1)/x_sum

#     Ei=w1/(n-1)
#     V_i = (w1*(n-1-w1)*Yi2)/((n-1)**2*(n-2)*Yi1**2)
#     GZi = (Gi-Ei)/np.sqrt(V_i)

#     xresid = (x - xnm) ** 2
#     h1 = np.mean(xresid)
#     h2 = np.mean(xresid**2)
#     denom = h1 * w1
#     if connectivity == CONNECTIVITY_QUEEN:
#         Hi = ((xp[:-2,1:-1]-xnm)**2+(xp[1:-1,:-2]-xnm)**2+(xp[2:,1:-1]-xnm)**2+(xp[1:-1,2:]-xnm)**2+(xp[:-2,:-2]-xnm)**2+(xp[:-2,2:]-xnm)**2+(xp[2:,:-2]-xnm)**2+(xp[2:,2:]-xnm)**2) / denom
#     else:
#         Hi = ((xp[:-2,1:-1]-xnm)**2+(xp[1:-1,:-2]-xnm)**2+(xp[2:,1:-1]-xnm)**2+(xp[1:-1,2:]-xnm)**2) / denom
#     # because all weights are equal to one
#     w2=w1
#     VarHi = 1/(n - 1) * (1/denom)**2 * (h2 - h1**2) * ((n * w2) - (w1**2))
#     HZi = (2*Hi)/VarHi
#     return Hi,GZi, HZi, VarHi
# G_and_H_numba_serial = numba.jit(G_and_H_numba.py_func, nopython=True, parallel=False, cache=False)

# # x = np.random.rand(100,100)
# # G,H,_ = G_and_H_numba(x)
# # G,H,_ = G_and_H_numba_serial(x)

# @numba.jit(nopython=True, parallel=False, cache=False)
# def G_and_H(x, parallel=None, connectivity=CONNECTIVITY_QUEEN):
#     if parallel is None:
#         if numba.get_num_threads() > 1 and x.size > 5000:
#             parallel = True
#         else:
#             parallel = False
#     if parallel:
#         return G_and_H_numba(x, connectivity=connectivity)
#     else:
#         return G_and_H_numba_serial(x, connectivity=connectivity)

# # x = np.random.rand(100,100)
# # G,H,VarHi = G_and_H_numba(x)

# def H_PV(x, VarHi):
#     dof = 2/VarHi
#     return 1 - scipy.stats.chi2.cdf(x, dof)
# # H_PV(H, VarHi)

# def G_PV(x):
#     return  1.0 - scipy.stats.norm.cdf(np.abs(x))
# # G_PV(G)


@numba.jit(nopython=True, parallel=True, cache=False)
def G_classical(x, connectivity=CONNECTIVITY_QUEEN, normalize=True):
    x_sum = np.float64(np.sum(x))

    # neighborhood sums
    xp = man_pad(x)
    if connectivity == CONNECTIVITY_QUEEN:
        xns = (xp[1:-1,1:-1] + xp[:-2,1:-1]+xp[1:-1,:-2]+xp[2:,1:-1]+xp[1:-1,2:]+xp[:-2,:-2]+xp[:-2,2:]+xp[2:,:-2]+xp[2:,2:])
    else:
        xns = (xp[1:-1,1:-1] + xp[:-2,1:-1]+xp[1:-1,:-2]+xp[2:,1:-1]+xp[1:-1,2:])

    if normalize:
        w1 = 4*connectivity + 1
        n = np.float64(x.size)
        x_mean = x_sum / n
        x2_sum = np.sum(x**2)
        s2 = x2_sum/n-x_mean**2

        return (xns-x_mean*w1)/np.sqrt(s2*((n*w1 - w1**2)/(n-1)))
    else:
        return xns/x_sum

# G(x, connectivity=CONNECTIVITY_QUEEN, normalize=True)

@numba.jit(nopython=True, parallel=True, cache=False)
def G_permutation(x, connectivity=CONNECTIVITY_QUEEN, n_iter = 99, seed=42):
    np.random.seed(seed)
    x_sum = np.float64(np.sum(x))

    # neighborhood sums
    xp = man_pad(x)
    if connectivity == CONNECTIVITY_QUEEN:
        xns = (xp[:-2,1:-1]+xp[1:-1,:-2]+xp[2:,1:-1]+xp[1:-1,2:]+xp[:-2,:-2]+xp[:-2,2:]+xp[2:,:-2]+xp[2:,2:])
    else:
        xns = (xp[:-2,1:-1]+xp[1:-1,:-2]+xp[2:,1:-1]+xp[1:-1,2:])
    Gi = xns/x_sum

    perm_G_counts = np.zeros((2,Gi.shape[0],Gi.shape[1]), dtype=np.uint16)
    for rep in numba.prange(n_iter):
    # for rep in range(n_iter):
        # random permutation of x
        xrp_test = man_pad(np.random.permutation(x.flatten()).reshape(x.shape))
        # since xrp_test is a permutation of x, we can use the same neighborhood means
        # x_test_sum = np.float64(np.sum(xrp_test[1:-1,1:-1]))
        for xi in numba.prange(x.shape[0]):
            for yi in numba.prange(x.shape[1]):
                # new neighborhood mean
                if connectivity == CONNECTIVITY_QUEEN:
                    xrns_init_replacement = (
                        x[xi,yi]+
                        xrp_test[xi+1-1,yi+1]+
                        xrp_test[xi+1,yi+1-1]+
                        xrp_test[xi+1+1,yi+1]+
                        xrp_test[xi+1,yi+1+1]+
                        xrp_test[xi+1-1,yi+1-1]+
                        xrp_test[xi+1+1,yi+1-1]+
                        xrp_test[xi+1-1,yi+1+1]+
                        xrp_test[xi+1+1,yi+1+1]
                        )
                else:
                    xrns_init_replacement = (
                        x[xi,yi]+
                        xrp_test[xi+1-1,yi+1]+
                        xrp_test[xi+1,yi+1-1]+
                        xrp_test[xi+1+1,yi+1]+
                        xrp_test[xi+1,yi+1+1]
                        )
                tmp_sum = x_sum - xrp_test[xi+1,yi+1] + x[xi,yi]
                Gir = xrns_init_replacement/tmp_sum

                # compare with observed Gi
                perm_G_counts[np.int64(Gir>Gi[xi,yi]),xi,yi] += 1

    GPi = (1+perm_G_counts[1])/(n_iter+1)
    return Gi, GPi

def G(x, n_iter=0, connectivity=CONNECTIVITY_QUEEN, seed=42):
    G = G_classical(x, connectivity=connectivity, normalize=True)
    if n_iter > 0:
        GP = G_permutation(x, connectivity=connectivity, n_iter=n_iter, seed=seed)[1]
    else:
        GP = (1.0 - scipy.stats.norm.cdf(np.abs(G)))*2 # scale to [0,1]
    return G, GP



@numba.jit(nopython=True, parallel=True, cache=False)
def H_classical(x, connectivity=CONNECTIVITY_QUEEN, normalize=True, return_var=False):
    w1=4*connectivity + 1

    # neighborhood means
    xp = man_pad(x)
    if connectivity == CONNECTIVITY_QUEEN:
        xnm = (x+xp[:-2,1:-1]+xp[1:-1,:-2]+xp[2:,1:-1]+xp[1:-1,2:]+xp[:-2,:-2]+xp[:-2,2:]+xp[2:,:-2]+xp[2:,2:])/w1
    else:
        xnm = (x+xp[:-2,1:-1]+xp[1:-1,:-2]+xp[2:,1:-1]+xp[1:-1,2:])/w1

    # observed Hi
    xresid = man_pad((x - xnm)**2)
    denom = np.mean(xresid[1:-1,1:-1]) * w1
    if connectivity == CONNECTIVITY_QUEEN:
        Hi = (xresid[1:-1,1:-1]+xresid[:-2,1:-1]+xresid[1:-1,:-2]+xresid[2:,1:-1]+xresid[1:-1,2:]+xresid[:-2,:-2]+xresid[:-2,2:]+xresid[2:,:-2]+xresid[2:,2:]) / denom
    else:
        Hi = (xresid[1:-1,1:-1]+xresid[:-2,1:-1]+xresid[1:-1,:-2]+xresid[2:,1:-1]+xresid[1:-1,2:]) / denom

    if normalize:
        h1 = denom/w1
        h2 = np.mean(xresid[1:-1,1:-1]**2)
        n = np.float64((x.shape[0]*x.shape[1]))
        w2=w1
        VarHi = 1/(n - 1) * (1/denom)**2 * (h2 - h1**2) * ((n * w2) - (w1**2))
        HZi = (2*Hi)/VarHi
        if return_var:
            return HZi, VarHi
        else:
            return HZi, None
    else:
        return Hi, None

# H(x, connectivity=CONNECTIVITY_QUEEN, normalize=False, return_var=False)

@numba.jit(nopython=True, parallel=True, cache=False)
def H_permutation(x, connectivity=CONNECTIVITY_QUEEN, n_iter = 99, seed=42):
    np.random.seed(seed)
    w1=4*connectivity + 1

    # neighborhood means
    xp = man_pad(x)
    if connectivity == CONNECTIVITY_QUEEN:
        xnm = (x+xp[:-2,1:-1]+xp[1:-1,:-2]+xp[2:,1:-1]+xp[1:-1,2:]+xp[:-2,:-2]+xp[:-2,2:]+xp[2:,:-2]+xp[2:,2:])/w1
    else:
        xnm = (x+xp[:-2,1:-1]+xp[1:-1,:-2]+xp[2:,1:-1]+xp[1:-1,2:])/w1

    # observed Hi
    xresid = man_pad((x - xnm)**2)
    denom = np.mean(xresid[1:-1,1:-1]) * w1
    if connectivity == CONNECTIVITY_QUEEN:
        Hi = (xresid[1:-1,1:-1]+xresid[:-2,1:-1]+xresid[1:-1,:-2]+xresid[2:,1:-1]+xresid[1:-1,2:]+xresid[:-2,:-2]+xresid[:-2,2:]+xresid[2:,:-2]+xresid[2:,2:]) / denom
    else:
        Hi = (xresid[1:-1,1:-1]+xresid[:-2,1:-1]+xresid[1:-1,:-2]+xresid[2:,1:-1]+xresid[1:-1,2:]) / denom

    perm_H_counts = np.zeros((2,Hi.shape[0],Hi.shape[1]), dtype=np.uint16)
    for rep in numba.prange(n_iter):
        # random permutation of x
        xrp_test = man_pad(np.random.permutation(x.flatten()).reshape(x.shape))
        if connectivity == CONNECTIVITY_QUEEN:
            xrnm_init = (xrp_test[1:-1,1:-1]+xrp_test[:-2,1:-1]+xrp_test[1:-1,:-2]+xrp_test[2:,1:-1]+xrp_test[1:-1,2:]+xrp_test[:-2,:-2]+xrp_test[:-2,2:]+xrp_test[2:,:-2]+xrp_test[2:,2:])/w1
        else:
            xrnm_init = (xrp_test[1:-1,1:-1]+xrp_test[:-2,1:-1]+xrp_test[1:-1,:-2]+xrp_test[2:,1:-1]+xrp_test[1:-1,2:])/w1
        xresidr_init = (xrp_test[1:-1,1:-1] - xrnm_init)**2
        n = x.shape[0] * x.shape[1]
        denomr_sum_init = np.sum(xresidr_init) * w1
        for xi, yi in np.ndindex(x.shape):
            init_rval = xrp_test[xi+1,yi+1]
            xrp_test[xi+1,yi+1]=x[xi,yi]
            xstart = max(0,xi-1)
            xstart_offset = xstart-(xi-1)
            xend = min(xi+2,x.shape[0])
            ystart = max(0,yi-1)
            ystart_offset = ystart-(yi-1)
            yend = min(yi+2,x.shape[1])

            # partial sum for normalization
            xresidr_init_initsubarray_sum = np.sum(xresidr_init[xstart:xend, ystart:yend]) * w1
            # new neighborhood means
            xrnm_init_replacement_array = np.zeros((xend-xstart, yend-ystart))
            for ii,i in enumerate(range(xstart, xend)):
                for jj,j in enumerate(range(ystart, yend)):
                    # +1 everywhere because of the padding
                    ip = i+1
                    jp = j+1
                    if connectivity == CONNECTIVITY_QUEEN:
                        xrnm_init_replacement_array[ii,jj] = (
                            xrp_test[ip,jp]+
                            xrp_test[ip-1,jp]+
                            xrp_test[ip,jp-1]+
                            xrp_test[ip+1,jp]+
                            xrp_test[ip,jp+1]+
                            xrp_test[ip-1,jp-1]+
                            xrp_test[ip+1,jp-1]+
                            xrp_test[ip-1,jp+1]+
                            xrp_test[ip+1,jp+1]
                            )/w1
                    else:
                        xrnm_init_replacement_array[ii,jj] = (
                            xrp_test[ip,jp]+
                            xrp_test[ip-1,jp]+
                            xrp_test[ip,jp-1]+
                            xrp_test[ip+1,jp]+
                            xrp_test[ip,jp+1]
                            )/w1
            # new residuals
            xresidr_init_replacement_array = np.zeros((xend-xstart, yend-ystart))
            for ii,i in enumerate(range(xstart, xend)):
                for jj,j in enumerate(range(ystart, yend)):
                    # +1 everywhere because of the padding
                    ip = i+1
                    jp = j+1
                    xresidr_init_replacement_array[ii,jj] = (xrp_test[ip,jp] - xrnm_init_replacement_array[ii,jj])**2

            # update the denominator, remove initial partial sum and add the new partial sum, and calculate the mean
            denomr = (denomr_sum_init - xresidr_init_initsubarray_sum + np.sum(xresidr_init_replacement_array)*w1)/n

            # pad for boundaries
            xresidr_init_replacement_array = man_pad(xresidr_init_replacement_array)

            # calculate Hi for the replacement, xstart_offset and ystart_offset are for the boundaries
            if connectivity == CONNECTIVITY_QUEEN:
                Hir =  (
                    xresidr_init_replacement_array[2-xstart_offset, 2-ystart_offset] +
                    xresidr_init_replacement_array[2-xstart_offset-1, 2-ystart_offset] +
                    xresidr_init_replacement_array[2-xstart_offset, 2-ystart_offset-1] +
                    xresidr_init_replacement_array[2-xstart_offset+1, 2-ystart_offset] +
                    xresidr_init_replacement_array[2-xstart_offset, 2-ystart_offset+1] +
                    xresidr_init_replacement_array[2-xstart_offset-1, 2-ystart_offset-1] +
                    xresidr_init_replacement_array[2-xstart_offset+1, 2-ystart_offset-1] +
                    xresidr_init_replacement_array[2-xstart_offset-1, 2-ystart_offset+1] +
                    xresidr_init_replacement_array[2-xstart_offset+1, 2-ystart_offset+1]
                    )/denomr
            else:
                Hir =  (
                    xresidr_init_replacement_array[2-xstart_offset, 2-ystart_offset] +
                    xresidr_init_replacement_array[2-xstart_offset-1, 2-ystart_offset] +
                    xresidr_init_replacement_array[2-xstart_offset, 2-ystart_offset-1] +
                    xresidr_init_replacement_array[2-xstart_offset+1, 2-ystart_offset] +
                    xresidr_init_replacement_array[2-xstart_offset, 2-ystart_offset+1]
                    )/denomr

            # compare with observed Hi
            perm_H_counts[np.int64(Hir>Hi[xi,yi]),xi,yi] += 1

            # restore the original permuted value
            xrp_test[xi+1,yi+1]=init_rval

    HPi = (1+perm_H_counts[1])/(n_iter+1)
    return Hi, HPi

def H(x, n_iter=99, connectivity=CONNECTIVITY_QUEEN, seed=42):
    H, VarH = H_classical(x, connectivity=connectivity, normalize=True, return_var=True)
    if n_iter > 0:
        HP = H_permutation(x, connectivity=connectivity, n_iter=n_iter, seed=seed)[1]
    else:
        dof = 2/VarH
        HP = 1 - scipy.stats.chi2.cdf(H, dof)
    return H, HP

# x = np.random.normal(size=(100, 100))
# # x = np.random.random(size=(100, 100))
# x = np.random.binomial(p=0.25,n=10,size=(100, 100)).astype(np.float64)
# x[:2, :] += 10
# houtiter = H(x, n_iter=999)[1]
# houtcl = H(x, n_iter=0)[1]
# # import scipy.stats
# # houtiter = scipy.stats.false_discovery_control(houtiter)
# # houtcl = scipy.stats.false_discovery_control(houtcl)
# np.round(x[:5,3:8],2)
# np.round(houtcl[:5,3:8],2)
# np.round(houtiter[:5,3:8],2)
# import matplotlib.pyplot as plt
# plt.close()
# fig, ax = plt.subplots(3, 1, figsize=(10, 15))
# ax[0].hist(houtiter[1:3,:].flatten(), bins=20, label='H permutation')
# ax[0].set_title('H permutation distribution')
# ax[0].set_xlabel('H permutation values')
# ax[0].set_ylabel('Frequency')
# # ax[0].set_xlim(0, 1)
# ax[1].hist(houtcl[1:3,:].flatten(), bins=20, label='H permutation')
# ax[1].set_title('H classical distribution')
# ax[1].set_xlabel('H classical values')
# ax[1].set_ylabel('Frequency')
# # ax[1].set_xlim(0, 1)
# ax[2].scatter(houtcl[1:3,:].flatten(),houtiter[1:3,:].flatten(), label='H permutation', s=1)
# ax[2].set_title('H permutation vs H classical')
# ax[2].set_xlabel('H classical values')
# ax[2].set_ylabel('H permutation values')
# # ax[2].set_xlim(0, 1)
# # ax[2].set_ylim(0, 1)
# plt.savefig('H_permutation.png')



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
