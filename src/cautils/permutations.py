import numpy as np
import numpy.typing as npt
import numba
import itertools


@numba.jit(nopython=True, parallel=False, cache=False)
def get_nuclei_boundary(resdmat: npt.NDArray, ch:int=-2, thr:float=1, maxdist:int = 3, offset:int=0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nuclei_boundary_outer = np.zeros(resdmat.shape[1], dtype=np.uint8)
    nuclei_boundary_inner = np.zeros(resdmat.shape[1], dtype=np.uint8)
    for i in range(resdmat.shape[1]):
        if np.sum(resdmat[ch,i,:]>=thr)==0:
            nuclei_boundary_inner[i] = 0
            nuclei_boundary_outer[i] = 0
        else:
            nuclei_boundary_outer[i] = resdmat.shape[2]-np.argmax(resdmat[ch,i,:][::-1]>=thr)-1
            nuclei_boundary_inner[i] = np.argmax(resdmat[ch,i,:]>=thr)
    if offset != 0:
        for i in range(resdmat.shape[1]):
            if int(nuclei_boundary_outer[i])+offset >= nuclei_boundary_inner[i]:
                nuclei_boundary_outer[i] += offset
            else:
                nuclei_boundary_outer[i] = nuclei_boundary_inner[i]
    nuclei_boundary_max = nuclei_boundary_outer + maxdist
    return nuclei_boundary_inner, nuclei_boundary_outer, nuclei_boundary_max

# (resdmat[-2,:,:]==1).astype(int)
@numba.jit(nopython=True, parallel=False, cache=False)
def get_nuclei_sum_marker(resdmat: npt.NDArray, nuclei_boundary_inner: npt.NDArray, nuclei_boundary_outer: npt.NDArray) -> np.ndarray:
    nuclei_marker_long = np.zeros((resdmat.shape[0]+1, resdmat.shape[1]))
    for i in range(resdmat.shape[1]):
        nuclei_marker_long[:-1,i] = np.sum(resdmat[:,i,nuclei_boundary_inner[i]:(nuclei_boundary_outer[i]+1)],axis=1)
        nuclei_marker_long[-1,i] = (nuclei_boundary_outer[i]+1)-nuclei_boundary_inner[i]
    nuclei_marker = np.sum(nuclei_marker_long,axis=1)
    return nuclei_marker

@numba.jit(nopython=True, parallel=False, cache=False)
def subset_resdmat(resdmat: npt.NDArray, nuclei_boundary_outer: npt.NDArray, nuclei_boundary_max: npt.NDArray) -> np.ndarray:
    maxdist = set(nuclei_boundary_max-nuclei_boundary_outer)
    assert len(maxdist)==1, "Not all values exist, reduce max distance!"
    maxdist = maxdist.pop()
    resdmat_sub = np.zeros((resdmat.shape[0], resdmat.shape[1], maxdist))
    for i in range(resdmat.shape[1]):
        resdmat_sub[:,i,:] = resdmat[:,i,(nuclei_boundary_outer[i]+1):(nuclei_boundary_max[i]+1)]
    return resdmat_sub

def get_combinations(indexes: list[int], return_array: bool = False) -> np.ndarray:
    assert all(np.array(indexes)<254)
    combs = []
    for i in range(0,len(indexes)+1):
        combs = combs+list(itertools.combinations(indexes, i))
    if return_array:
        combs_arr = np.ones((len(combs), len(indexes)), dtype=np.uint8)*255
        for i,co in enumerate(combs):
            combs_arr[i,:len(co)] = list(co)
        return combs_arr
    else:
        return combs

# def get_combinations_l2(combs):
#     allowed_combinations = []
#     for i,co1 in enumerate(combs):
#         for j,co2 in enumerate(combs):
#             if len(co1) <= len(co2):
#                 if all([x in co2 for x in co1]):
#                     allowed_combinations.append([i,j])
#     return np.array(allowed_combinations)

# @numba.jit(nopython=True, parallel=False, cache=False)
# def get_combinations_lx(combs_arr, permutations_arr):
#     l = permutations_arr.shape[1]
#     allowed_combinations = []
#     # for inds in itertools.permutations(range(combs_arr.shape[0]), l):
#     # for inds in itertools.permutations(range(100), l):
#     for inds in permutations_arr:
#         if np.all(np.diff(np.array([combs_arr[i][combs_arr[i]!=255].shape[0] for i in inds]))<=0):
#             all_layer_match = np.zeros(l-1, dtype=numba.boolean)
#             for j in range(1,l):
#                 t1ls = combs_arr[inds[j],:][combs_arr[inds[j],:]!=255]
#                 t2ls = combs_arr[inds[j-1],:][combs_arr[inds[j-1],:]!=255]
#                 outbl = np.zeros(t1ls.shape[0], dtype=numba.boolean)
#                 for k,t1 in enumerate(t1ls):
#                     for t2 in t2ls:
#                         if t1==t2:
#                             outbl[k]=True
#                 if np.all(outbl):
#                     all_layer_match[j-1] = True
#             if np.all(all_layer_match):
#                 allowed_combinations.append(inds)
#     return allowed_combinations

# @numba.jit(nopython=True, parallel=False, cache=False)
# def filter_combinations(combs_arr, allowed_combinations):
#     bi_len_to_keep = np.zeros(combs_arr.shape[0], dtype=numba.boolean)
#     for i in range(combs_arr.shape[0]):
#         bi_len_to_keep[i] = np.sum(combs_arr[i,:]<255)<4
#     ind_to_keep = np.arange(0,combs_arr.shape[0])[bi_len_to_keep]
#     assert np.all(np.diff(ind_to_keep)==1)
#     maxind = np.max(ind_to_keep) + 1

#     bi_len_2_to_keep = np.zeros(allowed_combinations.shape[0], dtype=numba.boolean)
#     for i in range(allowed_combinations.shape[0]):
#         bi_len_2_to_keep[i] = np.all(allowed_combinations[i] < maxind)

#     allowed_combinations_filt = allowed_combinations[bi_len_2_to_keep,]

#     bi_dir_to_keep = np.zeros(allowed_combinations_filt.shape[0], dtype=numba.boolean)
#     for i in range(allowed_combinations_filt.shape[0]):
#         tbi = np.zeros(allowed_combinations_filt.shape[1], dtype=numba.boolean)
#         for j in range(allowed_combinations_filt.shape[1]):
#             tt = combs_arr[allowed_combinations_filt[i,j]]
#             tt = tt[tt<255]
#             if len(tt)<2:
#                 tbi[j] = True
#             else:
#                 tds = np.array(list(np.diff(tt)) + [tt[0]-tt[-1]+combs_arr.shape[1]])
#                 tbi[j] = np.sum(tds>1)==1
#         bi_dir_to_keep[i] = np.all(tbi)

#     return allowed_combinations_filt[bi_dir_to_keep]


# @numba.jit(nopython=True, parallel=False, cache=False)
# def perm_generator(n: int, dim: int):
#     counter = [0]*dim 
#     # counter = np.zeros(dim, dtype=np.uint16)
#     yield [c for c in counter]
#     for j in range(n**dim-1):
#         counter[0] += 1
#         for i in range(dim-1):
#             if counter[i] == n:
#                 counter[i] = 0
#                 counter[i+1] += 1
#         yield [c for c in counter]

# n=4
# dim=3
# for co in perm_generator(n,dim):
#     print(co)

# @numba.jit(nopython=True, parallel=False, cache=False)
# def get_combinations_lx_filter(combs_arr, dmax):

#     # only allow maximum number of fields per layer,
#     # and since combs_arr is arrange from low to high,
#     # get maximum index 
#     maxn = combs_arr.shape[1]//2
#     bi_len_to_keep = np.zeros(combs_arr.shape[0], dtype=numba.boolean)
#     # bi_len_to_keep = np.zeros(combs_arr.shape[0], dtype=bool)
#     for i in range(combs_arr.shape[0]):
#         bi_len_to_keep[i] = np.sum(combs_arr[i,:]<255)<maxn
#     ind_to_keep = np.arange(0,combs_arr.shape[0])[bi_len_to_keep]
#     assert np.all(np.diff(ind_to_keep)==1)
#     maxind = np.max(ind_to_keep) + 1
#     assert maxind < combs_arr.shape[0]

#     allowed_combinations = []
#     tbi = np.zeros(dmax, dtype=numba.boolean)
#     # tbi = np.zeros(dmax, dtype=bool)
#     # generator
#     inds = [0]*dmax
#     for j in range(maxind**dmax):
#     # for inds in perm_generator(maxind,dmax):
#         if np.all(np.diff(np.array([combs_arr[i][combs_arr[i]!=255].shape[0] for i in inds]))<=0):
#             all_layer_match = np.zeros(dmax-1, dtype=numba.boolean)
#             # all_layer_match = np.zeros(dmax-1, dtype=bool)
#             for j in range(1,dmax):
#                 t1ls = combs_arr[inds[j],:][combs_arr[inds[j],:]!=255]
#                 t2ls = combs_arr[inds[j-1],:][combs_arr[inds[j-1],:]!=255]
#                 outbl = np.zeros(t1ls.shape[0], dtype=numba.boolean)
#                 # outbl = np.zeros(t1ls.shape[0], dtype=bool)
#                 for k,t1 in enumerate(t1ls):
#                     for t2 in t2ls:
#                         if t1==t2:
#                             outbl[k]=True
#                 if np.all(outbl):
#                     all_layer_match[j-1] = True
#             if np.all(all_layer_match):
#                 for j in range(dmax):
#                     tt = combs_arr[inds[j]]
#                     tt = tt[tt<255]
#                     if len(tt)<2:
#                         tbi[j] = True
#                     else:
#                         tds = np.array(list(np.diff(tt)) + [tt[0]-tt[-1]+combs_arr.shape[1]])
#                         tbi[j] = np.sum(tds>1)==1
#                 if np.all(tbi):
#                     allowed_combinations.append(inds)
#         # generator
#         inds[0] += 1
#         for i in range(dmax-1):
#             if inds[i] == maxind:
#                 inds[i] = 0
#                 inds[i+1] += 1
#     return allowed_combinations

@numba.jit(nopython=True, parallel=False, cache=False)
def meshgrid_like(dmax: int, width: int) -> np.ndarray:
    par_out = np.zeros((dmax**width,width), dtype=np.uint8)
    stepsizes = [dmax**i for i in range(width)]
    for i in range(par_out.shape[1]):
        c = 0
        for j in range(par_out.shape[0]):
            par_out[j,i] = int(c/stepsizes[-1])%dmax
            c += stepsizes[i]
    return par_out

@numba.jit(nopython=True, parallel=False, cache=False)
def get_combinations_lx_filter(combs_arr: npt.NDArray, dmax: int) -> np.ndarray:
    max_width = combs_arr.shape[1]//2-1
    width_ls = list(range(1,max_width+1))
    par_ls = [meshgrid_like(dmax, width) for width in width_ls]
    lens = [par.shape[0]*combs_arr.shape[1] for par in par_ls]
    total_len = sum(lens)
    # matcher_template = np.zeros((combs_arr.shape[1],dmax),dtype=bool)
    matcher_template = np.zeros((combs_arr.shape[1],dmax),dtype=numba.boolean)
    numbered_lookup = np.arange(combs_arr.shape[1],dtype=np.uint8)
    matcher_sub_template = np.zeros(combs_arr.shape[1],dtype=np.uint8)+255
    allowed_combinations_arr = np.zeros((total_len,dmax), dtype=np.uint8)
    r = 0
    for q in range(len(par_ls)):
        par = par_ls[q]
        for offset in range(combs_arr.shape[1]):
            for i in range(par.shape[0]):
                matcher = matcher_template.copy()
                for j in range(par.shape[1]):
                    matcher[(offset+j)%(combs_arr.shape[1]),:(par[i,j]+1)] = True

                for k in range(matcher.shape[1]):
                    matcher_sub = matcher_sub_template.copy()
                    matcher_sub[:np.sum(matcher[:,k])] = numbered_lookup[matcher[:,k]]

                    matchind = -1
                    for l in range(combs_arr.shape[0]):
                        if np.all(combs_arr[l,:] == matcher_sub):
                            matchind = l
                            break
                    allowed_combinations_arr[r,k] = matchind
                r+=1
    return  allowed_combinations_arr

# nangles = 8
# dmax=2
# combs_arr = get_combinations(list(range(nangles)), return_array=True)
# out = get_combinations_lx_filter(combs_arr, dmax)
# len(out)

# #7
# 15
# 43
# 85
# 141
# 211

# # 8
# 25
# 105
# 273
# 561


# permutations_arr = np.stack(np.meshgrid(*[list(range(combs_arr.shape[0])) for _ in range(dmax)])).swapaxes(0,dmax).reshape(-1,dmax)

@numba.jit(nopython=True, parallel=False, cache=False)
def calculate_single_distance_marker_sums(resdmat_sub: npt.NDArray, combs_arr: npt.NDArray, maxdist: int = 3)-> np.ndarray:
    nind = int(resdmat_sub.shape[0])
    nls = np.zeros((combs_arr.shape[0], maxdist, resdmat_sub.shape[0]+1), dtype=np.float64)
    for i,co in enumerate(combs_arr):
        co = co[co!=255]
        for j in range(maxdist):
            nls[i,j,nind] = float(len(co))
            nls[i,j,:-1] = np.sum(resdmat_sub[:,co,j],axis=1)
    return nls

@numba.jit(nopython=True, parallel=False, cache=False)
def calculate_all_distance_marker_sums(resdmat_sub: npt.NDArray, allowed_combinations: npt.NDArray, nls: npt.NDArray, maxdist: int = 3) -> np.ndarray:
    lac = len(allowed_combinations)
    if allowed_combinations.shape[1]>2:
        nnls = np.zeros((lac, resdmat_sub.shape[0]+1))
        for i in range(lac):
            for j in range(allowed_combinations.shape[1]):
                nnls[i,:] += nls[allowed_combinations[i,j],j,:]
    else:
        nnls = np.zeros((lac*(maxdist-1), resdmat_sub.shape[0]+1))
        for j in range(maxdist-1):
            if j==0:
                for i in range(lac):
                    nnls[i,:] = nls[allowed_combinations[i][0],j,:] + nls[allowed_combinations[i][1],j+1,:]
            else:
                for i in range(lac):
                    nnls[j*lac + i,:] = np.sum(nls[-1,:j,:],axis=0) + nls[allowed_combinations[i][0],j,:] + nls[allowed_combinations[i][1],j+1,:]
    return nnls

# out = calculate_all_distance_marker_sums(resdmat_sub, allowed_combinations, nls)


@numba.jit(nopython=True, parallel=False, cache=False)
def normalize_marker(nnls, nuclei_marker, normalize="all", invert=False):
    if invert:
        if normalize=="all":
            total_n = nuclei_marker[-1] - nnls[:,-1].copy().reshape(-1,1)
            nnlsnorm = (nuclei_marker[:-1] - nnls[:,:-1]) / total_n
        elif normalize=="sub":
            total_n = nnls[:,-1].copy().reshape(-1,1)
            nnlsnorm = nnls[:,:-1] / total_n
        else:
            nnlsnorm = nnls[:,:-1]
    else:
        if normalize=="all":
            total_n = nuclei_marker[-1] + nnls[:,-1].copy().reshape(-1,1)
            nnlsnorm = (nuclei_marker[:-1] + nnls[:,:-1]) / total_n
        elif normalize=="sub":
            total_n = nnls[:,-1].copy().reshape(-1,1)
            nnlsnorm = nnls[:,:-1] / total_n
        else:
            nnlsnorm = nnls[:,:-1]
    return nnlsnorm

@numba.jit(nopython=True, parallel=False, cache=False)
def get_max_shrinkage(resdmat, ch=-2, thr=1, maxdist=3, offset=0):
    nuclei_boundary_inner, nuclei_boundary_outer, _ = get_nuclei_boundary(resdmat, ch=ch, thr=thr, maxdist = np.abs(maxdist), offset=offset)
    nuc_diff = nuclei_boundary_outer-nuclei_boundary_inner
    return np.min(nuc_diff)

# combs_arr = sa.combs_arr
# allowed_combinations = sa.allowed_combinations
# res = []
# maxdist = -3
# for i in range(1000):
#     resdmat = sa.radial_intensities[i,:,:,:]
#     nuclei_boundary_inner, nuclei_boundary_outer, nuclei_boundary_max = get_nuclei_boundary(resdmat, ch=ch, thr=thr, maxdist = np.abs(maxdist), offset=offset)
#     nuc_diff = nuclei_boundary_outer-nuclei_boundary_inner
#     res.append(np.all((nuc_diff)>=-maxdist))
@numba.jit(nopython=True, parallel=False, cache=False)
def get_boundary_permutation_marker(resdmat, combs_arr, allowed_combinations, ch=-2, thr=1, maxdist=3, normalize="all", offset=0):
    nuclei_boundary_inner, nuclei_boundary_outer, nuclei_boundary_max = get_nuclei_boundary(resdmat, ch=ch, thr=thr, maxdist = np.abs(maxdist), offset=offset)
    if np.any(nuclei_boundary_max>=resdmat.shape[2]):
        raise RuntimeError("Maximum considered distance too large, decrease 'maxdist' or increase radial distance.")

    # inverse
    if maxdist < 0: 
        nuc_diff = nuclei_boundary_outer-nuclei_boundary_inner
        if not np.all((nuc_diff)>=-maxdist):
            raise RuntimeError("Nuclei too small for negative maxdist, increase 'maxdist'")
        nuclei_boundary_outer_red = nuclei_boundary_outer - -maxdist
        resdmat_sub = subset_resdmat(resdmat, nuclei_boundary_outer_red, nuclei_boundary_outer)
    else:
        resdmat_sub = subset_resdmat(resdmat, nuclei_boundary_outer, nuclei_boundary_max)
    nls = calculate_single_distance_marker_sums(resdmat_sub, combs_arr)
    nnls = calculate_all_distance_marker_sums(resdmat_sub, allowed_combinations, nls)
    if normalize=="all":
        nuclei_marker = get_nuclei_sum_marker(resdmat, nuclei_boundary_inner, nuclei_boundary_outer)
        return normalize_marker(nnls, nuclei_marker, "all", invert=maxdist<0)
    elif normalize=="sub":
        nuclei_marker = get_nuclei_sum_marker(resdmat, nuclei_boundary_inner, nuclei_boundary_outer)
        return normalize_marker(nnls, nuclei_marker, "sub", invert=maxdist<0)
    elif normalize=="none":
        return nnls
    else:
        raise ValueError("normalize has to be one of 'all', 'sub', 'none'")


@numba.jit(nopython=True, parallel=True, cache=False)
def get_boundary_permutation_marker_multiple(resdmata, combs_arr, allowed_combinations, mean_intensities, ch=-2, thr=1, maxdist=3, normalize="all", offset=0):
        n_bins = allowed_combinations.shape[0]//200
        n_bins = n_bins if n_bins>100 else 100
        ranges = np.zeros((resdmata.shape[0], 2))
        tmpbins = np.zeros((resdmata.shape[0], n_bins*10))
        for i in numba.prange(resdmata.shape[0]):
            tmpperm = get_boundary_permutation_marker(resdmata[i,:,:,:], combs_arr, allowed_combinations, ch=ch, thr=thr, maxdist=maxdist, normalize=normalize, offset=offset)[:,0]
            ranges[i,0] = np.min(tmpperm)
            ranges[i,1] = np.max(tmpperm)
            tmpbins[i,:] = np.histogram(tmpperm, bins=n_bins*10, range=[ranges[i,0], ranges[i,1]])[0]

        ranges[:,0] -= mean_intensities
        ranges[:,1] -= mean_intensities
        histrange = np.array([np.min(ranges[:,0]), np.max(ranges[:,1])])

        bins = np.linspace(histrange[0], histrange[1], n_bins+1)
        hist_data = np.zeros((resdmata.shape[0], n_bins))
        hist_grid_points = np.linspace(histrange[0], histrange[1], n_bins)
        for i in numba.prange(resdmata.shape[0]):
            ind_bins = np.digitize(np.linspace(ranges[i,0], ranges[i,1], n_bins*10), bins, right=True)
            for j in range(len(ind_bins)):
                hist_data[i,ind_bins[j]-1] += tmpbins[i,j]
        return hist_data, hist_grid_points

 
# resdmata=sa.radial_intensities[:,chind,:,:]
# combs_arr=sa.combs_arr
# allowed_combinations=sa.allowed_combinations
# mean_intensities=sa.intensities[:,channelname[0]].to_numpy()
# ch=-1
# thr=1
# maxdist=3
# normalize="all"

# resdmata = sa.radial_intensities
# combs_arr = sa.combs_arr
# allowed_combinations = sa.allowed_combinations
# import time
# print("calculate permutation")
# t1 = time.time()
# out = get_boundary_permutation_marker_multiple(resdmata, combs_arr, allowed_combinations, mean_intensities, ch=-1, thr=1, maxdist=3, normalize="all")
# t2 = time.time()
# print(f"done in: {t2-t1}")

# out.shape

