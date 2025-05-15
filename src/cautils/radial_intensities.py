from numba import njit
import numba
import cv2
import numpy as np
import matplotlib.pyplot as plt

# https://totologic.blogspot.com/2014/01/accurate-point-in-triangle-test.html
@njit(cache=True)
def side(x1:  np.float64, y1:  np.float64, x2:  np.float64, y2:  np.float64, x:  np.float64, y:  np.float64) ->  np.float64:
    return (y2 - y1)*(x - x1) + (-x2 + x1)*(y - y1)

@njit(cache=True)
def circle(phi: np.float64) -> np.ndarray[np.float64]:
    return np.array([np.cos(phi), np.sin(phi)])

@njit(cache=True)
def in_approx_cone(pt: np.ndarray[np.float64], r: np.float64, phi: np.float64, epsphi: np.float64, p1: np.ndarray[np.float64] = np.array([0.0,0.0]),eps: np.float64 = 1e-6) -> bool:
    p2 = r*circle(phi+epsphi-eps)+p1
    p3 = r*circle(phi)+p1
    p4 = r*circle(phi-epsphi+eps)+p1

    b1 = side(p1[0],p1[1],p2[0],p2[1],pt[0],pt[1])>=-eps
    b2 = side(p2[0],p2[1],p3[0],p3[1],pt[0],pt[1])>=-eps
    b3 = side(p3[0],p3[1],p4[0],p4[1],pt[0],pt[1])>=-eps
    b4 = side(p4[0],p4[1],p1[0],p1[1],pt[0],pt[1])>=-eps

    return b1 and b2 and b3 and b4

# @njit
# def in_approx_cone_broad_base(pt,r,phi,epsphi,p1=(0,0),eps=1e-6) -> bool:
#     p2 = r*circle(phi+epsphi-eps)+p1
#     p3 = r*circle(phi)+p1
#     p4 = r*circle(phi-epsphi+eps)+p1
#     p5 = 0.5*circle(phi-np.pi/2)+p1
#     p6 = 0.5*circle(phi-np.pi)+p1
#     p7 = 0.5*circle(phi+np.pi/2)+p1

#     b1 = side(p7[0],p7[1],p2[0],p2[1],pt[0],pt[1])>=-eps/10
#     b2 = side(p2[0],p2[1],p3[0],p3[1],pt[0],pt[1])>=-eps/10
#     b3 = side(p3[0],p3[1],p4[0],p4[1],pt[0],pt[1])>=-eps/10
#     b4 = side(p4[0],p4[1],p5[0],p5[1],pt[0],pt[1])>=-eps/10
#     b5 = side(p5[0],p5[1],p6[0],p6[1],pt[0],pt[1])>=-eps/10
#     b6 = side(p6[0],p6[1],p7[0],p7[1],pt[0],pt[1])>=-eps/10

#     return b1 and b2 and b3 and b4 and b5 and b6

@njit(cache=True, parrallel=True)
def create_kernel(tis: np.ndarray[np.float64], r: np.float64, phi: np.float64, epsphi: np.float64) -> np.ndarray[np.bool_]:
    p1 = np.array([tis[0]//2,tis[1]//2])
    ti = np.zeros(tis, dtype=numba.boolean)
    for t in range(ti.shape[0]):
        for l in range(ti.shape[1]):
            ti[t,l] = in_approx_cone((t,l),r,phi,epsphi,p1=p1)
    return ti


@njit(cache=True, parallel=True)
def create_kernel_list(tis: np.ndarray[np.float64], rs: np.float64, phis: np.float64, epsphi: np.float64):
    tils = list()
    for i in range(len(phis)):
        for j in range(len(rs)):
            ti = create_kernel(tis,rs[j],phis[i],epsphi)
            tils.append(ti)
    return tils

@njit(cache=True)
def create_kernel_diff_list(tils):
    tidls = list()
    tidls.append(np.zeros(tils[0].shape, dtype=numba.boolean))
    for k in range(len(tils)):
        tidls.append(np.logical_and(~tils[k-1], tils[k]))
    del tidls[0]
    return tidls

@njit(cache=True)
def create_bbox(p0, sh, imgsh):
    p0r = np.round(p0).astype(np.uint)
    # to relative coordinates (compared to img)
    kernel_bbox_rel = [
        p0r[0]-(sh[0]-1)//2,
        p0r[1]-(sh[1]-1)//2,
        p0r[0]+(sh[0]-1)//2+1,
        p0r[1]+(sh[1]-1)//2+1
    ]
    img_bbox = [0,0,imgsh[1],imgsh[2]]
    kernel_bbox_rel[0] = img_bbox[0] = max(kernel_bbox_rel[0],img_bbox[0])
    kernel_bbox_rel[1] = img_bbox[1] = max(kernel_bbox_rel[1],img_bbox[1])
    kernel_bbox_rel[2] = img_bbox[2] = min(kernel_bbox_rel[2],img_bbox[2])
    kernel_bbox_rel[3] = img_bbox[3] = min(kernel_bbox_rel[3],img_bbox[3])

    # back to kernel coordinates
    kernel_bbox = [
        kernel_bbox_rel[0]-p0r[0]+(sh[0]-1)//2,
        kernel_bbox_rel[1]-p0r[1]+(sh[1]-1)//2,
        kernel_bbox_rel[2]-p0r[0]+(sh[0]-1)//2,
        kernel_bbox_rel[3]-p0r[1]+(sh[1]-1)//2
    ]

    return img_bbox, kernel_bbox


@njit(cache=True)
def get_subimg_part(image, mask, mask_nuc, cellid, cellbbox, cellcentroid, rmax=10):
    bbox = cellbbox
    bbox[0]-=rmax
    bbox[1]-=rmax
    bbox[2]+=rmax
    bbox[3]+=rmax
    bbox[0] = bbox[0] if bbox[0] > 0 else 0
    bbox[1] = bbox[1] if bbox[1] > 0 else 0
    bbox[2] = bbox[2] if bbox[2] < image.shape[1] else image.shape[1]
    bbox[3] = bbox[3] if bbox[3] < image.shape[2] else image.shape[2]

    # centroid
    p0 = cellcentroid - np.array([bbox[0],bbox[1]])

    # subset
    subimg = np.zeros((image.shape[0]+10,bbox[2]-bbox[0],bbox[3]-bbox[1]), dtype=np.float64)
    subimg[:-10,:,:] = image[:,bbox[0]:bbox[2],bbox[1]:bbox[3]]
    subimg[-10,:,:] = mask[1,bbox[0]:bbox[2],bbox[1]:bbox[3]]
    subimg[-9,:,:] = mask[2,bbox[0]:bbox[2],bbox[1]:bbox[3]]
    subimg[-8,:,:] = mask_nuc[1,bbox[0]:bbox[2],bbox[1]:bbox[3]]
    subimg[-7,:,:] = mask_nuc[2,bbox[0]:bbox[2],bbox[1]:bbox[3]]

    submask = mask[0,bbox[0]:bbox[2],bbox[1]:bbox[3]]
    submasknuc = mask_nuc[0,bbox[0]:bbox[2],bbox[1]:bbox[3]]
    # any mask
    subimg[-6,:,:] = submask>0
    # mask of cellid
    subimg[-5,:,:] = submask == cellid
    # mask of other cells
    subimg[-4,:,:] = np.logical_and(subimg[-6,:,:], ~subimg[-5,:,:].astype(numba.boolean))
    # any nuclei
    subimg[-3,:,:] = submasknuc>0
    # nuclei of cellid
    subimg[-2,:,:] = submasknuc == cellid
    # nuclei of other cells
    subimg[-1,:,:] = np.logical_and(subimg[-3,:,:], ~subimg[-2,:,:].astype(numba.boolean))

    return subimg, p0


def scale_subimg(subimg, p0, scale=1, nch_nn=6):
    # scale image
    if scale != 1:
        subimg_scaled = np.zeros((subimg.shape[0], subimg.shape[1]*scale, subimg.shape[2]*scale))
        for i in range(subimg.shape[0]):
            if i < subimg.shape[0]-nch_nn:
                interpol = cv2.INTER_LINEAR
            else:
                interpol = cv2.INTER_NEAREST
            subimg_scaled[i,:,:] = cv2.resize(subimg[i,:,:], (subimg.shape[2]*scale, subimg.shape[1]*scale), interpolation=interpol)
        p0 = p0*scale
        return subimg_scaled, p0
    else:
        return subimg, p0


@njit(cache=True, parallel=True)
def bbox_centroids(mask):
    maxval = int(np.max(mask)+1)
    centroids = np.zeros((maxval,3), dtype=float)
    bboxs = np.zeros((maxval,4), dtype=np.int64)
    bboxs[:,0] = np.max(np.array(mask.shape))+1
    bboxs[:,1] = np.max(np.array(mask.shape))+1
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] != 0:
                centroids[mask[i,j],:] += np.array([1,i,j])
                if i < bboxs[mask[i,j],0]:
                    bboxs[mask[i,j],0] = i
                if i+1 > bboxs[mask[i,j],2]:
                    bboxs[mask[i,j],2] = i+1
                if j < bboxs[mask[i,j],1]:
                    bboxs[mask[i,j],1] = j
                if j+1 > bboxs[mask[i,j],3]:
                    bboxs[mask[i,j],3] = j+1
    
    keep = centroids[:,0]!=0
    centroids = centroids[keep,:]
    bboxs = bboxs[keep,:]
    centroids[:,1]/=centroids[:,0]
    centroids[:,2]/=centroids[:,0]
    return centroids[:,1:], bboxs[:,:]

def get_subimg(image, mask, mask_nuc, cellid, cellcentroid, cellbbox, rmax=10, scale=1):
    subimg, p0 = get_subimg_part(image, mask, mask_nuc, cellid, cellbbox, cellcentroid, rmax=rmax)
    subimg, p0 = scale_subimg(subimg, p0, scale=scale)
    return subimg, p0

@njit(cache=True, parallel=True)
def calculate_radial_intensities(subimg, tidn, rs, phis, p0):
    img_bbox, kernel_bbox = create_bbox(p0, tidn.shape, subimg.shape)

    tsubimg = subimg[:,img_bbox[0]:img_bbox[2],img_bbox[1]:img_bbox[3]]
    tidnf = tidn[kernel_bbox[0]:kernel_bbox[2],kernel_bbox[1]:kernel_bbox[3]]
    lrs = len(rs)
    # lphis = len(phis)

    resdmat = np.zeros((tsubimg.shape[0],len(phis),len(rs)), dtype=np.float64)
    nmat = np.zeros((len(phis),len(rs)), dtype=np.int64)
    for i in range(tidnf.shape[0]):
        for j in range(tidnf.shape[1]):
            if tidnf[i,j] > 0:
                tphi = np.int64((tidnf[i,j]-1)//lrs) # phi
                trs = np.int64((tidnf[i,j]-1)%lrs) # rs
                resdmat[:,tphi , trs] += tsubimg[:,i,j]
                nmat[tphi , trs] += 1 
    resmat = np.zeros((tsubimg.shape[0],len(phis),len(rs)), dtype=np.float64)
    for i in range(resmat.shape[0]):
        for j in range(resmat.shape[1]):
            resmat[i,j,:] = np.cumsum(resdmat[i,j,:])
    resdmat /= np.clip(nmat, a_min=1, a_max=None)
    nmatc = np.zeros(nmat.shape, dtype=np.int64)
    for i in range(nmat.shape[0]):
        nmatc[i,:] = np.cumsum(nmat[i,:])
    nmatc = np.clip(nmatc, a_min=1, a_max=None)
    resmat /= nmatc
    return resmat, resdmat

@njit(cache=True)
def nuclei_only_radial_intensities(resdmat, ch=-2, minp=1, thr=1):
    nuclei_boundary = np.zeros(resdmat.shape[1], dtype=np.int64)
    for i in range(resdmat.shape[1]):
        nuclei_boundary[i] = np.argmax(np.diff(resdmat[ch,i,minp:]>=thr)) + minp
    resdmatout = resdmat.copy()
    for i in range(resdmat.shape[1]):
        resdmatout[:,i,(nuclei_boundary[i]+1):] = 0
    return resdmatout

@njit(cache=True)
def cytoplasma_only_radial_intensities(resdmat, ch1=-2, ch2=-5, minp=1, thr=1):
    nuclei_boundary = np.zeros(resdmat.shape[1], dtype=np.int64)
    for i in range(resdmat.shape[1]):
        nuclei_boundary[i] = np.argmax(np.diff(resdmat[ch1,i,minp:]>=thr)) + minp
    cell_boundary = np.zeros(resdmat.shape[1], dtype=np.int64)
    for i in range(resdmat.shape[1]):
        cell_boundary[i] = np.argmax(np.diff(resdmat[ch2,i,minp:]>=thr)) + minp
    
    resdmatc = np.zeros(resdmat.shape)
    inds = np.arange(resdmat.shape[1])[(cell_boundary - nuclei_boundary)>0]
    for i in inds:
        if nuclei_boundary[i] < cell_boundary[i]:
            resdmatc[:,i,:(cell_boundary[i]-nuclei_boundary[i])] = resdmat[:,i,nuclei_boundary[i]+1:cell_boundary[i]+1]
    return resdmatc

@njit(cache=True)
def nocell_radial_intensities(resdmat, ch=-5, minp=1, thr=1):
    cell_boundary = np.zeros(resdmat.shape[1], dtype=np.int64)
    for i in range(resdmat.shape[1]):
        cell_boundary[i] = np.argmax(np.diff(resdmat[ch,i,minp:]>=thr)) + minp

    resdmatc = np.zeros(resdmat.shape)
    for i in range(resdmat.shape[1]):
        resdmatc[:,i,:(resdmat.shape[2]-cell_boundary[i]-1)] = resdmat[:,i,(cell_boundary[i]+1):resdmat.shape[2]]
    return resdmatc


@njit(cache=True)
def interp_weight(x, a=-0.5):
    if x < 1:
        return 1-(a+3)*x**2 + (a+2)*x**3
    elif np.abs(x) < 2:
        v = a*(x-1)*(x-2)**2
        if v >= 0:
            return v
        else:
            return 0
    else:
        return 0


@njit(cache=True)
def resize_1d(fp, extent):
    yp = np.arange(extent)
    xp = np.linspace(0, extent-1, num=len(fp))
    if len(fp)<extent+2:
        return np.interp(yp,xp,fp)
    else:
        ip = np.zeros(extent)
        for i in range(len(yp)):
            d = np.abs(np.array([xp[j]-yp[i] for j in range(len(xp))]))
            ws = np.array([interp_weight(x) for x in d])
            ws = ws/np.sum(ws)
            ip[i] = np.sum(fp*ws)
        return ip

@njit(cache=True)
def scale_cell_boundary(resdmatacelli, extent, ch=-5, invert=False, ch_to_fill = np.array([-6,-5,-3,-2])):
    # resdmatacelli = resdmata[i,:,:,:].copy()
    # resdmatacelli[ch,:,:]
    # outer boundary
    cms = [np.argmax(np.cumsum(resdmatacelli[ch,i,:]))+1 for i in range(resdmatacelli.shape[1])]
    # inner boundary (technical artifact)
    cmis = [np.argmax(resdmatacelli[ch,i,:]>0) for i in range(resdmatacelli.shape[1])]
    for chtf in ch_to_fill:
        for i in range(resdmatacelli.shape[1]):
            resdmatacelli[chtf,i,:cms[i]] = 1
    if invert:
        cmis = cms
        # cms = [resdmatacelli.shape[2]-i for i in cms]
        cms = [resdmatacelli.shape[2]+1]*resdmatacelli.shape[1]
    outm = np.zeros((resdmatacelli.shape[0],resdmatacelli.shape[1],extent))
    for j in range(resdmatacelli.shape[0]):
        for i in range(resdmatacelli.shape[1]):
            if cms[i] > 0:
                # outm[j,i,:] = cv2.resize(resdmatacelli[j,i,:cms[i]], (1, extent), interpolation=cv2.INTER_LINEAR).T
                outm[j,i,:] = resize_1d(resdmatacelli[j,i,cmis[i]:cms[i]],  extent)
    return outm

@njit(cache=True)
def scale_cell_boundary_all(resdmata, extent, ch=-5, invert=False, ch_to_fill = np.array([-6,-5,-3,-2])):
    # resdmata = inncc.copy()
    resdmataout = np.zeros((resdmata.shape[0],resdmata.shape[1],resdmata.shape[2],extent), dtype=np.float64)
    for i in range(resdmata.shape[0]):
        resdmataout[i,:,:,:] = scale_cell_boundary(resdmata[i,:,:,:], extent, ch=ch, invert=invert, ch_to_fill=ch_to_fill)
    return resdmataout


def get_kernel_from_output(inmat, scale=1):
    rmax = inmat.shape[3]
    rs = np.linspace(1,rmax*scale,rmax*scale)
    nangles = inmat.shape[2]
    phis = np.linspace(np.pi/nangles,2*np.pi-np.pi/nangles,nangles)-np.pi/(nangles)
    epsphi = np.pi/len(phis)-1e-6
    tis = (rmax*scale*2+1, rmax*scale*2+1)

    # across cells
    tils = create_kernel_list(tis,rs,phis,epsphi)
    tidls = create_kernel_diff_list(tils)
    np.sum(np.stack(tidls, axis=0), axis=0)
    tidnls = [(tidls[i]*(i+1)).astype(np.uint64) for i in range(len(tidls))]
    tidn = np.sum(np.stack(tidnls, axis=0), axis=0)
    return tidn


# cellid = 0
# inmat = np.concatenate([incellsc,innnsc], axis=3)
def to_euclid_coords(ind, inmat, tidn = None, scale=1):
    rmax = inmat.shape[3]
    lrs = rmax*scale 
    if tidn is None:
        tidn = get_kernel_from_output(inmat, scale=scale)
    cellroundimgall = np.zeros([inmat.shape[1],tidn.shape[0], tidn.shape[1]])
    for i in range(tidn.shape[0]):
        for j in range(tidn.shape[1]):
            if tidn[i,j] > 0:
                tphi = np.int64((tidn[i,j]-1)//lrs) # phi
                trs = np.int64((tidn[i,j]-1)%lrs) # rs
                if scale == 1:
                    cellroundimgall[:,i,j] = inmat[ind,:,tphi,trs]
                else:
                    for k in range(inmat.shape[1]):
                        cellroundimgall[k,i,j] = resize_1d(inmat[ind,k,tphi,:],int(inmat.shape[3]*scale))[trs]
    return cellroundimgall


def get_subimg_all(ind, cellid, image, mask_all, mask_all_nuc, rmax=10, scale=1, return_centroid=False):
    mask_nuc = mask_all_nuc[0,:,:].astype(int)
    centroidarr, bboxarr = bbox_centroids(mask_nuc)

    subimg, p0 = get_subimg(image, mask_all, mask_all_nuc, cellid, centroidarr[ind,:], bboxarr[ind,:], rmax=rmax, scale=scale)
    tbbox, tbboxti = create_bbox(p0, (rmax*2+1,rmax*2+1), subimg.shape)
    tsubimg = subimg[:,tbbox[0]:tbbox[2],tbbox[1]:tbbox[3]]
    if return_centroid:
        p0 = p0 - np.array([tbbox[0],tbbox[1]])
        return tsubimg, p0
    else:
        return tsubimg

