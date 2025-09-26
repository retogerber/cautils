import numpy as np
import polars as pl
import cv2
import numba
import tqdm
from itertools import product
from numba.experimental import jitclass
import scipy


spec = [
    ('A', numba.float32[:,:]),
    ('t', numba.float32[:]),
    ('c', numba.float32[:])
]
@jitclass(spec)
class AffineTransform:
    def __init__(self, A, t, c):
        self.A = A
        self.t = t
        self.c = c
    def get_matrix(self):
        M = np.zeros((2,3), dtype=np.float32)
        M[0:2,0:2] = self.A
        M[0,2] = self.A[0,0] * -self.c[0] + self.A[0,1] * -self.c[1] + self.c[0]
        M[1,2] = self.A[1,0] * -self.c[0] + self.A[1,1] * -self.c[1] + self.c[1]
        M[0:2,2] += self.t
        return M
    def get_matrix_c(self, c):
        M = np.zeros((2,3), dtype=np.float32)
        M[0:2,0:2] = self.A
        M[0,2] = self.A[0,0] * -c[0] + self.A[0,1] * -c[1] + c[0]
        M[1,2] = self.A[1,0] * -c[0] + self.A[1,1] * -c[1] + c[1]
        M[0:2,2] += self.t
        return M
    def set_center(self, c):
        self.c = c

# theta = 0.25*np.pi
# cos_t, sin_t = np.cos(theta), np.sin(theta)
# A = np.array([[cos_t, -sin_t],
#               [sin_t,  cos_t]], dtype=np.float32)
# # A = np.eye(2, dtype=np.float32)
# t = np.ones(2, dtype=np.float32) *2
# c = np.ones(2, dtype=np.float32)

# at = AffineTransform(A, t, c)
# at.get_matrix()
# at.get_matrix_c([0,0])

# import SimpleITK as sitk
# tr = sitk.AffineTransform(A.flatten().astype(np.float64), t.astype(np.float64), c.astype(np.float64))
# tr.GetMatrix()
# tr.GetTranslation()
# tr.GetCenter()
# tr.TransformPoint((1,0))
# at.get_matrix()[:2,:2].astype(np.float64) @ np.array([[1],[2]]) + at.get_matrix()[:2,2:3].astype(np.float64)

spec = [
    ('transforms', numba.types.ListType(AffineTransform.class_type.instance_type))
]
@jitclass(spec)
class AffineTransformList:
    def __init__(self, transforms):
        self.transforms = transforms
    def get_matrices(self):
        Mout = np.zeros((len(self.transforms), 2, 3), dtype=np.float32)
        for i in range(len(self.transforms)):
            Mout[i,:,:] = self.transforms[i].get_matrix()
        return Mout
    def set_center(self, center):
        for i in range(len(self.transforms)):
            self.transforms[i].set_center(center)
    def set_centers(self, centers):
        for i in range(len(self.transforms)):
            self.transforms[i].set_center(centers[i,:])
    def get_matrices_center(self, center):
        Mout = np.zeros((len(self.transforms), 2, 3), dtype=np.float32)
        for i in range(len(self.transforms)):
            Mout[i,:,:] = self.transforms[i].get_matrix_c(center)
        return Mout
    def get_matrices_centers(self, centers):
        Mout = np.zeros((len(self.transforms), 2, 3), dtype=np.float32)
        for i in range(len(self.transforms)):
            Mout[i,:,:] = self.transforms[i].get_matrix_c(centers[i,:])
        return Mout


# AffineTransformList(l).get_matrices()


@numba.jit(cache=False)
def create_affine_transform(translate: np.float32, rotate: np.float32, scale: np.float32, shear: np.float32, center: np.float32 = np.array((0,0), dtype=np.float32)):
    """Create AffineTransform from translation, rotation, scale, shear."""
    theta = rotate
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A_rot = np.array([[cos_t, -sin_t],
                      [sin_t,  cos_t]], dtype=np.float32)
    A_scale = np.array([[scale[0], 0],
                        [0, scale[1]]], dtype=np.float32)
    A_shear = np.array([[1, shear[0]],
                        [shear[1], 1]], dtype=np.float32)
    A = A_rot @ A_scale @ A_shear
    at = AffineTransform(A, translate, center)
    return at

# create_affine_transform(
#     np.array((1,2), dtype=np.float32), 
#     np.float32(0.25*np.pi), 
#     np.array((1,1), dtype=np.float32), 
#     np.array((0,0), dtype=np.float32), 
#     np.array((1,1), dtype=np.float32))

# create_affine_transform((1,2), 0.25*np.pi, (1,1), (0,0), (1,1)).get_matrix()
# create_affine_transform((1,2), 0.25*np.pi, (1,1), (0,0)).get_matrix_c((1,1))


@numba.jit(cache=False)
def create_affine_transforms(combs, center = np.array((0,0), dtype=np.float32)):
    l = numba.typed.List()
    for i in range(combs.shape[0]):
        at = create_affine_transform(
            combs[i,:2],
            combs[i,2],
            combs[i,3:5],
            combs[i,5:],
            center
        )
        l.append(at)
    return AffineTransformList(l)

# atl = create_affine_transforms(combs.astype(np.float32), np.array((10,10), dtype=np.float32))

# atl = create_affine_transforms(combs, (10,10))
# atl.transforms[0].set_center(np.array((5,5), dtype=np.float32))

# atl.set_center(np.array((5,5), dtype=np.float32))
# atl.set_centers(np.repeat(np.array((5,5), dtype=np.float32).reshape(1,2), (combs.shape[0]), axis=0))
# atl.get_matrices()


@numba.njit(cache=False)
def bbox_label(mask):
    maxval = int(np.max(mask) + 1)
    labels = np.zeros((maxval, 2), dtype=np.uint32)
    labels[:, 0] = np.arange(maxval)
    bboxs = np.zeros((maxval, 4), dtype=np.int64)
    bboxs[:, 0] = np.max(np.array(mask.shape)) + 1
    bboxs[:, 1] = np.max(np.array(mask.shape)) + 1
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] != 0:
                labels[mask[i, j], 1] = 1
                if i < bboxs[mask[i, j], 0]:
                    bboxs[mask[i, j], 0] = i
                if i + 1 > bboxs[mask[i, j], 2]:
                    bboxs[mask[i, j], 2] = i + 1
                if j < bboxs[mask[i, j], 1]:
                    bboxs[mask[i, j], 1] = j
                if j + 1 > bboxs[mask[i, j], 3]:
                    bboxs[mask[i, j], 3] = j + 1

    keep = labels[:, 1] == 1
    labels = labels[keep, 0]
    bboxs = bboxs[keep, :]
    return labels, bboxs[:, :]

# atl = create_affine_transforms(combs, np.array((10,10), dtype=np.float32))

def get_padding(mask,atl):
    lpad, rpad, tpad, bpad = 0,0,0,0
    label, bbox = bbox_label(mask)
    bbox_rel = bbox.copy()
    bbox_rel[:,2] = bbox_rel[:,2]-bbox_rel[:,0]
    bbox_rel[:,3] = bbox_rel[:,3]-bbox_rel[:,1]
    bbox_rel[:,0] = 0
    bbox_rel[:,1] = 0
    xdiff = bbox[:,2]-bbox[:,0]
    ydiff = bbox[:,3]-bbox[:,1]
    xydiff = xdiff*ydiff
    inds = np.intersect1d(np.argsort(xdiff)[-int(len(xdiff)*0.1):], np.argsort(ydiff)[-int(len(ydiff)*0.1):])
    inds = np.intersect1d(inds, np.argsort(xydiff)[-int(len(xydiff)*0.1):])
    bbox_rel = bbox_rel[inds,:]
    for l,b in zip(label,bbox_rel):
        Msc = atl.get_matrices_center(np.array(((b[3]-b[1]-1)/2, (b[2]-b[0]-1)/2), dtype=np.float32))
        for i in range(Msc.shape[0]):
            corners = np.array([[b[0], b[1]], [b[0], b[3]], [b[2], b[1]], [b[2], b[3]]])
            corners = (Msc[i,:2,:2] @ corners.T).T + Msc[i,:2,2:3].T
            lpad = np.max([lpad, b[0]-np.min(corners[:,0])])
            rpad = np.max([rpad, np.max(corners[:,0])-b[2]])
            tpad = np.max([tpad, b[1]-np.min(corners[:,1])])
            bpad = np.max([bpad, np.max(corners[:,1])-b[3]])
    pad = int(np.ceil(np.max([lpad, rpad, tpad, bpad])))+1
    return pad

def get_affine_options():
    return {
        "translation_lower": -1.0, 
        "translation_upper": 1.0, 
        "translation_steps": 3,
        "rotation_lower": -15/180*np.pi, 
        "rotation_upper": 15/180*np.pi, 
        "rotation_steps": 3,
        "scale_lower": 0.8, 
        "scale_upper": 1.2, 
        "scale_steps": 3,
        "shear_lower": -0.1, 
        "shear_upper": 0.1, 
        "shear_steps": 3
    }


def compose_affine_combinations(
    n=-1,
    translation_lower=-1.0, 
    translation_upper=1.0, 
    translation_steps=3,
    rotation_lower=-15/180*np.pi, 
    rotation_upper=15/180*np.pi, 
    rotation_steps=3,
    scale_lower=0.8, 
    scale_upper=1.2, 
    scale_steps=3,
    shear_lower=-0.1, 
    shear_upper=0.1, 
    shear_steps=3
    ):
    if n<0:
        scale_x_perm = np.linspace(scale_lower, scale_upper, num=scale_steps)
        scale_y_perm = np.linspace(scale_lower, scale_upper, num=scale_steps)
        rotation_perm = np.linspace(rotation_lower, rotation_upper, num=rotation_steps)
        translate_x_perm = np.linspace(translation_lower, translation_upper, num=translation_steps)
        translate_y_perm = np.linspace(translation_lower, translation_upper, num=translation_steps)
        shear_x_perm = np.linspace(shear_lower, shear_upper, num=shear_steps)
        shear_y_perm = np.linspace(shear_lower, shear_upper, num=shear_steps)

        combs = np.array(list(product(
            translate_x_perm[:],
            translate_y_perm[:],
            rotation_perm[:],
            scale_x_perm[:],
            scale_y_perm[:],
            shear_x_perm[:],
            shear_y_perm[:]
        )), dtype=np.float32)
    else:
        combs = np.empty((n,7), dtype=np.float32)
        combs[:,0] = np.random.uniform(translation_lower, translation_upper, size=n)
        combs[:,1] = np.random.uniform(translation_lower, translation_upper, size=n)
        combs[:,2] = np.random.uniform(rotation_lower, rotation_upper, size=n)
        combs[:,3] = np.random.uniform(scale_lower, scale_upper, size=n)
        combs[:,4] = np.random.uniform(scale_lower, scale_upper, size=n)
        combs[:,5] = np.random.uniform(shear_lower, shear_upper, size=n)
        combs[:,6] = np.random.uniform(shear_lower, shear_upper, size=n)
    return combs

def get_random_mask(mask, pad=None, atl=None, label=None, bbox=None, centers=None, seed=0):
    if pad is None:
        # create extrem cases of affine transformations to estimate padding
        pad = get_padding(mask, create_affine_transforms(compose_affine_combinations(translation_steps=2, rotation_steps=2, scale_steps=2, shear_steps=2)))

    if label is None or bbox is None or centers is None:
        label, bbox = bbox_label(mask)
        centers = np.stack(((bbox[:,2]-bbox[:,0]-1)/2, (bbox[:,3]-bbox[:,1]-1)/2), axis=1)
        # account for padding
        centers[:,0]  = centers[:,0]+bbox[:,0]-np.clip(bbox[:,0]-pad,0, mask.shape[0])
        centers[:,1]  = centers[:,1]+bbox[:,1]-np.clip(bbox[:,1]-pad,0, mask.shape[1])
        # pad the bbox
        bbox[:,0] = np.clip(bbox[:,0]-pad,0, mask.shape[0])
        bbox[:,1] = np.clip(bbox[:,1]-pad,0, mask.shape[1])
        bbox[:,2] = np.clip(bbox[:,2]+pad,0, mask.shape[0])
        bbox[:,3] = np.clip(bbox[:,3]+pad,0, mask.shape[1])
    
    if atl is None:
        combs = compose_affine_combinations(n=len(label))
        atl = create_affine_transforms(combs)
    np.random.seed(seed)
    perml = np.random.permutation(len(label))

    masker = np.zeros_like(mask)
    for i in perml:
        l = label[i]
        b = bbox[i]
        submask = (mask[b[0]:b[2], b[1]:b[3]]==l).astype(np.uint8)
        M = atl.transforms[i].get_matrix_c(centers[i,::-1])
        submask = cv2.warpAffine(submask, M[:2,:], submask.shape[::-1])
        masker[b[0]:b[2], b[1]:b[3]][submask.astype(bool)] = l
    masker = cv2.morphologyEx(masker.astype(np.uint16), cv2.MORPH_CLOSE, np.ones((3,3), dtype=np.uint8))
    return masker

def get_random_masks(mask, n=2, affine_options={}):
    # create extrem cases of affine transformations to estimate padding

    tmp_affine_options = get_affine_options()
    tmp_affine_options.update(affine_options)
    tmp_affine_options.update({
        "translation_steps": 2,
        "rotation_steps": 2,
        "scale_steps": 2,
        "shear_steps": 2
    })
    pad = get_padding(mask, create_affine_transforms(compose_affine_combinations(**tmp_affine_options)))

    label, bbox = bbox_label(mask)
    centers = np.stack(((bbox[:,2]-bbox[:,0]-1)/2, (bbox[:,3]-bbox[:,1]-1)/2), axis=1)
    # account for padding
    centers[:,0]  = centers[:,0]+bbox[:,0]-np.clip(bbox[:,0]-pad,0, mask.shape[0])
    centers[:,1]  = centers[:,1]+bbox[:,1]-np.clip(bbox[:,1]-pad,0, mask.shape[1])
    # pad the bbox
    bbox[:,0] = np.clip(bbox[:,0]-pad,0, mask.shape[0])
    bbox[:,1] = np.clip(bbox[:,1]-pad,0, mask.shape[1])
    bbox[:,2] = np.clip(bbox[:,2]+pad,0, mask.shape[0])
    bbox[:,3] = np.clip(bbox[:,3]+pad,0, mask.shape[1])
    
    combs = compose_affine_combinations(n=len(label), **affine_options)
    atl = create_affine_transforms(combs)

    masks = np.empty((n,mask.shape[0],mask.shape[1]), dtype=mask.dtype)
    for i in range(n):
        masks[i,:,] = get_random_mask(mask, pad, atl, label, bbox, centers, seed=i)
    return masks

def get_cell_intensities(img, mask, names, cellids, aggr_type="mean"):
    if len(cellids) == 0:
        return pl.DataFrame({
            "ObjectNumber": [],
            **{name: [] for name in names}
        })
    mask_long = mask.ravel().astype(int)
    img_long = np.stack([img[i,:,:].ravel() for i in range(img.shape[0])])
    img_long = img_long[:,mask_long != 0]
    mask_long = mask_long[mask_long != 0]
    if aggr_type == "area":
        data1 = {
            "area": scipy.ndimage.sum(mask_long>0, labels=mask_long, index=cellids)
        }
    else:
        data1 = {
            name: scipy.ndimage.mean(img_long[i,:], labels=mask_long, index=cellids) for i,name in enumerate(names)
        }
    data = {**data1}
    df = pl.DataFrame(
        data=data
    )
    df = df.with_columns(ObjectNumber=cellids)
    return df

def affine_medians(mask, image, channel_names, n_iter=10, affine_options={}):
    """
    Compute median of mean intensities after applying random affine transformations to each cell. Does create complete random masks for n_iter times.

    Parameters
    ----------
    mask : np.ndarray
        Labeled mask of shape (H, W) where each unique integer represents a different cell
    image : np.ndarray
        Multi-channel image of shape (C, H, W)
    channel_names : list of str
        List of channel names corresponding to the channels in the image
    n_iter : int, optional
        Number of random affine transformations to sample per cell. Default is 10.
    affine_options : dict, optional
        Dictionary specifying the ranges and steps for affine transformation parameters. See `get_affine_options` for default values.

    Returns
    -------
    df : pl.DataFrame
        DataFrame with median intensities for each cell and channel after affine transformations.
    """
    labels, bbox = bbox_label(mask)
    df = get_cell_intensities(mask[np.newaxis,:,:], mask, ["ch"], labels, aggr_type="area")

    masks = get_random_masks(mask, n=n_iter, affine_options=affine_options)
    for m in range(masks.shape[0]):
        df1 = get_cell_intensities(image, masks[m,:,:], [f"{ch}_{m}" for ch in channel_names], labels)
        df = df.join(df1, on="ObjectNumber", how="left", suffix=f"_{iter}")

    for ch in channel_names:
        df = df.with_columns(
            pl.concat_list([f"{ch}_{i}" for i in range(n_iter)]).list.median().alias(f"{ch}")
        )
        df = df.drop([f"{ch}_{i}" for i in range(n_iter)])
    return df


# import timeit 
# timeit.timeit("get_padding(mask, create_affine_transforms(compose_affine_combinations(translation_steps=2, rotation_steps=2, scale_steps=2, shear_steps=2)))", globals=globals(), number=10)/10
# timeit.timeit("get_random_mask(mask)", globals=globals(), number=10)
# timeit.timeit("get_random_mask(mask, pad=15)", globals=globals(), number=10)
# timeit.timeit("get_random_masks(mask, n=10)", globals=globals(), number=1)

# tifffile.imwrite("/home/retger/celltyping_imc/data/tmp/test_mask.tiff", mask)
# rmask = get_random_mask(mask, seed=0)
# tifffile.imwrite("/home/retger/celltyping_imc/data/tmp/test_mask_random.tiff", rmask)

# timeit.timeit("masks = get_random_masks(mask, n=10)", globals=globals(), number=1)
# timeit.timeit("affine_medians(mask, image, channel_names, n_iter=10)", globals=globals(), number=1)

def direct_affine_medians(mask, image, channel_names, n_iter=-1, sample_fraction=1.0, affine_options={}):
    """
    Compute median of mean intensities after applying random affine transformations to each cell. Does not create complete random masks, but applies transformations directly to each cell.

    Parameters
    ----------
    mask : np.ndarray
        Labeled mask of shape (H, W) where each unique integer represents a different cell
    image : np.ndarray
        Multi-channel image of shape (C, H, W)
    channel_names : list of str
        List of channel names corresponding to the channels in the image
    n_iter : int, optional
        Number of random affine transformations to sample per cell. If -1, use all combinations.
        Default is -1.
    sample_fraction : float, optional
        Fraction of total affine combinations to sample if n_iter is -1. Default is 1.0 (use all).
    affine_options : dict, optional
        Dictionary specifying the ranges and steps for affine transformation parameters.
    
    Returns
    -------
    df : pl.DataFrame
        DataFrame with median intensities for each cell and channel after affine transformations.
    """
    combs = compose_affine_combinations(**affine_options, n=n_iter)
    if sample_fraction<1.0:
        n_sample = int(combs.shape[0]*sample_fraction)
        print(f"Sampling {n_sample} out of {combs.shape[0]} affine transformations")
        idx = np.random.choice(np.arange(combs.shape[0]), size=n_sample, replace=False)
        combs = combs[idx,:]

    atl = create_affine_transforms(combs)
    pad = get_padding(mask,atl)
    label, bbox = bbox_label(mask)
    centers = np.stack(((bbox[:,2]-bbox[:,0]-1)/2, (bbox[:,3]-bbox[:,1]-1)/2), axis=1)
    # account for padding
    centers[:,0]  = centers[:,0]+bbox[:,0]-np.clip(bbox[:,0]-pad,0, mask.shape[0])
    centers[:,1]  = centers[:,1]+bbox[:,1]-np.clip(bbox[:,1]-pad,0, mask.shape[1])
    # pad the bbox
    bbox[:,0] = np.clip(bbox[:,0]-pad,0, mask.shape[0])
    bbox[:,1] = np.clip(bbox[:,1]-pad,0, mask.shape[1])
    bbox[:,2] = np.clip(bbox[:,2]+pad,0, mask.shape[0])
    bbox[:,3] = np.clip(bbox[:,3]+pad,0, mask.shape[1])

    n_iters = combs.shape[0]
    median_intensities = np.empty((len(label),image.shape[0]), dtype=float)
    for i in tqdm.tqdm(range(len(label))):
        l = label[i]
        b = bbox[i]
        submask = (mask[b[0]:b[2], b[1]:b[3]]==l).astype(np.uint8)
        cropped_img = image[:, b[0]:b[2], b[1]:b[3]]
        Msc = atl.get_matrices_center(centers[i,::-1])
        tmp_mean_intensities = np.empty((n_iters,image.shape[0]), dtype=float)
        for j in range(Msc.shape[0]):
            submask2 = cv2.warpAffine(submask, Msc[j,:2,:], submask.shape[::-1])
            tmp_mean_intensities[j,:]=np.mean(cropped_img[:,submask2==1], axis=(1))
        median_intensities[i,:] = np.median(tmp_mean_intensities, axis=0)
    df = pl.DataFrame(
        data={f"{ch}": median_intensities[:,i] for i,ch in enumerate(channel_names)}
    ).with_columns(ObjectNumber=label)
    return df
    
