import skimage
import numpy as np
import numpy.typing as npt
import scipy.ndimage
import polars as pl


def get_cell_intensities(img: npt.ArrayLike, mask: npt.ArrayLike, names: list[str], type: str = "mean") -> pl.DataFrame:
    assert img.shape[1:] == mask.shape
    assert img.ndim == 3
    if img.shape[0]==1 and isinstance(names, str):
        pass
    else:
        assert len(names) == img.shape[0]
    rp = skimage.measure.regionprops(mask)
    xy = np.array([r.centroid for r in rp])
    cellids = np.array([r.label for r in rp])
    mask_long = mask.ravel().astype(int)
    img_long = np.stack([img[i,:,:].ravel() for i in range(img.shape[0])])
    img_long = img_long[:,mask_long != 0]
    mask_long = mask_long[mask_long != 0]
    if type == "mean":
        data = {
            name: scipy.ndimage.mean(img_long[i,:], labels=mask_long, index=cellids) for i,name in enumerate(names)
        }
    elif type == "max":
        data = {
            name: scipy.ndimage.maximum(img_long[i,:], labels=mask_long, index=cellids) for i,name in enumerate(names)
        }
    elif type == "std":
        data = {
            name: scipy.ndimage.standard_deviation(img_long[i,:], labels=mask_long, index=cellids) for i,name in enumerate(names)
        }
    else:
        raise ValueError("type must be one of 'mean', 'max', 'min', 'std'")
    df = pl.DataFrame(
        data=data
    )
    df = df.with_columns(ObjectNumber=cellids)
    df = df.with_columns(area = scipy.ndimage.sum(mask>0, labels=mask, index=cellids))
    df = df.with_columns(x=xy[:,0]).with_columns(y=xy[:,1])
    return df

