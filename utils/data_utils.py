import rasterio
from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling
from rasterio.crs import CRS
import numpy as np
import math
from skimage.transform import resize, rescale
from contextlib import contextmanager  
import os
from numpy.lib.stride_tricks import as_strided

class ImageNormalizeStandardize():
    def __init__(self, sensors_specs, norm_type='quantile'):
        if 'percentiles' in sensors_specs:
            self.sensor_idx = -1 if not 'sensor_idx' in sensors_specs else sensors_specs['sensor_idx']
            sensors_specs['sensor_idx'] = self.sensor_idx
            sensors_specs = {'N/A': sensors_specs}
            
        self.bands = {sensors_specs[key]['sensor_idx']: np.array(sensors_specs[key]['bands']) for key in sensors_specs}
        self.selected_band_indices =  {sensors_specs[key]['sensor_idx']: np.array(sensors_specs[key]['selected_bands']).astype('int') for key in sensors_specs}

        self.min_value = {sensors_specs[key]['sensor_idx']: np.array(sensors_specs[key]['percentiles']['1']).reshape(-1,1,1) for key in sensors_specs}
        self.max_value = {sensors_specs[key]['sensor_idx']: np.array(sensors_specs[key]['percentiles']['99']).reshape(-1,1,1) for key in sensors_specs}
        self.mean = {sensors_specs[key]['sensor_idx']: np.array(sensors_specs[key]['mean']).reshape(-1,1,1) for key in sensors_specs}
        self.std = {sensors_specs[key]['sensor_idx']: np.array(sensors_specs[key]['std']).reshape(-1,1,1) for key in sensors_specs}

    def __call__(self, x, sensor_idx=None, nodata_mask=None):
        if sensor_idx is None:
            sensor_idx = self.sensor_idx
        band_indices = self.selected_band_indices[sensor_idx]
        x = x[band_indices]
        max_val = self.max_value[sensor_idx][band_indices]
        min_val = self.min_value[sensor_idx][band_indices]
        mean_val = self.mean[sensor_idx][band_indices]
        std_val = self.std[sensor_idx][band_indices]

        x = (x - min_val) / (max_val - min_val)
        x = np.clip(x, 0, 1).astype(np.float32)
        x = (x - mean_val) / std_val

        if not (nodata_mask is None):
            x[:,nodata_mask] = 0

        return x, self.bands[sensor_idx][band_indices]
    
class ImageNormalize():
    def __init__(self, sensors_specs, norm_type='quantile', selected_bands=None):
        if 'percentiles' in sensors_specs:
            self.sensor_idx = -1 if not 'sensor_idx' in sensors_specs else sensors_specs['sensor_idx']
            sensors_specs['sensor_idx'] = self.sensor_idx
            sensors_specs = {'N/A': sensors_specs}
            if not (selected_bands is None):
                sensors_specs['N/A']['selected_bands'] = selected_bands
            
        self.bands = {sensors_specs[key]['sensor_idx']: np.array(sensors_specs[key]['bands']).astype('int') for key in sensors_specs}
        self.selected_band_indices =  {sensors_specs[key]['sensor_idx']: np.array(sensors_specs[key]['selected_bands']).astype('int') for key in sensors_specs}

        self.min_value = {sensors_specs[key]['sensor_idx']: np.array(sensors_specs[key]['percentiles']['1']).reshape(-1,1,1) for key in sensors_specs}
        self.max_value = {sensors_specs[key]['sensor_idx']: np.array(sensors_specs[key]['percentiles']['99']).reshape(-1,1,1) for key in sensors_specs}

    def __call__(self, x, sensor_idx=None, nodata_mask=None):
        if sensor_idx is None:
            sensor_idx = self.sensor_idx
        band_indices = self.selected_band_indices[sensor_idx]
        x = x[band_indices]
        max_val = self.max_value[sensor_idx][band_indices]
        min_val = self.min_value[sensor_idx][band_indices]
        img = (x - min_val) / (max_val - min_val)
        img = np.clip(img, 0, 1).astype(np.float32)
        if not (nodata_mask is None):
            img[:,nodata_mask] = 0

        return img, self.bands[sensor_idx][band_indices]

@contextmanager
def merge_rasterio(root_path, mode, bands, ref_band, out_size=(264,264)):
    chs = []
    profile = None
    for band in bands:
        img_f_name = os.path.basename(root_path)
        band_path = os.path.join(root_path, f'{img_f_name}_{band}.tif')
        with rasterio.open(band_path, mode) as ds:
            ch = ds.read(1,
                out_shape=(
                    1,
                    out_size[0],
                    out_size[1]
                ),
                resampling=Resampling.bilinear
            )
            if ref_band == band:
                profile = ds.profile
        chs.append(ch.squeeze())
    img = np.stack(chs, axis=0) # [C,264,264]
    profile.update({"count": len(bands)})

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset: # Open as DatasetWriter
            dataset.write(img)
            del img
            del chs
        with memfile.open() as dataset:  # Reopen as DatasetReader
            yield dataset  # Note yield not return  `

def read_image(filepath, patch_size=None):
    if 'tif' in filepath:
        with rasterio.open(filepath, 'r') as src:
            # return (C, H, W)
            if patch_size:
                img = src.read(
                    out_shape=(
                        src.count,
                        patch_size[0],
                        patch_size[1]
                    ),
                    resampling=Resampling.bilinear
                )
            else:
                img = src.read()
        return img.astype(np.float32) 

@contextmanager
def resample_rasterio_ds(rasterio_ds, scale):
    if scale == 1.0:
        yield rasterio_ds
    else:
        height = int(rasterio_ds.height * scale)
        width = int(rasterio_ds.width * scale)
        profile = rasterio_ds.profile
        data = rasterio_ds.read(
                out_shape=(rasterio_ds.count, height, width),
                resampling=Resampling.bilinear,
            )
        
        if not (rasterio_ds.crs is None):
            t = rasterio_ds.transform
            transform = Affine(t.a / scale, t.b, t.c, t.d, t.e / scale, t.f)
            profile.update(transform=transform, driver='GTiff', height=height, width=width)
        else:
            profile.update(driver='JPEG', height=height, width=width)

        with MemoryFile() as memfile:
            with memfile.open(**profile) as dataset: # Open as DatasetWriter
                dataset.write(data)
                del data

            with memfile.open() as dataset:  # Reopen as DatasetReader
                yield dataset  # Note yield not return  

def meters_to_degrees(spatial_resolution_meters, latitude=0):
  """Converts spatial resolution from meters to degrees for a given latitude.

  Args:
    spatial_resolution_meters: Spatial resolution in meters.
    latitude: Latitude in degrees.

  Returns:
    Spatial resolution in degrees.
  """

  earth_radius = 6371000  # Earth's radius in meters (approximate)
  degrees_per_meter = 1 / (math.pi * earth_radius / 180)
  degrees_resolution = spatial_resolution_meters * degrees_per_meter * math.cos(math.radians(latitude))
  return degrees_resolution

@contextmanager
def georef_rasterio_ds(rasterio_ds, resolution, apply_georef):
    if (not (rasterio_ds.crs is None)) or (not apply_georef):
        yield rasterio_ds
    else:
        # Arbitrary top-left coordinates 
        degrees_resolution = meters_to_degrees(resolution)
        top_left_x = 0
        top_left_y = 0
        
        transform = Affine(degrees_resolution, 0.0, 
                           top_left_x, 0.0, -degrees_resolution, top_left_y)

        profile = rasterio_ds.profile
        profile.update(transform=transform, crs=CRS.from_epsg(4326))

        data = rasterio_ds.read(
                out_shape=(rasterio_ds.count, rasterio_ds.height, rasterio_ds.width)
            )
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dataset: # Open as DatasetWriter
                dataset.write(data)
                del data

            with memfile.open() as dataset:  # Reopen as DatasetReader
                yield dataset  # Note yield not return  

def vertical_flip(img, p=0.5):
    if np.random.rand() < p:
        img = img[:,:,::-1]
    return img

def horizontal_flip(img, p=0.5):
    if np.random.rand() < p:
        img = img[:,::-1,:]
    return img

def rotate(img):
    rot_angle = np.random.choice(4) # angle = rot_angle*90
    return np.rot90(img,k=rot_angle,axes=(2,1))

def aug_img(img):
    img = vertical_flip(img)
    img = horizontal_flip(img)
    img = rotate(img)
    return img

def vertical_flip_pair(img1, img2, p=0.5):
    if np.random.rand() < p:
        img1 = img1[:,:,::-1]
        img2 = img2[:,:,::-1]
    return img1, img2

def horizontal_flip_pair(img1, img2, p=0.5):
    if np.random.rand() < p:
        img1 = img1[:,::-1,:]
        img2 = img2[:,::-1,:]
    return img1, img2

def rotate_pair(img1, img2):
    rot_angle = np.random.choice(4) # angle = rot_angle*90
    return np.rot90(img1,k=rot_angle,axes=(2,1)), np.rot90(img2,k=rot_angle,axes=(2,1))

def aug_img_pair(img1, img2):
    img1, img2 = vertical_flip_pair(img1, img2)
    img1, img2 = horizontal_flip_pair(img1, img2)
    img1, img2 = rotate_pair(img1, img2)
    return img1, img2

def batch_patchify(images, patch_size):
    """
    Split a batch of images into patches for images with shape (batch_size, channels, height, width).

    Parameters:
        images (np.ndarray): Batch of images, of shape (batch_size, channels, height, width).
        patch_size (int, int): Patch size as (patch_height, patch_width).

    Returns:
        np.ndarray: Patchified batch of images, with shape 
                    (batch_size, channels, num_patches_y, num_patches_x, patch_height, patch_width).
    """
    batch_size, channels, img_height, img_width = images.shape
    patch_height = patch_width = patch_size

    # Calculate the number of patches along each dimension
    num_patches_y = img_height // patch_height
    num_patches_x = img_width // patch_width

    # Use as_strided to create the patchified view
    patches = np.lib.stride_tricks.as_strided(
        images,
        shape=(batch_size, num_patches_y, num_patches_x, channels, patch_height, patch_width),
        strides=(
            images.strides[0],
            images.strides[2] * patch_height,
            images.strides[3] * patch_width,
            images.strides[1],
            images.strides[2],
            images.strides[3]
        ), 
        writeable=True        
    )

    return patches

def patchify(img, patch_size):
    patch_shape = (patch_size, patch_size)
    patch_shape = (img.shape[0],) + patch_shape
    img_shape = np.array(img.shape)
    patch_shape = np.array(patch_shape, dtype=img_shape.dtype)

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in patch_shape)
    window_strides = np.array(img.strides)

    indexing_strides = img[slices].strides

    win_indices_shape = (
        (np.array(img.shape) - np.array(patch_shape)) // np.array(patch_shape)
    ) + 1

    new_shape = tuple(list(win_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    return as_strided(img, shape=new_shape, strides=strides, writeable=True)[0]

def scale_center_crop_image(img, out_size, order=3):
    mid_size = (256/224) * out_size
    scale = max(mid_size / img.shape[1], mid_size/ img.shape[2])
    img = rescale(img.transpose(1,2,0), (scale,scale,1), order=order, anti_aliasing=True).transpose(2,0,1)
    center_x = int(img.shape[1]/2)
    center_y = int(img.shape[2]/2)
    out_legnth_x = min(out_size, img.shape[1])
    out_length_y = min(out_size, img.shape[2])
    img = crop_img(img, max(0, center_x-int(out_legnth_x/2)), max(0, center_y-int(out_length_y/2)), out_legnth_x, out_length_y)
    return img

def random_resized_crop_img_pair_get_params(img1, img2, ratio, scale):
    _, height_1, width_1 = img1.shape
    _, height_2, width_2 = img2.shape
    area_1 = height_1 * width_1
    area_2 = height_2 * width_2
    log_ratio = np.log(ratio)

    for _ in range(10):
        random_scale = np.random.uniform(scale[0], scale[1])
        target_area_1 = area_1 * random_scale
        target_area_2 = area_2 * random_scale
        
        aspect_ratio = np.exp(np.random.uniform(log_ratio[0], log_ratio[1]))

        w_1 = int(round(math.sqrt(target_area_1 * aspect_ratio)))
        h_1 = int(round(math.sqrt(target_area_1 / aspect_ratio)))
        
        w_2 = int(round(math.sqrt(target_area_2 * aspect_ratio)))
        h_2 = int(round(math.sqrt(target_area_2 / aspect_ratio)))

        if 0 < w_1 <= width_1 and 0 < h_1 <= height_1 and 0 < w_2 <= width_2 and 0 < h_2 <= height_2:
            i_1 = np.random.randint(0, height_1 - h_1 + 1)
            j_1 = np.random.randint(0, width_1 - w_1 + 1)
            i_2 = int(round((i_1 / (height_1 - h_1 + 1)) * (height_2 - h_2 + 1)))
            j_2 = int(round((j_1 / (width_1 - w_1 + 1)) * (width_2 - w_2 + 1)))
            return i_1, j_1, h_1, w_1, i_2, j_2, h_2, w_2

    # Fallback to central crop
    in_ratio = float(width_1) / float(height_1)
    if in_ratio < min(ratio):
        w_1 = width_1
        h_1 = int(round(w_1 / min(ratio)))
        w_2 = width_2
        h_2 = int(round(w_2 / min(ratio)))        
    elif in_ratio > max(ratio):
        h_1 = height_1
        w_1 = int(round(h_1 * max(ratio)))
        h_2 = height_2
        w_2 = int(round(h_2 * max(ratio)))
    else:  # whole image
        w_1 = width_1
        h_1 = height_1
        w_2 = width_2
        h_2 = height_2
    i_1 = (height_1 - h_1) // 2
    j_1 = (width_1 - w_1) // 2
    i_2 = (height_2 - h_2) // 2
    j_2 = (width_2 - w_2) // 2
    return i_1, j_1, h_1, w_1, i_2, j_2, h_2, w_2

def random_resized_crop_img_pair(img1, img2, out_size, ratio=(1.0, 1.0), scale=(0.2, 1.0)):
    i_1, j_1, h_1, w_1, i_2, j_2, h_2, w_2 = random_resized_crop_img_pair_get_params(img1, img2, ratio, scale)
    img1 = crop_img(img1, i_1, j_1, h_1, w_1)
    img2 = crop_img(img2, i_2, j_2, h_2, w_2)

    img1 = resize(img1.transpose(1,2,0), (out_size, out_size), order=3, anti_aliasing=True).transpose(2,0,1) #TODO try order=3
    img2 = resize(img2.transpose(1,2,0), (out_size, out_size), order=3, anti_aliasing=True).transpose(2,0,1)
    return img1, img2

def random_resized_crop_img_get_params(img1, ratio, scale):
    _, height_1, width_1 = img1.shape
    area_1 = height_1 * width_1
    log_ratio = np.log(ratio)

    for _ in range(10):
        target_area_1 = area_1 * np.random.uniform(scale[0], scale[1])
        aspect_ratio = np.exp(np.random.uniform(log_ratio[0], log_ratio[1]))

        w_1 = int(round(math.sqrt(target_area_1 * aspect_ratio)))
        h_1 = int(round(math.sqrt(target_area_1 / aspect_ratio)))
        
        if 0 < w_1 <= width_1 and 0 < h_1 <= height_1:
            i_1 = np.random.randint(0, height_1 - h_1 + 1)
            j_1 = np.random.randint(0, width_1 - w_1 + 1)
            return i_1, j_1, h_1, w_1

    # Fallback to central crop
    in_ratio = float(width_1) / float(height_1)
    if in_ratio < min(ratio):
        w_1 = width_1
        h_1 = int(round(w_1 / min(ratio)))
    elif in_ratio > max(ratio):
        h_1 = height_1
        w_1 = int(round(h_1 * max(ratio)))
    else:  # whole image
        w_1 = width_1
        h_1 = height_1
    i_1 = (height_1 - h_1) // 2
    j_1 = (width_1 - w_1) // 2
    return i_1, j_1, h_1, w_1

def random_resized_crop_img(img, out_size, ratio=(1.0, 1.0), scale=(0.2, 1.0), order=3):
    i_1, j_1, h_1, w_1 = random_resized_crop_img_get_params(img, ratio, scale)
    img = crop_img(img, i_1, j_1, h_1, w_1)
    img = resize(img.transpose(1,2,0), (out_size, out_size), order=order, anti_aliasing=True).transpose(2,0,1) #TODO try order=3
    return img

def crop_img(img, i, j, h, w):
    return img[:, i:i + h, j:j + w]