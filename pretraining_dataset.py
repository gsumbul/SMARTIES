import torch
from utils.data_utils import aug_img_pair, ImageNormalizeStandardize, random_resized_crop_img_pair, merge_rasterio, patchify
import numpy as np
import os
import rasterio
import functools

class PretrainingCollateFn:
    def __call__(self, samples):
        first_img_patches, second_img_patches, first_proj_indices, second_proj_indices = zip(*samples)
        return torch.cat(
            first_img_patches, 0), torch.cat(
                second_img_patches, 0), torch.cat(
                        first_proj_indices, 0), torch.cat(
                            second_proj_indices, 0)

class PretrainingPairsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_size,
        sensor_specs,
        spectrum_specs,
        patch_size,
        first_file_reader=rasterio.open,
        second_file_reader=rasterio.open,
        **kwargs
    ):
        self.input_size = input_size
        self.patch_size = patch_size
        self.nb_patch_length = int(self.input_size / self.patch_size)
        self.projection_conversion = {i: spectrum_specs[i]['projection_idx'] for i in spectrum_specs}
        self.bands = {sensor_specs[sensor]['sensor_idx']: np.array(sensor_specs[sensor]['bands']).astype('int')[np.array(sensor_specs[sensor]['selected_bands']).astype('int')] for sensor in sensor_specs}
        self.projection_indices = {
            sensor_specs[sensor]['sensor_idx']: np.array(
                [self.projection_conversion[i] for i in self.bands[sensor_specs[sensor]['sensor_idx']]]) for sensor in sensor_specs}
        self.normalize_standardize = ImageNormalizeStandardize(sensor_specs) 
        self.first_file_reader = first_file_reader
        self.second_file_reader = second_file_reader

    def __getitem__(self, index: int):
        first_path, second_path = self.samples[index]

        with self.first_file_reader(first_path, 'r') as first_image_ds, self.second_file_reader(second_path, 'r') as second_image_ds:
            first_img = first_image_ds.read().astype(np.float32)
            second_img = second_image_ds.read().astype(np.float32)

        first_sensor_idx = 0 if first_img.shape[0] == 3 else 2
        first_img, _ = self.normalize_standardize(first_img, sensor_idx=first_sensor_idx)

        second_sensor_idx = 1 if second_img.shape[0] == 13 else 3
        second_img, _ = self.normalize_standardize(second_img, sensor_idx=second_sensor_idx)

        first_img, second_img = aug_img_pair(first_img, second_img)
        first_img, second_img = random_resized_crop_img_pair(first_img, second_img, self.input_size, scale=(0.25, 1))
        
        first_img_patches, second_img_patches = patchify(
                first_img, self.patch_size), patchify(
                    second_img, self.patch_size)

        first_nb_band = first_img_patches.shape[2]
        second_nb_band = second_img_patches.shape[2]

        first_img_patches = np.tile(first_img_patches, (1,1,int(12/first_nb_band),1,1))
        second_img_patches = np.tile(second_img_patches, (1,1,int(12/second_nb_band),1,1))
        
        first_proj_indices = np.tile(self.projection_indices[first_sensor_idx].reshape(
            1,1,-1), (self.nb_patch_length, self.nb_patch_length, int(12/first_nb_band))).astype(np.int32)
        
        second_proj_indices = np.tile(self.projection_indices[second_sensor_idx].reshape(
            1,1,-1), (self.nb_patch_length, self.nb_patch_length, int(12/second_nb_band))).astype(np.int32)
            
        return (
            torch.as_tensor(np.expand_dims(first_img_patches,0).astype(np.float32)), 
            torch.as_tensor(np.expand_dims(second_img_patches,0).astype(np.float32)),
            torch.as_tensor(np.expand_dims(first_proj_indices,0)),
            torch.as_tensor(np.expand_dims(second_proj_indices,0))
        )
    
    def __len__(self):
        return len(self.samples)

class ImgPairList:
    def __init__(
        self,
        first_root_path,
        second_root_path,
        img_pair_list_path
    ):
        with open(img_pair_list_path) as fp:
            self.img_pair_list = [line.strip().split() for line in fp]

        samples = [
            (os.path.join(first_root_path, first_img_f_name), os.path.join(second_root_path, second_img_f_name))
            for first_img_f_name, second_img_f_name in self.img_pair_list
        ]
        self.samples = samples

class BigEarthNetPairs(ImgPairList, PretrainingPairsDataset):
    first_BAND_NAMES = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    first_REF_PROFILE_BAND = 'B04'
    second_BAND_NAMES = ['VV','VH']
    second_REF_PROFILE_BAND = 'VV'
    def __init__(self, **kwargs):
        super().__init__(first_root_path=kwargs['first_root_path'], second_root_path=kwargs['second_root_path'], img_pair_list_path=kwargs['img_pair_list_path'])
        PretrainingPairsDataset.__init__(self, **kwargs)
        self.first_file_reader = functools.partial(merge_rasterio, bands=self.first_BAND_NAMES, ref_band=self.first_REF_PROFILE_BAND, out_size=(120,120))
        self.second_file_reader = functools.partial(merge_rasterio, bands=self.second_BAND_NAMES, ref_band=self.second_REF_PROFILE_BAND, out_size=(120,120))

class fMoWPairs(ImgPairList, PretrainingPairsDataset):
    def __init__(self, **kwargs):
        super().__init__(first_root_path=kwargs['first_root_path'], second_root_path=kwargs['second_root_path'], img_pair_list_path=kwargs['img_pair_list_path'])
        PretrainingPairsDataset.__init__(self, **kwargs)

def build_pretrain_loader(args):
    pretraining_ds = build_pretrain_dataset(args)
    collate_fn = PretrainingCollateFn()
    return torch.utils.data.DataLoader(
        pretraining_ds,
        batch_size=args.batch_size,
        num_workers=args.nb_workers_per_gpu,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_fn,
        shuffle=True,
        batch_sampler=None,
        prefetch_factor=args.prefetch_factor if args.nb_workers_per_gpu > 0 else None,
        persistent_workers=True if args.nb_workers_per_gpu > 0 else False
    )

def build_pretrain_dataset(args):
    return torch.utils.data.ConcatDataset([
        fMoWPairs(
            first_root_path='data/fMoW/',
            second_root_path='data/fMoW-S2/fmow-sentinel/',
            img_pair_list_path='eval_splits/fMoW-train-pairs-nontemporal.txt',
            input_size=args.input_size,
            sensor_specs=args.sensors_specs,
            spectrum_specs=args.spectrum_specs,
            patch_size=args.patch_size,
            file_reader=rasterio.open
        ),
        BigEarthNetPairs(
            first_root_path='data/BigEarthNet-v1/S2/',
            second_root_path='data/BigEarthNet-v1/S1/',
            img_pair_list_path='eval_splits/BigEarthNet-train-pairs-nontemporal.txt',
            input_size=args.input_size,
            sensor_specs=args.sensors_specs,
            spectrum_specs=args.spectrum_specs,
            patch_size=args.patch_size,
            file_reader=rasterio.open
        )
    ])