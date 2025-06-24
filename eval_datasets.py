import torch
import os
import rasterio
import numpy as np
from utils.data_utils import aug_img_pair, scale_center_crop_image, aug_img, patchify, random_resized_crop_img, merge_rasterio, georef_rasterio_ds, resample_rasterio_ds, ImageNormalizeStandardize
import functools
import json
from skimage.transform import rescale
    
class ImgList:
    def __init__(
        self,
        root_path,
        img_list_path,
        img_list = None
    ):
        if img_list_path is None:
            self.img_list = img_list
        else:
            with open(img_list_path) as fp:
                self.img_list = [line.strip() for line in fp]
        classes, class_to_idx = self.find_classes(self.img_list)
        samples = [
            (os.path.join(root_path, img_f_name), class_to_idx[self.filename_to_class(img_f_name)])
            for img_f_name in self.img_list
        ]
        self.classes = classes
        self.nb_classes = len(self.classes)
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def filename_to_class(self, img_f_name):
        return os.path.dirname(img_f_name).split("/")[1]

    def find_classes(self, img_list):
        classes = sorted(list({self.filename_to_class(img_f_name) for img_f_name in img_list}))
        if len(classes) == 0:
            raise FileNotFoundError(f"Couldn't find any classes in filenames.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

class PtEvalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_path,
        input_size,
        sensor_specs,
        spectrum_specs,
        patch_size,
        scale,
        is_train,
        eval_type,
        is_patchify,
        **kwargs
    ):
        self.input_size = input_size
        self.patch_size = patch_size
        self.nb_patch_length = int(self.input_size / self.patch_size)
        self.resolution = None
        self.apply_georef = False
        self.scale = scale
        self.projection_conversion = {i: spectrum_specs[i]['projection_idx'] for i in spectrum_specs}
        self.bands = np.array(sensor_specs['bands'])
        self.selected_band_indices =  np.array(sensor_specs['selected_bands']).astype('int')
        self.projection_indices = np.array([self.projection_conversion[i] for i in self.bands[self.selected_band_indices]])
        self.normalize_standardize = ImageNormalizeStandardize(sensor_specs)
        df = torch.load(root_path) 
        self.is_train = is_train
        if self.is_train:
            self.images = df["train_imgs"] if 'DynamicWorld' in root_path else df["train_images"] 
            self.labels = df['train_labels'] if 'DynamicWorld' in root_path else df['train_labels']
        else:
            self.images = df["val_imgs"] if 'DynamicWorld' in root_path else df["validation_images"]
            self.labels = df['val_labels'] if 'DynamicWorld' in root_path else df["validation_labels"]
        self.images = self.images.float().numpy()
        self.labels[self.labels == -1] = -100
        del df
        self.eval_type = eval_type
        self.is_patchify = is_patchify

    def __getitem__(self, index: int):
        img = self.images[index]
        target = self.labels[index].reshape(1,96,96)
        if type(target) is np.ndarray:
            target = torch.as_tensor(target.reshape(1,-1))

        img, _ = self.normalize_standardize(img)
        
        img, target = self.apply_transforms(img, target)
        if self.is_patchify:
            img = patchify(img, self.patch_size)
        
        proj_indices = np.tile(self.projection_indices.reshape(
            1,1,-1), (self.nb_patch_length, self.nb_patch_length, 1)).astype(np.int32)
        
        return (
            torch.as_tensor(np.expand_dims(img,0).astype(np.float32)), 
            torch.as_tensor(np.expand_dims(proj_indices,0))
        ), target
    
    def __len__(self):
        return len(self.images)

    def apply_transforms(self, img, target):
        if self.is_train:
            target = target.numpy()
            img, target = aug_img_pair(img, target)
            target = torch.tensor(target.copy())
            scale = (self.input_size / img.shape[1], self.input_size/ img.shape[2], 1)
            img = rescale(img.transpose(1,2,0), scale, order=3, anti_aliasing=True).transpose(2,0,1)
        else:
            scale = (self.input_size / img.shape[1], self.input_size/ img.shape[2], 1)
            img = rescale(img.transpose(1,2,0), scale, order=3, anti_aliasing=True).transpose(2,0,1)
        return img, target
    
class EvalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_size,
        sensor_specs,
        spectrum_specs,
        patch_size,
        scale,
        is_train,
        eval_type,
        is_patchify,
        file_reader=rasterio.open,
        **kwargs
    ):
        self.input_size = input_size
        self.patch_size = patch_size
        self.nb_patch_length = int(self.input_size / self.patch_size)
        self.resolution = None
        self.apply_georef = False
        self.scale = scale
        self.projection_conversion = {i: spectrum_specs[i]['projection_idx'] for i in spectrum_specs}
        self.bands = np.array(sensor_specs['bands']).astype('int')
        self.selected_band_indices =  np.array(sensor_specs['selected_bands']).astype('int')
        self.projection_indices = np.array([self.projection_conversion[i] for i in self.bands[self.selected_band_indices]])
        self.normalize_standardize = ImageNormalizeStandardize(sensor_specs)
        self.file_reader = file_reader
        self.is_train = is_train
        self.eval_type = eval_type
        self.is_patchify = is_patchify

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        if type(target) is np.ndarray:
            target = torch.as_tensor(target.reshape(1,-1))

        with self.file_reader(path, 'r') as image_ds:
            with georef_rasterio_ds(image_ds, self.resolution, self.apply_georef) as image_ds_georef:
                with resample_rasterio_ds(image_ds_georef, self.scale) as image_ds_resampled:
                    img = image_ds_resampled.read().astype(np.float32)
        img, _ = self.normalize_standardize(img)
        
        img = self.apply_transforms(img)
        if self.is_patchify:
            img = patchify(img, self.patch_size)
        
        proj_indices = np.tile(self.projection_indices.reshape(
            1,1,-1), (self.nb_patch_length, self.nb_patch_length, 1)).astype(np.int32)
        
        return (
            torch.as_tensor(np.expand_dims(img,0).astype(np.float32)), 
            torch.as_tensor(np.expand_dims(proj_indices,0))
        ), target
    
    def __len__(self):
        return len(self.samples)
            
    def apply_transforms(self, img):
        if self.eval_type == 'kNN':
            scale = (self.input_size / img.shape[1], self.input_size/ img.shape[2], 1)
            img = rescale(img.transpose(1,2,0), scale, order=3, anti_aliasing=True).transpose(2,0,1)
        elif self.is_train:
            img = aug_img(img)
            scale = (self.input_size / img.shape[1], self.input_size/ img.shape[2], 1)
            img = rescale(img.transpose(1,2,0), scale, order=3, anti_aliasing=True).transpose(2,0,1)
        else:
            scale = (self.input_size / img.shape[1], self.input_size/ img.shape[2], 1)
            img = rescale(img.transpose(1,2,0), scale, order=3, anti_aliasing=True).transpose(2,0,1)
        return img

class EvalCollateFn:
    def __call__(self, samples): 
        data, target = zip(*samples)
        if torch.is_tensor(target[0]):
            target = torch.cat(target,0)
        else:
            target = torch.tensor(target)
        img_patches, proj_indices = zip(*data)
        return torch.cat(
            img_patches, 0), torch.cat(
                proj_indices), target

class EvalPairCollateFn:
    def __call__(self, samples): 
        data1, data2, target = zip(*samples)
        if torch.is_tensor(target[0]):
            target = torch.cat(target,0)
        else:
            target = torch.tensor(target)
        img1_patches, img1_projection_indices = zip(*data1)
        img2_patches, img2_projection_indices = zip(*data2)
        return (torch.cat(
            img1_patches, 0), torch.cat(
                img1_projection_indices)), (torch.cat(
                        img2_patches, 0), torch.cat(
                            img2_projection_indices)), target

class DFC2020(PtEvalDataset):
    def __init__(self, **kwargs):
        PtEvalDataset.__init__(self, **kwargs)

class EuroSAT_MS(ImgList, EvalDataset):
    def __init__(self, **kwargs):
        super().__init__(root_path=kwargs['root_path'], img_list_path=kwargs['img_list_path'])
        EvalDataset.__init__(self, **kwargs)

    def apply_transforms(self, img):
        if (self.eval_type == 'ft') and self.is_train:
            img = aug_img(img)
        scale = (self.input_size / img.shape[1], self.input_size/ img.shape[2], 1)
        return rescale(img.transpose(1,2,0), scale, order=3, anti_aliasing=True).transpose(2,0,1)

class RESISC45(ImgList, EvalDataset):
    def __init__(self, **kwargs):
        super().__init__(root_path=kwargs['root_path'], img_list_path=kwargs['img_list_path'])
        EvalDataset.__init__(self, **kwargs)

    def apply_transforms(self, img):
        if self.is_train:
            if self.eval_type == 'ft':
                img = aug_img(img)
                img = random_resized_crop_img(img, self.input_size, scale=(0.25, 1), order=3)
                return img
            else:
                return scale_center_crop_image(img, self.input_size, order=3)
        else:
            return scale_center_crop_image(img, self.input_size, order=3)

class UC_Merced(ImgList, EvalDataset):
    def __init__(self, **kwargs):
        super().__init__(root_path=kwargs['root_path'], img_list_path=kwargs['img_list_path'])
        EvalDataset.__init__(self, **kwargs)
    
    def apply_transforms(self, img):
        return scale_center_crop_image(img, self.input_size)

class WHU_RS19(ImgList, EvalDataset):
    def __init__(self, **kwargs):
        super().__init__(root_path=kwargs['root_path'], img_list_path=kwargs['img_list_path'])
        EvalDataset.__init__(self, **kwargs)

    def apply_transforms(self, img):
        scale = (self.input_size / img.shape[1], self.input_size/ img.shape[2], 1)
        return rescale(img.transpose(1,2,0), scale, order=1, anti_aliasing=True).transpose(2,0,1)
        
class BigEarthNet(EvalDataset):
    NEW_LABELS = [
        'Urban fabric',
        'Industrial or commercial units',
        'Arable land',
        'Permanent crops',
        'Pastures',
        'Complex cultivation patterns',
        'Land principally occupied by agriculture, with significant areas of natural vegetation',
        'Agro-forestry areas',
        'Broad-leaved forest',
        'Coniferous forest',
        'Mixed forest',
        'Natural grassland and sparsely vegetated areas',
        'Moors, heathland and sclerophyllous vegetation',
        'Transitional woodland/shrub',
        'Beaches, dunes, sands',
        'Inland wetlands',
        'Coastal wetlands',
        'Inland waters',
        'Marine waters'
    ]

    GROUP_LABELS = {
        'Continuous urban fabric': 'Urban fabric',
        'Discontinuous urban fabric': 'Urban fabric',
        'Non-irrigated arable land': 'Arable land',
        'Permanently irrigated land': 'Arable land',
        'Rice fields': 'Arable land',
        'Vineyards': 'Permanent crops',
        'Fruit trees and berry plantations': 'Permanent crops',
        'Olive groves': 'Permanent crops',
        'Annual crops associated with permanent crops': 'Permanent crops',
        'Natural grassland': 'Natural grassland and sparsely vegetated areas',
        'Sparsely vegetated areas': 'Natural grassland and sparsely vegetated areas',
        'Moors and heathland': 'Moors, heathland and sclerophyllous vegetation',
        'Sclerophyllous vegetation': 'Moors, heathland and sclerophyllous vegetation',
        'Inland marshes': 'Inland wetlands',
        'Peatbogs': 'Inland wetlands',
        'Salt marshes': 'Coastal wetlands',
        'Salines': 'Coastal wetlands',
        'Water bodies': 'Inland waters',
        'Water courses': 'Inland waters',
        'Coastal lagoons': 'Marine waters',
        'Estuaries': 'Marine waters',
        'Sea and ocean': 'Marine waters'
    }
    def __init__(self, **kwargs):
        self.root_path = kwargs['root_path']

        if kwargs['img_list_path'] is None:
            self.img_list = kwargs['img_list']
        else:
            img_list_path = kwargs['img_list_path']
            with open(img_list_path) as fp:
                self.img_list = fp.read().splitlines()
        
        samples = [
            (os.path.join(self.root_path, img_f_name), self.filename_to_class_multi_hot(img_f_name))
            for img_f_name in self.img_list
        ]
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.nb_classes = 19
        EvalDataset.__init__(self, **kwargs)
        self.file_reader = functools.partial(merge_rasterio, bands=self.BAND_NAMES, ref_band=self.REF_PROFILE_BAND, out_size=(120,120))

    def filename_to_class_multi_hot(self, img_f_name):
        with open(os.path.join(self.root_path, img_f_name, f'{img_f_name}_labels_metadata.json'), 'r') as f:
            labels = json.load(f)['labels']
        target = np.zeros((len(self.NEW_LABELS),), dtype=np.float32)
        for label in labels:
            if label in self.GROUP_LABELS:
                target[self.NEW_LABELS.index(self.GROUP_LABELS[label])] = 1
            elif label not in set(self.NEW_LABELS):
                continue
            else:
                target[self.NEW_LABELS.index(label)] = 1
        return target

    def apply_transforms(self, img):
        if self.is_train:
            if self.eval_type == 'ft':
                img = aug_img(img)
            scale = (self.input_size / img.shape[1], self.input_size/ img.shape[2], 1)
            img = rescale(img.transpose(1,2,0), scale, order=3, anti_aliasing=True).transpose(2,0,1)
        else:
            scale = (self.input_size / img.shape[1], self.input_size/ img.shape[2], 1)
            img = rescale(img.transpose(1,2,0), scale, order=3, anti_aliasing=True).transpose(2,0,1)
        return img

class BigEarthNetMM(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        img_pair_list_path = kwargs['img_pair_list_path']

        with open(img_pair_list_path) as fp:
            img_pair_list = [line.strip().split() for line in fp]
        s2_img_list = [img_f_name for img_f_name, _ in img_pair_list]
        s1_img_list = [img_f_name for _, img_f_name in img_pair_list]

        self.s2_ds = BigEarthNetS2(
            sensor_specs=kwargs['sensor_specs'][0],
            root_path=kwargs['root_path'][0], 
            img_list_path=None,
            img_list=s2_img_list,
            input_size=kwargs['input_size'],
            spectrum_specs=kwargs['spectrum_specs'],
            patch_size=kwargs['patch_size'],
            scale=kwargs['scale'],
            is_train=kwargs['is_train'],
            eval_type=kwargs['eval_type'],
            is_patchify=kwargs['is_patchify']
            )
        self.s1_ds = BigEarthNetS1(
            sensor_specs=kwargs['sensor_specs'][1],
            root_path=kwargs['root_path'][1],
            img_list_path=None,
            img_list=s1_img_list,
            input_size=kwargs['input_size'],
            spectrum_specs=kwargs['spectrum_specs'],
            patch_size=kwargs['patch_size'],
            scale=kwargs['scale'],
            is_train=kwargs['is_train'],
            eval_type=kwargs['eval_type'],
            is_patchify=kwargs['is_patchify']
            )

        self.is_train = self.s2_ds.is_train
        self.eval_type = self.s2_ds.eval_type
        self.input_size = self.s2_ds.input_size
        self.is_patchify = self.s2_ds.is_patchify
        self.patch_size = self.s2_ds.patch_size
        self.nb_patch_length = self.s2_ds.nb_patch_length

    def mixup_pairs(self, x1, x2, proj_indices1, proj_indices2, mixup_ratio=0.5):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        """
        PH, PW, C, H, W = x1.shape  # batch, length, dim
        L = PH*PW
        x1 = x1.reshape(L, C, H, W)
        x2 = x2.reshape(L, C, H, W)
        proj_indices1 = proj_indices1.reshape(L,-1)
        proj_indices2 = proj_indices2.reshape(L,-1)

        len_x1 = int(L * (1 - mixup_ratio))
        noise = np.random.rand(L)
        
        # sort noise for each sample
        ids_shuffle = np.argsort(noise) # ascend: small is keep, large is remove

        # keep the first subset
        ids_keep_x1 = ids_shuffle[:len_x1]
        ids_keep_x2 = ids_shuffle[len_x1:]

        ids_keep_x1_full = np.tile(np.expand_dims(ids_keep_x1, axis=(-1, -2, -3)), (1,C,H,W))
        ids_keep_x2_full = np.tile(np.expand_dims(ids_keep_x2, axis=(-1, -2, -3)), (1,C,H,W))
        
        x1_mixed = np.zeros_like(x1)
        np.put_along_axis(x1_mixed, ids_keep_x1_full, np.take_along_axis(x1, ids_keep_x1_full, axis=0), axis=0)
        np.put_along_axis(x1_mixed, ids_keep_x2_full, np.take_along_axis(x2, ids_keep_x2_full, axis=0), axis=0)

        x2_mixed = np.zeros_like(x2)
        np.put_along_axis(x2_mixed, ids_keep_x2_full, np.take_along_axis(x1, ids_keep_x2_full, axis=0), axis=0)
        np.put_along_axis(x2_mixed, ids_keep_x1_full, np.take_along_axis(x2, ids_keep_x1_full, axis=0), axis=0)

        ids_keep_x1_full = np.tile(np.expand_dims(ids_keep_x1, axis=-1), (1,proj_indices1.shape[-1]))
        ids_keep_x2_full = np.tile(np.expand_dims(ids_keep_x2, axis=-1), (1,proj_indices2.shape[-1]))

        proj_indices1_mixed = np.zeros_like(proj_indices1)
        np.put_along_axis(proj_indices1_mixed, ids_keep_x1_full, np.take_along_axis(proj_indices1, ids_keep_x1_full, axis=0), axis=0)
        np.put_along_axis(proj_indices1_mixed, ids_keep_x2_full, np.take_along_axis(proj_indices2, ids_keep_x2_full, axis=0), axis=0)

        proj_indices2_mixed = np.zeros_like(proj_indices2)
        np.put_along_axis(proj_indices2_mixed, ids_keep_x2_full, np.take_along_axis(proj_indices1, ids_keep_x2_full, axis=0), axis=0)
        np.put_along_axis(proj_indices2_mixed, ids_keep_x1_full, np.take_along_axis(proj_indices2, ids_keep_x1_full, axis=0), axis=0)
        
        return x1_mixed.reshape(PH, PW, C, H, W), x2_mixed.reshape(PH, PW, C, H, W), proj_indices1_mixed.reshape(PH, PW, -1), proj_indices2_mixed.reshape(PH, PW, -1)
    
    def __getitem__(self, index: int):
        s2path, target = self.s2_ds.samples[index]
        s1path, _ = self.s1_ds.samples[index]
        
        if type(target) is np.ndarray:
            target = torch.as_tensor(target.reshape(1,-1))

        with self.s2_ds.file_reader(s2path, 'r') as s2image_ds:
            with georef_rasterio_ds(s2image_ds, self.s2_ds.resolution, self.s2_ds.apply_georef) as s2image_ds_georef:
                with resample_rasterio_ds(s2image_ds_georef, self.s2_ds.scale) as s2image_ds_resampled:
                    s2img = s2image_ds_resampled.read().astype(np.float32)
        s2img, _ = self.s2_ds.normalize_standardize(s2img)

        with self.s1_ds.file_reader(s1path, 'r') as s1image_ds:
            with georef_rasterio_ds(s1image_ds, self.s1_ds.resolution, self.s1_ds.apply_georef) as s1image_ds_georef:
                with resample_rasterio_ds(s1image_ds_georef, self.s1_ds.scale) as s1image_ds_resampled:
                    s1img = s1image_ds_resampled.read().astype(np.float32)
        s1img, _ = self.s1_ds.normalize_standardize(s1img)
        
        s1img, s2img = self.apply_transforms(s1img, s2img)

        if self.is_patchify:
            s1img = patchify(s1img, self.patch_size)
            s2img = patchify(s2img, self.patch_size)

        first_nb_band = s1img.shape[2]
        second_nb_band = s2img.shape[2]

        s1img = np.tile(s1img, (1,1,int(12/first_nb_band),1,1))
        s2img = np.tile(s2img, (1,1,int(12/second_nb_band),1,1))
        
        s1proj_indices = np.tile(self.s1_ds.projection_indices.reshape(
            1,1,-1), (self.nb_patch_length, self.nb_patch_length, int(12/first_nb_band))).astype(np.int32)
        
        s2proj_indices = np.tile(self.s2_ds.projection_indices.reshape(
            1,1,-1), (self.nb_patch_length, self.nb_patch_length, int(12/second_nb_band))).astype(np.int32)

        img1, img2, projection_indices1, projection_indices2 = self.mixup_pairs(s1img, s2img, s1proj_indices, s2proj_indices, mixup_ratio=0.5)
        
        return (
            torch.as_tensor(np.expand_dims(img1,0).astype(np.float32)), 
            torch.as_tensor(np.expand_dims(projection_indices1,0))
        ), (
            torch.as_tensor(np.expand_dims(img2,0).astype(np.float32)), 
            torch.as_tensor(np.expand_dims(projection_indices2,0))
        ), target
    
    def __len__(self):
        return len(self.s2_ds.samples)
    
    def apply_transforms(self, img1, img2):
        if (self.is_train) and (self.eval_type == 'ft'):
            img1, img2 = aug_img_pair(img1, img2)
        scale = (self.input_size / img1.shape[1], self.input_size/ img1.shape[2], 1)
        img1 = rescale(img1.transpose(1,2,0), scale, order=3, anti_aliasing=True).transpose(2,0,1)

        scale = (self.input_size / img2.shape[1], self.input_size/ img2.shape[2], 1)
        img2 = rescale(img2.transpose(1,2,0), scale, order=3, anti_aliasing=True).transpose(2,0,1)

        return img1, img2
    
class BigEarthNetS2(BigEarthNet):
    BAND_NAMES = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    REF_PROFILE_BAND = 'B04'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
class BigEarthNetS1(BigEarthNet):
    BAND_NAMES = ['VV', 'VH']
    REF_PROFILE_BAND = 'VV'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

DS_CLASS_MAPPING = {
    'EuroSAT': EuroSAT_MS,
    'WHU-RS19': WHU_RS19,
    'BigEarthNetS2': BigEarthNetS2,
    'BigEarthNetS1': BigEarthNetS1,
    'UC-Merced': UC_Merced,
    'RESISC45': RESISC45,
    'BigEarthNetMM': BigEarthNetMM,
    'DFC2020': DFC2020
}

def build_eval_loader(args, is_patchify=True):
    if args.multi_modal:
        dataset_eval_train = build_paired_eval_dataset(
            args, args.eval_specs[args.eval_dataset]['train_img_pair_list'], args.eval_type, is_patchify=is_patchify, is_train=True
        )
        dataset_eval_test = build_paired_eval_dataset(
            args, args.eval_specs[args.eval_dataset]['val_img_pair_list'], args.eval_type, is_patchify=is_patchify
        )
    else:
        dataset_eval_train = build_eval_dataset(
            args, args.eval_specs[args.eval_dataset]['train_img_list'], args.eval_type, is_patchify=is_patchify, is_train=True
        )
        dataset_eval_test = build_eval_dataset(
            args, args.eval_specs[args.eval_dataset]['val_img_list'], args.eval_type, is_patchify=is_patchify
        )
    data_loader_eval_train = torch.utils.data.DataLoader(
        dataset_eval_train,
        batch_size=args.eval_batch_size,
        num_workers=args.nb_workers_per_gpu,
        pin_memory=args.pin_mem,
        drop_last=True if args.num_gpus > 1 else False,
        collate_fn=EvalCollateFn() if not args.multi_modal else EvalPairCollateFn(),
        shuffle=False if args.eval_type =='kNN' else True,
        batch_sampler=None,
        prefetch_factor=args.prefetch_factor if args.nb_workers_per_gpu > 0 else None,
        persistent_workers=True if args.nb_workers_per_gpu > 0 else False
    )
    
    data_loader_eval_test = torch.utils.data.DataLoader(
        dataset_eval_test,
        batch_size=args.eval_batch_size,
        num_workers=args.nb_workers_per_gpu,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=EvalCollateFn() if not args.multi_modal else EvalPairCollateFn(),
        shuffle=False,
        batch_sampler=None,
        prefetch_factor=args.prefetch_factor if args.nb_workers_per_gpu > 0 else None,
        persistent_workers=True if args.nb_workers_per_gpu > 0 else False
    )
    return data_loader_eval_train, data_loader_eval_test

def build_eval_dataset(args, img_list_path, eval_type, is_patchify=True, is_train=False):    
    return DS_CLASS_MAPPING[args.eval_dataset](
        input_size = args.input_size,
        sensor_specs = args.eval_specs[args.eval_dataset]['sensor_specs'],
        spectrum_specs = args.spectrum_specs,
        patch_size=args.patch_size,
        root_path = args.eval_specs[args.eval_dataset]['root_path'],
        scale = args.eval_scale,
        img_list_path = img_list_path,
        is_train=is_train,
        is_patchify=is_patchify,
        eval_type=eval_type
    )

def build_paired_eval_dataset(args, img_pair_list_path, eval_type, is_patchify=True, is_train=False):    
    sources = args.eval_specs[args.eval_dataset]['sources']
    return DS_CLASS_MAPPING[args.eval_dataset](
        input_size = args.input_size,
        sensor_specs = [args.eval_specs[i]['sensor_specs'] for i in sources],
        spectrum_specs = args.spectrum_specs,
        patch_size=args.patch_size,
        root_path = [args.eval_specs[i]['root_path'] for i in sources],
        scale = args.eval_scale,
        img_pair_list_path = img_pair_list_path,
        is_train=is_train,
        is_patchify=is_patchify,
        eval_type=eval_type
    )