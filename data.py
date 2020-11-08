import numpy as np
import torch
import nibabel as nib

from torchio.data.image import Image
import torchio

import h5py as h5
import numpy as np

import torch
from torch.utils.data import Dataset
from torchio.data.subject import Subject

class H5DSImage(Image):
    def __init__(self, h5DS=None, lazypatch=True, imtype=torchio.INTENSITY, **kwargs):
        kwargs['path'] = ''
        kwargs['type'] = imtype
        super().__init__(**kwargs)
        self.h5DS = h5DS
        self.lazypatch = lazypatch
        if not self.lazypatch:
            self.load()

    def load(self) -> None:
        if self._loaded:
            return
        if self.lazypatch:
            tensor, affine = self.h5DS, np.eye(4)
        else:
            tensor, affine = self.read_and_check_h5(self.h5DS)
        self[torchio.DATA] = tensor
        self[torchio.AFFINE] = affine
        self._loaded = True

    @property
    def spatial_shape(self):
        if self.lazypatch:
            return self.shape
        else:
            return self.shape[1:]

    def crop(self, index_ini, index_fin):
        new_origin = nib.affines.apply_affine(self.affine, index_ini)
        new_affine = self.affine.copy()
        new_affine[:3, 3] = new_origin
        i0, j0, k0 = index_ini
        i1, j1, k1 = index_fin
        if len(self.data.shape) == 4:
            patch = self.data[:, i0:i1, j0:j1, k0:k1]
        else:
            patch = np.expand_dims(self.data[i0:i1, j0:j1, k0:k1], 0)
        if not isinstance(self.data, torch.Tensor):
            patch = torch.from_numpy(patch)
        kwargs = dict(
            tensor=patch,
            affine=new_affine,
            type=self.type,
            path=self.path,
            h5DS=self.h5DS
        )
        for key, value in self.items():
            if key in torchio.data.image.PROTECTED_KEYS: continue
            kwargs[key] = value  
        return self.__class__(**kwargs)

    def read_and_check_h5(self, h5DS):
        tensor, affine = torch.from_numpy(h5DS[()]).unsqueeze(0), np.eye(4)
        tensor = super().parse_tensor_shape(tensor)
        if self.channels_last:
            tensor = tensor.permute(3, 0, 1, 2)
        if self.check_nans and torch.isnan(tensor).any():
            warnings.warn(f'NaNs found in file "{path}"')
        return tensor, affine


class MoodTrainSet(Dataset):
    def __init__(self, indices=None, region='brain', data_path='MOOD_train.h5', torchiosub=True, lazypatch=True, preload=False):
        self.h5 = h5.File(data_path, 'r', swmr=True)
        self.samples = []
        if indices:
            self.samples = [self.h5[region][str(i).zfill(5)]for i in indices]
            # self.samples2 = [self.h5[region][str(i).zfill(5)][:] for i in indices]
        else:
            self.samples = [self.h5[region][i] for i in list(self.h5[region])]
        if preload:
            print('Preloading MoodTrainSet')
            for i in range(len(self.samples)):
                self.samples[i] = self.samples[i][:]
        self.torchiosub = torchiosub
        self.lazypatch = lazypatch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        if self.torchiosub:
            return Subject({'img':H5DSImage(self.samples[item], lazypatch=self.lazypatch)})
        else:
            return torch.from_numpy(self.samples[item][()]).unsqueeze(0)

class MoodValSet(Dataset):
    def __init__(self, load_abnormal=True, load_normal=True, loadASTrain=False, data_path='MOOD_val.h5', torchiosub=True, lazypatch=True, preload=False):
        self.h5 = h5.File(data_path, 'r', swmr=True)
        self.samples = []
        if load_abnormal:
            self.samples+=[(self.h5['abnormal'][i], self.h5['abnormal_mask'][i]) for i in list(self.h5['abnormal'])]
        if load_normal:
            self.samples+=[self.h5['normal'][i] for i in list(self.h5['normal'])]
        if preload:
            print('Preloading MoodValSet')
            for i in range(len(self.samples)):
                if len(self.samples[i]) == 2:
                    self.samples[i] = (self.samples[i][0][:], self.samples[i][1][:])
                else:
                    self.samples[i] = self.samples[i][:]
        self.loadASTrain = loadASTrain
        self.torchiosub = torchiosub
        self.lazypatch = lazypatch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        if self.loadASTrain:
            if self.torchiosub:
                return Subject({'img':H5DSImage(self.samples[item][0], lazypatch=self.lazypatch)})
            else:
                return torch.from_numpy(self.samples[item][0][()]).unsqueeze(0)
        else:
            if self.torchiosub:
                if len(self.samples[item]) == 2:
                    return Subject({'img':H5DSImage(self.samples[item][0], lazypatch=self.lazypatch),
                                    'gt':H5DSImage(self.samples[item][1], lazypatch=self.lazypatch)})
                else:
                    return Subject({'img':H5DSImage(self.samples[item], lazypatch=self.lazypatch),
                                    'gt':H5DSImage(self.samples[item], lazypatch=self.lazypatch)}) #this is dirty. TODO

            else:
                return (torch.from_numpy(self.samples[item][0][()]).unsqueeze(0), torch.from_numpy(self.samples[item][1][()]).unsqueeze(0))