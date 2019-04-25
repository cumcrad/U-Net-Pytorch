from torchvision.datasets import ImageFolder
import torch
import numpy as np
from PIL import Image
from pathlib import Path

IMG_EXTENSIONS = ['.png', '.nii', '.nii.gz', '.npy']


def npy_loader(path, img_size, interpolation):
    return Image.fromarray(np.load(path)).resize([img_size, img_size], interpolation)


class NpyFolder(ImageFolder):
    def __init__(self, root, context, img_size, interp, transform=None, target_transform=None,
                 loader=npy_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.context = context
        self.npys = self.samples
        self.img_size = img_size
        self.interp = interp

    def get_path(self, index):
        path, _ = self.samples[index]
        return path

    def get_slice_num(self, index):
        return int(self.get_path(index).split('-')[-1].split('_')[-1].split('.')[0][len('slice'):])

    def get_sample_by_id(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path, self.img_size, self.interp)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_filename(self, index):
        path, _ = self.samples[index]
        return Path(path).name

    def __getitem__(self, index):
        slice_num = self.get_slice_num(index)
        main_slice = self.get_sample_by_id(index)
        inputs_b = main_slice

        # load slices before middle slice
        inputs_a = []
        for i in range(index - self.context, index):
            if i < 0 or self.get_slice_num(index) >= slice_num:
                inputs = inputs_b
            else:
                inputs = self.get_sample_by_id(i)
            inputs_a.append(inputs)

        # load slices after middle slice
        inputs_c = []
        for i in range(index + 1, index + self.context + 1):
            if i >= len(self) or self.get_slice_num(i) <= slice_num:
                inputs = inputs_b
            else:
                inputs = self.get_sample_by_id(i)
            inputs_c.append(inputs)

        # concatenate all slices for context
        inputs = inputs_a + [inputs_b] + inputs_c
        inputs = torch.cat(inputs, 0)
        return inputs, self.get_filename(index)


class NumpyFolder:
    def __init__(self, main_root, label_root, context, img_size, transform=None):
        self.main_folder = NpyFolder(main_root, context, img_size, Image.BICUBIC, transform=transform)
        self.label_folder = NpyFolder(label_root, context, img_size, Image.NEAREST, transform=transform)

    def __getitem__(self, index):
        main_img, _ = self.main_folder.__getitem__(index)
        label_img, _ = self.label_folder.__getitem__(index)
        return main_img, label_img, _

    def __len__(self):
        assert len(self.main_folder) == len(self.label_folder)
        return len(self.main_folder)

