import os

from PIL import Image

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset


class SingleDataset(BaseDataset):
    """Load one domain for H&E to IHC inference."""

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        phase_dir = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_A = phase_dir if os.path.isdir(phase_dir) else opt.dataroot
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        path = self.A_paths[index]
        image = Image.open(path).convert('RGB')
        tensor = self.transform(image)
        return {'A': tensor, 'B': tensor, 'A_paths': path, 'B_paths': path}

    def __len__(self):
        return len(self.A_paths)
