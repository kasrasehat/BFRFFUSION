import random
import cv2
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import torch
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor


class CustomPairedImageDataset(data.Dataset):
    def __init__(self, opt):
        super(CustomPairedImageDataset, self).__init__()
        self.opt = opt
        # File client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder],
                ['lq', 'gt'],
                self.opt['meta_info_file'],
                self.filename_tmpl
            )
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder],
                ['lq', 'gt'],
                self.filename_tmpl
            )

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load gt and lq images
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # === Begin Modification ===
        # Get the original dimensions
        h, w, c = img_gt.shape

        # Randomly select a scale factor
        scale_factors = [0.75, 0.85, 0.95]
        scale_factor = random.choice(scale_factors)

        # Calculate new crop size
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)

        # Randomly select top-left corner for the crop
        if h == new_h:
            top = 0
        else:
            top = random.randint(0, h - new_h)
        if w == new_w:
            left = 0
        else:
            left = random.randint(0, w - new_w)

        # Crop the images
        img_gt = img_gt[top:top + new_h, left:left + new_w, :]
        img_lq = img_lq[top:top + new_h, left:left + new_w, :]

        # Resize images to 512x512
        img_gt = cv2.resize(img_gt, (512, 512), interpolation=cv2.INTER_LINEAR)
        img_lq = cv2.resize(img_lq, (512, 512), interpolation=cv2.INTER_LINEAR)
        # === End Modification ===

        # Data augmentation (optional)
        if self.opt['phase'] == 'train':
            # Flip and rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt.get('use_hflip', False), self.opt.get('use_rot', False))

        # Color space transformation (optional)
        # if self.opt.get('color', None) == 'y':
        #     img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
        #     img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # Convert images to tensors
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        
        # round and clip
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.
        
        # Normalize
        if self.mean is not None or self.std is not None:
            # normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'txt':''}

    def __len__(self):
        return len(self.paths)
