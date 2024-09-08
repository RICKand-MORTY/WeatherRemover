"""
Reference from Transweather,made some modifications.
"""
import random
import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import re
from PIL import ImageFile
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir,train_filename, random_flip, random_rotate):
        super().__init__()
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        train_list = train_data_dir + train_filename
        with open(train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [i.strip().replace('input','gt') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]

        img_id = re.split('/',input_name)[-1][:-4]

        input_img = Image.open(self.train_data_dir + input_name).convert('RGB')
        gt_img = Image.open(self.train_data_dir + gt_name).convert('RGB')

        width, height = input_img.size

        if width < crop_width and height < crop_height :
            input_img = input_img.resize((crop_width,crop_height), Image.LANCZOS)
            gt_img = gt_img.resize((crop_width, crop_height), Image.LANCZOS)
        elif width < crop_width :
            input_img = input_img.resize((crop_width,height), Image.LANCZOS)
            gt_img = gt_img.resize((crop_width,height), Image.LANCZOS)
        elif height < crop_height :
            input_img = input_img.resize((width,crop_height), Image.LANCZOS)
            gt_img = gt_img.resize((width, crop_height), Image.LANCZOS)

        width, height = input_img.size
        #random crop
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        input_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_crop_img)
        gt = transform_gt(gt_crop_img)

        # --- Check the channel is 3 or not --- #
        if list(input_im.shape)[0] != 3 or list(gt.shape)[0] != 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        if self.random_flip and random.random() < 0.5:
            input_im = torch.flip(input_im, dims=[-1])
            gt = torch.flip(gt, dims=[-1])

        if self.random_rotate:
            r = random.random()
            # 随机旋转
            if r < 0.25:
                # 90
                input_im = torch.rot90(input_im, k=1, dims=(1, 2))
                gt = torch.rot90(gt, k=1, dims=(1, 2))

            elif r < 0.5:
                # 270
                input_im = torch.rot90(input_im, k=3, dims=(1, 2))
                gt = torch.rot90(gt, k=3, dims=(1, 2))
            elif r < 0.75:
                # 180
                input_im = torch.rot90(input_im, k=2, dims=(1, 2))
                gt = torch.rot90(gt, k=2, dims=(1, 2))

        return input_im, gt, img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)

