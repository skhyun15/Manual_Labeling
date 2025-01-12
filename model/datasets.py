import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
from collections import namedtuple

import natsort


class BoneSegmentDataset(Dataset):

    def __init__(self, root, augment=False, num_classes=17, resize=512):

        self.root = os.path.expanduser(root)
        self.images_dir = os.path.join(self.root, "SPECT")
        self.targets_dir = os.path.join(self.root, "target")
        self.augment = augment
        self.resize = resize

        self.images = []
        self.targets = []

        # Ensure that this matches the above mapping!#!@#!@#
        # For example 4 classes, means we should map to the ids=(0,1,2,3)
        # This is used to specify how many outputs the network should product...
        self.num_classes = num_classes

        # self.mappingrgb = {
        #     0: [255, 255, 255],  # 검정색
        #     1: [222, 148, 80],  # 회색
        #     2: [147, 71, 238],  # 초록색
        #     3: [187, 19, 208],  # 어두운 회색
        #     4: [98, 43, 249],  # 노랑색
        #     5: [166, 58, 136],  # 자홍색
        #     6: [202, 45, 114],  # 청록색
        #     7: [103, 209, 30],  # 어두운 빨강
        #     8: [235, 57, 90],  # 어두운 초록
        #     9: [18, 14, 75],  # 어두운 파랑
        #     10: [209, 156, 101],  # 올리브색
        #     11: [230, 13, 166],  # 보라색
        #     12: [200, 150, 134],  # 어두운 청록
        #     13: [242, 6, 88],  # 밝은 회색
        #     14: [250, 186, 207],  # 빨강색
        #     15: [144, 173, 28],  # 파랑색
        #     16: [232, 28, 225],  # 흰색
        # }

        self.mappingrgb = {
            0: [0, 0, 0],  # 클래스 0: 검정색
            1: [128, 0, 0],  # 클래스 1: 어두운 빨강
            2: [0, 128, 0],  # 클래스 2: 어두운 초록
            3: [128, 128, 0],  # 클래스 3: 어두운 노랑
            4: [0, 0, 128],  # 클래스 4: 어두운 파랑
            5: [128, 0, 128],  # 클래스 5: 어두운 자주
            6: [0, 128, 128],  # 클래스 6: 어두운 청록
            7: [128, 128, 128],  # 클래스 7: 회색
            8: [64, 0, 0],  # 클래스 8: 진한 빨강
            9: [192, 0, 0],  # 클래스 9: 밝은 빨강
            10: [64, 128, 0],  # 클래스 10: 올리브
            11: [192, 128, 0],  # 클래스 11: 황토색
            12: [64, 0, 128],  # 클래스 12: 보라색
            13: [192, 0, 128],  # 클래스 13: 분홍색
            14: [64, 128, 128],  # 클래스 14: 연한 청록
            15: [192, 128, 128],  # 클래스 15: 연한 분홍
            16: [255, 255, 255],  # 클래스 16: 흰색
        }

        # =============================================
        # Check that inputs are valid
        # =============================================
        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError(
                "Dataset not found or incomplete. Please make sure all required folders for the"
                ' specified "SPECT" and "target" are inside the "root" directory'
            )

        # =============================================
        # Read in the paths to all images
        # =============================================

        images_dir_list = natsort.natsorted(os.listdir(self.images_dir))

        for path in images_dir_list:
            self.images.append(os.path.join(self.images_dir, path))

        targets_dir_list = natsort.natsorted(os.listdir(self.targets_dir))

        for path in targets_dir_list:
            self.targets.append(os.path.join(self.targets_dir, path))

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of images: {}\n".format(self.__len__())
        fmt_str += "    Augment: {}\n".format(self.augment)
        fmt_str += "    Root Location: {}\n".format(self.root)
        return fmt_str

    def __len__(self):
        return len(self.images)

    def _mask_to_class(self, mask):
        """
        Given the cityscapes dataset, this maps to a 0..classes numbers.
        This is because we are using a subset of all masks, so we have this "mapping" function.
        This mapping function is used to map all the standard ids into the smaller subset.
        """
        maskimg = torch.zeros((mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in self.mapping:
            maskimg[mask == k] = self.mapping[k]
        return maskimg

    def _mask_to_rgb(self, mask):
        """
        Given the Cityscapes mask file, this converts the ids into rgb colors.
        This is needed as we are interested in a sub-set of labels, thus can't just use the
        standard color output provided by the dataset.
        """
        rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in self.mappingrgb:
            rgbimg[0][mask == k] = self.mappingrgb[k][0]
            rgbimg[1][mask == k] = self.mappingrgb[k][1]
            rgbimg[2][mask == k] = self.mappingrgb[k][2]
        return rgbimg

    def class_to_rgb(self, mask):
        """
        This function maps the classification index ids into the rgb.
        For example after the argmax from the network, you want to find what class
        a given pixel belongs too. This does that but just changes the color
        so that we can compare it directly to the rgb groundtruth label.
        """

        rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in self.mappingrgb:
            rgbimg[0][mask == k] = self.mappingrgb[k][0]
            rgbimg[1][mask == k] = self.mappingrgb[k][1]
            rgbimg[2][mask == k] = self.mappingrgb[k][2]
        return rgbimg

    def __getitem__(self, index):

        # first load the RGB image
        image_npy = np.load(self.images[index])
        image_tensor = torch.from_numpy(image_npy).float()
        image_tensor = image_tensor.unsqueeze(0)

        # normalization
        image_tensor /= torch.max(image_tensor)
        image_tensor = torch.tanh(image_tensor)

        assert image_tensor.size()[1] == 512, f"got size of {image_tensor.size()}"
        # next load the target
        target = Image.open(self.targets[index]).convert("L")
        assert target.size[1] == 512, f"got size of {target.size}"
        # If augmenting, apply random transforms
        # Else we should just resize the image down to the correct size
        if self.augment:
            # Resize
            aug_size = int(self.resize + 20 * self.resize / 512)
            image_tensor = TF.resize(
                image_tensor,
                size=(aug_size, aug_size),
                interpolation=Image.BILINEAR,
            )
            target = TF.resize(
                target,
                size=(aug_size, aug_size),
                interpolation=Image.NEAREST,
            )
            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                image_tensor, output_size=(self.resize, self.resize)
            )
            image_tensor = TF.crop(image_tensor, i, j, h, w)
            target = TF.crop(target, i, j, h, w)
            # Random horizontal flipping
            # if random.random() > 0.5:
            #     image = TF.hflip(image)
            #     target = TF.hflip(target)
            # Random vertical flipping
            # (I found this caused issues with the sky=road during prediction)
            # if random.random() > 0.5:
            #    image = TF.vflip(image)
            #    target = TF.vflip(target)
        else:
            # Resize
            image_tensor = TF.resize(
                image_tensor,
                size=(self.resize, self.resize),
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
            target = TF.resize(
                target, size=(self.resize, self.resize), interpolation=Image.NEAREST
            )

        # convert to pytorch tensors
        # target = TF.to_tensor(target)
        target = torch.from_numpy(np.array(target, dtype=np.uint8))
        # image = TF.to_tensor(image)

        # convert the labels into a mask
        # targetrgb = self.mask_to_rgb(target)
        # targetmask = self.mask_to_class(target)

        # targetmask = targetmask.long()

        # targetmask = targetmask.to(torch.int).to(torch.long)에 관한 try 에러문 만들기
        # targetrgb = targetrgb.long()
        # targetrgb = targetrgb.to(torch.int).to(torch.long)
        # finally return the image pair
        assert image_tensor.size() == (
            1,
            self.resize,
            self.resize,
        ), "size is not correct"
        assert target.size() == (self.resize, self.resize), "size is not correct"
        return image_tensor, target


# test 코드

if __name__ == "__main__":
    # Create the dataset
    dataset = BoneSegmentDataset(
        root="/projects3/pi/nhcho/Sev_WBBS/skhyun/segmentation/data/ANT",
        resize=256,
        augment=True,
    )

    # Create a loader to traverse the dataset
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Get a batch of data
    for i, data in enumerate(loader):
        images, targets = data
        print("Images: ", images.size())
        print("Targets: ", targets.size())
        break
