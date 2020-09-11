import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms

import os, random, cv2, math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from config import cfg

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img


class Market_Loader(Data.Dataset):
    def __init__(self, root, size, specific=None, type='train'):
        assert type in ['train', 'query', 'gallery', 'test'], 'Unkown phase [Opt: \'train\', \'query\', \'gallery\', \'test\'].'

        images = os.listdir(root)
        classes = list(set([self.get_name(im) for im in images ]))
        if specific is not None:
            self.data = [im for im in images if im in specific]
        else:
            self.data = images

        self.root = root
        self.type = type
        self.size = size
        self.images = images
        self.classes = classes

    def __getitem__(self, index):
        img_name = self.data[index]
        label = self.classes.index(self.get_name(img_name))

        img = Image.open(os.path.join(self.root, img_name)).convert('RGB')

        transformer = self.get_transform()
        img = transformer(img)

        return img, label, img_name

    def get_transform(self, ):
        if self.type == 'train':
            transformer = transforms.Compose([
                transforms.Resize(self.size, interpolation=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                RandomErasing(probability=0.4, mean=[0.0, 0.0, 0.0])
                ])
        else:
            transformer = transforms.Compose([
                transforms.Resize(self.size, interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

        return transformer

    @staticmethod
    def __flip__(img, flip):
        if flip:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    @staticmethod
    def get_name(image):
        return int(image.split('_')[0])

    def __len__(self):
        return len(self.data)


class Market_Triplet_Loader(Data.Dataset):
    def __init__(self, root, size, k, repeat, specific=None, type='train'):
        assert type in ['train', 'query', 'gallery', 'test'], 'Unkown phase [Opt: \'train\', \'query\', \'gallery\', \'test\'].'

        images = os.listdir(root)
        classes = sorted(list(set([self.get_name(im) for im in images ])))
        assert len(classes) == cfg.NUM_CLASS
        if specific is not None:
            self.images = [im for im in images if im in specific]
        else:
            self.images = images
        pool = {i:[] for i in range(len(classes))}
        for im in self.images:
            key = classes.index(self.get_name(im))
            pool[key].append(im)

        self.k = k
        self.repeat = repeat
        self.root = root
        self.type = type
        self.size = size
        self.classes = classes
        self.pool = pool
        self.preprocess()

    def __getitem__(self, index):
        img_path = self.data[index][0]
        label = self.data[index][1]

        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')

        transformer = self.get_transform()
        img = transformer(img)

        return img, label, img_path

    def get_transform(self, ):
        if self.type == 'train':
            transformer = transforms.Compose([
                transforms.Resize(self.size, interpolation=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0])
                ])
        else:
            transformer = transforms.Compose([
                transforms.Resize(self.size, interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

        return transformer

    def preprocess(self,):
        self.data = []
        keys = list(self.pool.keys())
        for _ in range(self.repeat):
            count = 0
            random.shuffle(keys)
            for key in keys:
                imgs = self.pool[key]
                if len(imgs) < self.k:
                    count += 1
                    imgs = list(np.random.choice(imgs, self.k, replace=True))
                else:
                    imgs = list(np.random.choice(imgs, self.k, replace=False))
                for im in imgs:
                    self.data.append((im, key))
        print ('*************************************************************************************************')
        print ("Warning: There are %d people having less than k=%d images, the value of 'k' may be reconsidered!" % (count, self.k))
        print ('*************************************************************************************************')

    @staticmethod
    def __flip__(img, flip):
        if flip:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    @staticmethod
    def get_name(image):
        return int(image.split('_')[0])

    @staticmethod
    def get_camera(image, type):
        return int(image.split('_')[1][1])

    def __len__(self):
        return len(self.data)