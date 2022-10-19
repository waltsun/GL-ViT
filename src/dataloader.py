import os
import random

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from time import perf_counter

pre_transform = transforms.Resize(256, interpolation=3)

transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

class Dataloader:
    def __init__(self, way_num: int, support_num: int, query_num: int, dataset_path: str, phase: str):
        self.way_num = way_num
        self.support_num = support_num
        self.query_num = query_num
        self.sample_num = self.support_num + self.query_num
        self.dataset_path = dataset_path
        self.phase = phase
        self.phase_path = os.path.join(self.dataset_path, self.phase)

        assert phase in ["train", "val", "test"]

        self.label_list = os.listdir(self.phase_path)
        self.tot_classes = len(self.label_list)
        self.image_list = []

        self.images_map = []
        for label in self.label_list:
            label_path = os.path.join(self.phase_path, label)
            self.image_list.append(os.listdir(label_path))
            self.images_map.append({})

    def get_episode_batch(self):
        sampled_labels = random.sample(range(self.tot_classes), self.way_num)
        support_set = []
        query_set = []
        for i, label_idx in enumerate(sampled_labels):
            image_list = self.image_list[label_idx]
            label_path = os.path.join(self.phase_path, self.label_list[label_idx])
            sampled_images_names = random.sample(image_list, self.sample_num)
            sampled_images = []
            for name in sampled_images_names:
                if name in self.images_map[label_idx]:
                    sampled_images.append(self.images_map[label_idx][name])
                else:
                    image_path = os.path.join(label_path, name)
                    fp = open(image_path,'rb')
                    image = Image.open(fp)
                    image = pre_transform(image)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image = transform(image)
                    fp.close()
                    self.images_map[label_idx][name] = image
                    sampled_images.append(image)

            sampled_support = sampled_images[: self.support_num]
            sampled_support = torch.stack(sampled_support, dim = 0)
            support_set.append(sampled_support)

            sampled_query = sampled_images[self.support_num: ]
            query_set.extend([(image, label_idx, i) for image in sampled_query])

        support_images = torch.cat(support_set, dim = 0)
        support_labels = torch.LongTensor(sampled_labels)

        random.shuffle(query_set)
        query_images, query_labels, query_episode_labels = zip(*query_set)
        query_images = torch.stack(query_images, dim = 0)
        query_labels = torch.LongTensor(query_labels)
        query_episode_labels = torch.LongTensor(query_episode_labels)

        return support_images, support_labels, query_images, query_labels, query_episode_labels
