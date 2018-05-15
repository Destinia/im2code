import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
import os
import pickle
import numpy as np
from PIL import Image
import random
from itertools import chain
from util.util import get_vocab, get_images, get_labels, get_image_size


class UI2codeDataset(data.Dataset):
    def __init__(self, opt, phase):
        self.opt = opt
        self.root = opt.data_root
        if phase == 'train':
            data_path = opt.train_data_path
        elif phase == 'val':
            data_path = opt.val_data_path
        elif phase == 'test':
            data_path = opt.test_data_path
        self.image_paths = get_images(data_path)
        self.ids = list(self.image_paths.keys())
        print('#image: ', len(self.image_paths))
        self.vocab = opt.vocab
        self.labels = get_labels(opt.label_path)
        # self.transform = transforms.Compose([transforms.ToTensor(),
        #     transforms.Normalize([0.2731853791024895], [0.24186649347904463])])
        self.transform = transforms.ToTensor()

    def load_data(self):
        return self
    
    def __getitem__(self, index):
        image_path = self.image_paths[self.ids[index]]
        label = self.labels[self.ids[index]]
        # image = Image.open(os.path.join(self.root, 'processedImage', image_path))
        # image = np.dot(np.asarray(image, dtype=np.float32), [0.299, 0.5870, 0.1140])
        # image = torch.from_numpy(image).unsqueeze(0).float()
        image = Image.open(os.path.join(self.root, 'processedImage', image_path)).convert('L')
        if self.transform is not None:
            image = self.transform(image)
        skeleton = [self.opt.bos]
        skeleton.extend([self.vocab[t] if t in self.vocab else self.vocab['<unk>'] for t in label])
        skeleton.append(self.opt.eos)
        target = torch.Tensor(skeleton)
        return image, target

    def get_vocab(self):
        return self.vocab

    def __len__(self):
        return len(self.image_paths)

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    num_non_zero = sum(lengths)
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap
    return images, targets, num_non_zero


class UI2codeDataloader():
    def __init__(self, opt, phase='train'):
        self.dataset = UI2codeDataset(opt, phase)
        self.batch_size = opt.batch_size
        self.shuffle = (not opt.serial_batches) and phase == 'train'
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                      batch_size=opt.batch_size,
                                                      shuffle=self.shuffle,
                                                      num_workers=opt.nThreads,
                                                      collate_fn=collate_fn)
    def load_data(self):
        return self

    def get_vocab(self):
        return self.dataset.get_vocab()

    def __len__(self):
        return len(self.dataset) // self.batch_size
    
    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data


class Image2latexDataset(data.Dataset):
    def __init__(self, opt, phase):
        self.opt = opt
        self.phase = phase
        self.root = opt.data_root
        self.image_paths = get_images(os.path.join(
            self.root, phase + '.matching.txt'))
        self.ids = list(self.image_paths.keys())
        print('#image: ', len(self.image_paths))
        self.vocab = opt.vocab
        self.labels = get_labels(os.path.join(
            self.root, phase + '.formulas.norm.txt'))
        self.transform = transforms.ToTensor()

    def load_data(self):
        return self

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(os.path.join(
            self.root, 'images_'+self.phase, image_path)).convert('L')
        if self.transform is not None:
            image = self.transform(image)
        skeleton = [self.opt.bos]
        skeleton.extend(
            [self.vocab[t] if t in self.vocab else self.vocab['<unk>'] for t in label])
        skeleton.append(self.opt.eos)
        target = torch.Tensor(skeleton)
        return image, target

    def get_vocab(self):
        return self.vocab

    def __len__(self):
        return len(self.image_paths)

class BucketSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.opt = data_source.opt

        buckets = dict()
        batch_size = self.opt.batch_size
        for id in self.data_source.ids:
            image_path = self.data_source.image_paths[id]
            size = get_image_size(os.path.join(
                self.opt.data_root, 'images_' + self.data_source.phase, image_path))
            if size in buckets:
                buckets[size].append(id)
            else:
                buckets[size] = [id]
        self.sorted_ids = []
        for size, ids in buckets.items():
            if self.opt.isTrain:
                random.shuffle(ids)
            self.sorted_ids.extend(
                np.array(ids[:(len(ids) // batch_size * batch_size)]).reshape(-1, batch_size).tolist())
        if self.opt.isTrain:
            random.shuffle(self.sorted_ids)
        # self.sorted_ids = chain.from_iterable(self.sorted_ids)
    
    def __iter__(self):
        return iter(self.sorted_ids)

    def __len__(self):
        return len(self.sorted_ids)

        
class Image2latexDataloader():
    def __init__(self, opt, phase='train'):
        self.dataset = Image2latexDataset(opt, phase)
        self.batch_size = opt.batch_size
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                      batch_sampler=BucketSampler(self.dataset),
                                                      num_workers=opt.nThreads,
                                                      collate_fn=collate_fn)

    def load_data(self):
        return self

    def get_vocab(self):
        return self.dataset.get_vocab()

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
