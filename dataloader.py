import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
from PIL import Image

from util.util import get_vocab, get_images, get_labels


class UI2codeDataset(data.Dataset):
    def __init__(self, opt, phase):
        self.opt = opt
        self.root = opt.data_root
        self.image_paths = get_images(opt, phase)
        self.ids = list(self.image_paths.keys())
        print('#image: ', len(self.image_paths))
        self.vocab = opt.vocab
        self.labels = get_labels(opt)
        # self.transform = transforms.Compose([transforms.ToTensor(),
        #     transforms.Normalize([0.2731853791024895], [0.24186649347904463])])
        self.transform = transforms.ToTensor()

    def load_data(self):
        return self
    
    def __getitem__(self, index):
        image_path = self.image_paths[self.ids[index]]
        label = self.labels[self.ids[index]]
        image = Image.open(os.path.join(self.root, 'processedImage', image_path)).convert('L')
        if self.transform is not None:
            image = self.transform(image)
        skeleton = [self.vocab['<s>']]
        skeleton.extend([self.vocab[t] if t in self.vocab else self.vocab['<unk>'] for t in label])
        skeleton.append(self.vocab['</s>'])
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
    masks = torch.zeros(len(captions), max(lengths))
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
        masks[i, :lengths[i]] = 1
    return images, targets, masks


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

