from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import os


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_vocab(opt):
    vocab = dict()
    vocab['<PAD>'] = 0
    vocab['<BOS>'] = 1
    vocab['<EOS>'] = 2
    vocab['<UNK>'] = 3

    with open(opt.vocab_path) as f:
        for index, line in enumerate(f.readlines()):
            vocab[line.strip()] = index+4
    return vocab

def get_rev_vocab(opt):
    vocab = list()
    vocab.extend(['<PAD>', '<BOS>', '<EOS>', '<UNK>'])
    with open(opt.vocab_path) as f:
        for index, line in enumerate(f.readlines()):
            vocab.append(line.strip())
    return vocab

def get_images(opt, phase):
    images = list()
    data_path = ''
    if phase == 'train':
        data_path=opt.train_data_path
    elif phase == 'val':
        data_path=opt.val_data_path
    elif phase == 'test':
        data_path=opt.test_data_path
    with open(data_path) as f:
        lines = [line.strip().split() for line in f.readlines()]
        images = {int(line[1]):line[0] for line in lines}
    return images


def get_labels(opt):
    with open(opt.label_path) as f:
        return [f.split() for f in f.readlines()]
   

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output
