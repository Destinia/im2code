from __future__ import print_function
import torch
import numpy as np
import heapq
from PIL import Image
from torch.nn.utils import clip_grad_norm
import torch.nn as nn
from torch.autograd import Variable
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
    vocab['<pad>'] = 0
    vocab['<s>'] = 1
    vocab['</s>'] = 2
    vocab['<unk>'] = 3

    with open(opt.vocab_path, encoding='utf8') as f:
        for index, line in enumerate(f.readlines()):
            vocab[line.strip()] = index+4
    return vocab

def get_rev_vocab(opt):
    vocab = list()
    vocab.extend(['<pab>', '<s>', '</s>', '<unk>'])
    with open(opt.vocab_path, encoding='utf8') as f:
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

## for beam search
class Sequence(object):
    """Represents a complete or partial sequence."""

    def __init__(self, sentence, state, logprob, score, context):
        """Initializes the Sequence.
        Args:
          sentence: List of word ids in the sequence.
          state: Model state after generating the previous word.
          logprob: Log-probability of the sequence.
          score: Score of the sequence.
          context: image_feature context
        """
        self.sentence = sentence
        self.state = state
        self.logprob = logprob
        self.score = score
        self.context = context

    def __cmp__(self, other):
        """Compares Sequences by score."""
        assert isinstance(other, Sequence)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
        assert isinstance(other, Sequence)
        return self.score < other.score

    # Also for Python 3 compatibility.
    def __eq__(self, other):
        assert isinstance(other, Sequence)
        return self.score == other.score


class TopN(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN. This is a destructive operation.
        The only method that can be called immediately after extract() is reset().
        Args:
          sort: Whether to return the elements in descending sorted order.
        Returns:
          A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []

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


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        reward = to_contiguous(reward.repeat(input.size(1))).view(-1)
        input = to_contiguous(input).view(-1)
        mask = (seq > 0).float()
        mask = to_contiguous(
            torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * Variable(mask)
        output = torch.sum(output) / torch.sum(mask)

        return output

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def clip_norm_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            clip_grad_norm(param, grad_clip)


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr
