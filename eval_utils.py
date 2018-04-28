import Levenshtein
from zss import Node, simple_distance

from multiprocessing import Pool
import numpy as np


def wordErrorRate(results, targets, eos_index):
    def tokens2str(tokens):
        ret = ''
        for t in tokens:
            if int(t) < 3:
                break
            ret = ret + chr(int(t)) + ' '
        return ret
    results_str = [tokens2str(r) for r in results]
    targets_str = [tokens2str(r) for r in targets]

    edit_distance = [min(1, Levenshtein.distance(r, t)/len(t)) for r, t in zip(results_str, targets_str)]
    return 1 - sum(edit_distance)/len(edit_distance)

def wordErrorRateOrigin(results, targets, vocab):
    def tokens2str(tokens):
        ret = ''
        for t in tokens:
            if vocab[int(t)] == '</s>':
                break
            ret = ret + vocab[int(t)] + ' '
        return ret
    results_str = [tokens2str(r) for r in results]
    targets_str = [tokens2str(r) for r in targets]

    edit_distance = [min(1, Levenshtein.distance(r, t)/len(t)) for r, t in zip(results_str, targets_str)]
    return 1 - sum(edit_distance)/len(edit_distance)

def wordErrorRateSplit(results, targets, vocab):
    def tokens2str(tokens):
        ret = ''
        for t in tokens:
            if vocab[int(t)] == '</s>':
                break
            ret = ret + vocab[int(t)] + ' '
        return ret
    results_str = [tokens2str(r) for r in results]
    targets_str = [tokens2str(r) for r in targets]

    edit_distance = [1 - min(1, Levenshtein.distance(r, t)/len(t)) for r, t in zip(results_str, targets_str)]
    return edit_distance




OPEN = 5
CLOSE = 6
EOS = 2


def build_tree(seq):
    root = Node('root')
    hierarchy = [root, None]  # 0 for parent 1 for current node
    for t in seq:
        try:
            token = chr(t)
            if t == OPEN:
                hierarchy.append(None)
                continue
            if t == CLOSE:
                del hierarchy[-1]
                if len(hierarchy) < 2:
                    return root
                continue
            if t == EOS:
                break
            newNode = Node(token)
            hierarchy[-2].addkid(newNode)
            hierarchy[-1] = newNode
        except:
            return root
    return root

def tree_distance(r, t):
    return 1 - min(1, simple_distance(build_tree(r), build_tree(t)) / len(t))

def tree_distances_multithread(results, targets):
    with Pool(processes=8) as pool:
        tree_distance_all = pool.starmap(tree_distance, zip(results, targets))
    
    return np.array(tree_distance_all)
