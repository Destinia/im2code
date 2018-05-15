import Levenshtein
from zss import Node, simple_distance

from multiprocessing import Pool
import numpy as np
from ted import ted


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
PAD = 0
bot_component = [59, 56, 130, 243, 210, 109, 188, 70, 68, 4, 197, 24, 46, 77, 8, 293, 286, 154, 122, 57, 179, 145, 60, 14, 111, 84, 27, 80, 104, 134, 153, 165, 107, 54, 47, 76, 88, 277, 18, 20, 19, 207, 129,
         26, 92, 133, 37, 208, 25, 203, 49, 30, 247, 266, 9, 62, 10, 36, 74, 11, 12, 71, 22, 64, 52, 45, 73, 23, 21, 61, 48, 67, 155, 32, 260, 301, 271, 43, 16, 39, 33, 209, 7, 28, 95, 86, 65, 13, 167, 42, 78, 159]

bot_component_chr = list(map(lambda x: chr(x), bot_component))

def build_tree(seq, meta):
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

def build_tree_gt(seq, meta):
    root = Node('root')
    hierarchy = [root, None]  # 0 for parent 1 for current node
    for t in seq:
        token = chr(t)
        if t == OPEN:
            hierarchy.append(None)
            continue
        if t == CLOSE:
            del hierarchy[-1]
            if len(hierarchy) < 2:
                return root
            continue
        if t == EOS or t == PAD:
            break
        newNode = Node(token)
        hierarchy[-2].addkid(newNode)
        hierarchy[-1] = newNode
    return root

def weighted_distance(n1, n2):
    if n1 == n2:
        return 0
    bot1 = n1 in bot_component_chr
    bot2 = n2 in bot_component_chr
    if n2 == '' and not bot1:
        return 2
    if n1 == '' and not bot2:
        return 2
    if bot1 != bot2 :
        return 2
    return 1

def tree_distance(r, t, m):
    for i, token in enumerate(r):
        if token == m[2] or token == m[3]:
            r = r[:i]
            break
    if m[2] in t:
        t = t[:t.index(m[2])]
    length = len(t) - t.count(m[0])*2
    # print(r, tl, m, length)
    # print('result:', ted(r, tl, m), simple_distance(build_tree(
    #     r, m), build_tree_gt(tl, m)))
    # return 1 - min(1, simple_distance(build_tree(r, m), build_tree_gt(tl, m), label_dist=weighted_distance) / length)
    return 1 - min(1, ted(r, t, m) / length)
    

def tree_distances_multithread(results, targets, opt={}):

    # with Pool(processes=8) as pool:
    #     tree_distance_all = pool.starmap(tree_distance, zip(results, targets, [[OPEN, CLOSE, EOS, PAD]]*len(results)))
    if type(results):
        results = results.tolist()
    if type(targets):
        targets = targets.tolist()
    tree_distance_all = [tree_distance(
        r, t, [OPEN, CLOSE, EOS, PAD]) for r, t in zip(results, targets)]
    return np.array(tree_distance_all)
