import Levenshtein
from zss import Node, simple_distance



def wordErrorRate(results, targets, eos_index):
    def tokens2str(tokens):
        ret = ''
        for t in tokens:
            if int(t) == eos_index:
                break
            ret = ret + chr(int(t)) + ' '
        return ret
    results_str = [tokens2str(r, ) for r in results]
    targets_str = [tokens2str(r) for r in targets]

    edit_distance = [min(1, Levenshtein.distance(r, t)/len(t)) for r, t in zip(results_str, targets_str)]
    return 1 - sum(edit_distance)/len(edit_distance)


class TreeEditDistance():
    def __init__(self, opt):
        self.vocab = opt.rev_vocab
        self.opening_tag = opt.opening_tag
        self.closing_tag = opt.closing_tag
        self.eos = opt.eos

    @staticmethod
    def build_tree(self, seq):

        root = Node('root')
        hierarchy = [root, None] # 0 for parent 1 for current node
        for t in seq:
            token = self.vocab[t]
            if token == self.opening_tag:
                hierarchy.append(None)
                continue
            if token == self.closing_tag:
                del hierarchy[-1]
                continue
            if t == self.eos:
                break
            newNode = Node(token)
            hierarchy[-2].addkid(newNode)
            hierarchy[-1] = newNode
        return root

    @staticmethod
    def distance(self, result, target):
        try:
            return simple_distance(self.build_tree(result), self.build_tree(target))
        except:
            return 0 ## build tree error penalty


OPEN = 5
CLOSE = 6
EOS = 2


def build_tree(seq):
    root = Node('root')
    hierarchy = [root, None]  # 0 for parent 1 for current node
    for t in seq:
        token = chr(t)
        if t == OPEN:
            hierarchy.append(None)
            continue
        if t == CLOSE:
            del hierarchy[-1]
            continue
        if t == EOS:
            break
        newNode = Node(token)
        hierarchy[-2].addkid(newNode)
        hierarchy[-1] = newNode
    return root


def tree_distance(r, t):
    return simple_distance(build_tree(r), build_tree(t))/len(t)


