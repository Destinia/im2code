from data_loaders import get_rev_vocab
from Levenshtein import distance

vocab = get_rev_vocab()

def tokens2str(tokens):
    ret = ''
    for t in tokens:
        if vocab[t] == '<EOS>':
            break
        ret = ret + vocab[t] + ' '
    return ret


def wordErrorRate(results, targets):
    results_str = [tokens2str(r) for r in results]
    targets_str = [tokens2str(r) for r in targets]

    edit_distance = [min(1, distance(r, t)/len(t)) for r, t in zip(results_str, targets_str)]
    return sum(edit_distance)/len(edit_distance)
