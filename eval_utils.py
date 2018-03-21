from Levenshtein import distance



def wordErrorRate(results, targets, vocab):
    def tokens2str(tokens):
        ret = ''
        for t in tokens:
            if vocab[int(t)] == '</s>':
                break
            ret = ret + vocab[int(t)] + ' '
        return ret
    results_str = [tokens2str(r, ) for r in results]
    targets_str = [tokens2str(r) for r in targets]

    edit_distance = [min(1, distance(r, t)/len(t)) for r, t in zip(results_str, targets_str)]
    return 1 - sum(edit_distance)/len(edit_distance)
