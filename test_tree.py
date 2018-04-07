import time
import pickle
from multiprocessing import Pool, cpu_count
from dataloader import UI2codeDataloader
from options.test_options import TestOptions
from zss import Node, simple_distance
from multiprocessing import Pool
from eval_utils import TreeEditDistance, distance


def test(a, b):
    print(a, b)
    return 0

def main(opt):
    # tree_eval = TreeEditDistance(opt)
    # def eval_func(r, t):
    #     return tree_eval.distance(r, t)
    data_loader = UI2codeDataloader(opt, phase='val')
    results = pickle.load(open('/save/test_result.pkl', 'rb'))
    # distance = build_tree_creator(opt)
    for i, (images, captions, masks) in enumerate(data_loader):
        start_time = time.time()
        # distance(results[0], captions[0])
        with Pool(processes=16) as pool:
            tree_distance_greedy = pool.starmap(distance, zip(
                results[i], captions.numpy()[:, 1:]))
        print('distance', tree_distance_greedy)
        print('time consume: ', time.time()-start_time)


if __name__ == '__main__':
    opt = TestOptions().parse()
    main(opt)
