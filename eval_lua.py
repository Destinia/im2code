import torch
import os
from tqdm import tqdm
from options.test_options import TestOptions
from dataloader import UI2codeDataloader
from torch.autograd import Variable
from torch.utils.serialization import load_lua
from models.model import EncoderCNN, EncoderRNN, AttentionDecoder
import pickle
from eval_utils import wordErrorRate, tree_distances_multithread, wordErrorRateOrigin
import time
import numpy as np
from load_lua_model import load_encoderCNN, load_encoderRNN, load_decoder


def test_len(tokens):
    l = 0
    for t in tokens:
        if t == 2:
            return l
        l = l + 1
    return l


def main(opt):
    data_loader = UI2codeDataloader(opt, phase=opt.phase)
    opt.target_vocab_size = len(opt.vocab)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    encoderCNN_lua = load_lua('./cnn.t7')
    encoderCNN_lua.cuda()
    encoderCNN_lua.evaluate()
    encoderCNN = load_encoderCNN(opt)
    encoderCNN.eval()
    encoderRNN = load_encoderRNN(opt)
    encoderRNN.eval()
    decoder = load_decoder(opt)
    decoder.eval()
    if torch.cuda.is_available():
        encoderCNN.cuda()
        encoderRNN.cuda()
        decoder.cuda()
    accuracy_origin, accuracy_greedies, accuracy_tree = [], [], []
    average_lens = []
    for (images, captions, num_nonzeros) in tqdm(data_loader):
        images = Variable(images, requires_grad=False).cuda()
        # pre_feat = encoderCNN_lua.forward(images)
        features_lua = torch.stack(encoderCNN_lua.forward(images))

        # features = encoderCNN.forward((images-128)/128)
        encoded_features = encoderRNN(features_lua)
        # test_feat = test_encoderRNN(features_lua)
        output_greedy = decoder.beam(encoded_features).cpu().numpy()
        # beam_output, _ = decoder.decode_beam(encoded_features)
        # accuracy_beam = wordErrorRate(
        #     beam_output, captions[:, 1:], opt.eos)
        # accuracy_greedy = wordErrorRate(
        #     output_greedy, captions[:, 1:], opt.eos)
        # accuracy_beam = wordErrorRateOrigin(
        #     beam_output, captions[:, 1:], opt.rev_vocab)
        accuracy_greedy_origin = wordErrorRateOrigin(
            output_greedy, captions[:, 1:], opt.rev_vocab)
        accuracy_greedy = wordErrorRate(
            output_greedy, captions[:, 1:], opt.eos)
        # accuracy_tree = treeEval.distance(beam_output, captions.data[:, 1:])
        # tree_distance_beam = treeEval.distance(beam_output, captions.data[:, 1:])
        # start_time = time.time()
        accuracy_tree.append(np.average(tree_distances_multithread(
            output_greedy.tolist(), captions[:, 1:].tolist())))

        ## save for test
        avg_l = [test_len(r) for r in output_greedy]
        average_lens.append(sum(avg_l) / len(avg_l))
        accuracy_origin.append(accuracy_greedy_origin)
        accuracy_greedies.append(accuracy_greedy)
        # accuracy_beams.append(accuracy_beam)

        # print(accuracy_greedy, accuracy_beam, accuracy_tree)
        # print(accuracy_beam, accuracy_greedy)
    # print('accuracy beam search: %.4f'%(sum(accuracy_beams)/len(accuracy_beams)))
    print('Origin accuracy greedy search: %.4f' %
          (sum(accuracy_origin) / len(accuracy_origin)))
    print('Accuracy greedy search: %.4f' %
          (sum(accuracy_greedies) / len(accuracy_greedies)))
    print('Tree Accuracy greedy search: %.4f' %
          (sum(accuracy_tree) / len(accuracy_tree)))
    print('Average len: %.4f' %
          (sum(average_lens) / len(average_lens)))


if __name__ == '__main__':
    opt = TestOptions().parse()
    main(opt)
