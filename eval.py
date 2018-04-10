import torch
import os
from tqdm import tqdm
from options.test_options import TestOptions
from dataloader import UI2codeDataloader
from torch.autograd import Variable

from models.model import EncoderCNN, EncoderRNN, AttentionDecoder
import pickle
from eval_utils import wordErrorRate, tree_distance, wordErrorRateOrigin
from multiprocessing import Pool
import time

def main(opt):
    data_loader = UI2codeDataloader(opt, phase='val')
    opt.target_vocab_size = len(opt.vocab)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    encoderCNN = EncoderCNN(opt)
    encoderRNN = EncoderRNN(opt)
    decoder = AttentionDecoder(opt)
    encoderCNN.load_state_dict(torch.load(os.path.join(
        opt.results_dir, 'encoder-cnn-%s.pkl' % (opt.model_name))))
    encoderRNN.load_state_dict(torch.load(os.path.join(
        opt.results_dir, 'encoder-rnn-%s.pkl' % (opt.model_name))))
    decoder.load_state_dict(torch.load(os.path.join(
        opt.results_dir, 'decoder-%s.pkl' % (opt.model_name))))

    if torch.cuda.is_available():
        encoderCNN.cuda()
        encoderRNN.cuda()
        decoder.cuda()
    accuracy_origin, accuracy_greedies, accuracy_tree = [], [], []
    for (images, captions, masks) in tqdm(data_loader):
        images = Variable(images, requires_grad=False).cuda()
        masks = Variable(masks, requires_grad=False).cuda()

        features = encoderCNN(images)
        encoded_features = encoderRNN(features)
        output_greedy = decoder.decode(encoded_features)
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
        output_greedy = output_greedy.cpu().numpy()
        # start_time = time.time()
        with Pool(processes=16) as pool:
            tree_distance_greedy = pool.starmap(tree_distance, zip(
                output_greedy, captions.numpy()[:, 1:]))
            tree_distance_greedy_accuracy = sum(tree_distance_greedy)/len(tree_distance_greedy)
        accuracy_tree.append(tree_distance_greedy_accuracy)

        ## save for test
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
    print('Accuracy greedy search: %.4f' %
          (sum(accuracy_tree) / len(accuracy_tree)))
if __name__ == '__main__':
    opt = TestOptions().parse()
    main(opt)
