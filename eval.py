import torch
import os
from tqdm import tqdm
from options.test_options import TestOptions
from dataloader import UI2codeDataloader
from torch.autograd import Variable

from models.model import EncoderCNN, EncoderRNN, AttentionDecoder
import pickle
from eval_utils import wordErrorRate, TreeEditDistance

def main(opt):
    data_loader = UI2codeDataloader(opt, phase='val')
    opt.target_vocab_size = len(opt.vocab)
    treeEval = TreeEditDistance(opt)
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
    accuracy_beams, accuracy_greedies = [], []
    for (images, captions, masks) in tqdm(data_loader):
        images = Variable(images, requires_grad=False).cuda()
        captions = Variable(captions, requires_grad=False).cuda()
        masks = Variable(masks, requires_grad=False).cuda()

        features = encoderCNN(images)
        encoded_features = encoderRNN(features)
        output_greedy = decoder.decode(encoded_features)
        beam_output = decoder.beam_search(encoded_features)
        accuracy_beam = wordErrorRate(
            beam_output, captions.data[:, 1:], opt.eos)
        accuracy_greedy = wordErrorRate(
            output_greedy, captions.data[:, 1:], opt.eos)
        # accuracy_tree = treeEval.distance(beam_output, captions.data[:, 1:])
        # tree_distance_beam = treeEval.distance(beam_output, captions.data[:, 1:])
        # tree_distance_greedy = treeEval.distance(beam_output, captions.data[:, 1:])
        accuracy_greedies.append(accuracy_greedy)
        accuracy_beams.append(accuracy_beam)

        # print(accuracy_greedy, accuracy_beam, accuracy_tree)
        # print(accuracy_beam, accuracy_greedy)
    print('accuracy beam search: %.4f'%(sum(accuracy_beams)/len(accuracy_beams)))
    print('accuracy greedy search: %.4f'%(sum(accuracy_greedies)/len(accuracy_greedies)))
if __name__ == '__main__':
    opt = TestOptions().parse()
    main(opt)
