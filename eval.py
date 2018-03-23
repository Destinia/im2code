import torch
import os
from options.test_options import TestOptions
from dataloader import UI2codeDataloader
from torch.autograd import Variable

from models.model import EncoderCNN, EncoderRNN, AttentionDecoder
import pickle
from eval_utils import wordErrorRate

def main(opt):
    data_loader = UI2codeDataloader(opt, phase='train')
    opt.vocab = data_loader.get_vocab()
    opt.rev_vocab = [t[0]
                    for t in list(sorted(opt.vocab.items(), key=lambda x: x[1]))]
    opt.target_vocab_size = len(opt.vocab)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    encoderCNN = EncoderCNN(opt)
    encoderRNN = EncoderRNN(opt)
    decoder = AttentionDecoder(opt)
    encoderCNN.load_state_dict(torch.load(os.path.join(
        opt.results_dir, 'encoder-cnn-%d.pkl' % (opt.model_iter))))
    encoderRNN.load_state_dict(torch.load(os.path.join(
        opt.results_dir, 'encoder-rnn-%d.pkl' % (opt.model_iter))))
    decoder.load_state_dict(torch.load(os.path.join(
        opt.results_dir, 'decoder-%d.pkl' % (opt.model_iter))))

    if torch.cuda.is_available():
        encoderCNN.cuda()
        encoderRNN.cuda()
        decoder.cuda()
    for (images, captions, masks) in data_loader:
        images = Variable(images, requires_grad=False).cuda()
        captions = Variable(captions, requires_grad=False).cuda()
        masks = Variable(masks, requires_grad=False).cuda()

        features = encoderCNN(images)
        encoded_features = encoderRNN(features)
        output_greedy = decoder.decoder(encoded_features)
        beam_output, beam_score = decoder.decode_beam(encoded_features)
        accuracy = wordErrorRate(
            output_greedy[:, :-1], captions.data[:, 1:], opt.rev_vocab)
        print(accuracy)

if __name__ == '__main__':
    opt = TestOptions().parse()
    main(opt)
