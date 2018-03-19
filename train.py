from options.train_options import TrainOptions
from dataloader import UI2codeDataloader

import numpy as np
import torch
import torch.nn as nn
import pickle
import time
import os
from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence
from models.model import EncoderCNN, EncoderRNN, AttentionDecoder
from util.util import LanguageModelCriterion

try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None


def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)

# def validate(model, loader):


def train(opt):
    data_loader = UI2codeDataloader(opt)
    opt.target_vocab_size = len(data_loader.get_vocab())
    val_data_loader = UI2codeDataloader(opt, phase='val')
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    tf_summary_writer = tf and tf.summary.FileWriter(opt.checkpoint_path)

    # model = create_model(opt)
    # visualizer = Visualizer(opt)
    encoderCNN = EncoderCNN(opt)
    encoderRNN = EncoderRNN(opt)
    decoder = AttentionDecoder(opt)
    if torch.cuda.is_available():
        encoderCNN.cuda()
        encoderRNN.cuda()
        decoder.cuda()

    criterion = LanguageModelCriterion()
    params = list(decoder.parameters()) + \
        list(encoderRNN.parameters()) + list(encoderCNN.parameters())
    optimizer = torch.optim.SGD(params, lr=opt.learning_rate)
    total_steps = 0
    infos = {}
    histories = {}

    for epoch in range(opt.num_epochs):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, (images, captions, masks) in enumerate(data_loader):
            iter_start_time = time.time()
            total_steps += opt.batch_size
            images = Variable(images, requires_grad=False).cuda()
            captions = Variable(captions, requires_grad=False).cuda()
            masks = Variable(masks, requires_grad=False).cuda()
            # targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            features = encoderCNN(images)
            encoded_features = encoderRNN(features)
            outputs = decoder(encoded_features, captions)
            loss = criterion(outputs, captions, masks)
            loss.backward()
            optimizer.step()
            train_loss = loss.data[0]

            if total_steps % opt.print_freq == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                      % (epoch, opt.num_epochs, total_steps, dataset_size,
                         loss.data[0]))
                if tf is not None:
                    add_summary_value(tf_summary_writer, 'train_loss', train_loss, total_steps)
                    add_summary_value(tf_summary_writer, 'learning_rate', opt.learning_rate, total_steps)
                    # add_summary_value(tf_summary_writer, 'scheduled_sampling_prob', model.ss_prob, total_steps)
                    tf_summary_writer.flush()
            ## validation
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, total_steps))
                torch.save(decoder.state_dict(),
                           os.path.join(opt.checkpoint_path,
                                        'decoder-%d-%d.pkl' % (epoch + 1, total_steps)))
                torch.save(encoderRNN.state_dict(),
                           os.path.join(opt.checkpoint_path,
                                        'encoder-rnn-%d-%d.pkl' % (epoch + 1, total_steps)))
                torch.save(encoderCNN.state_dict(),
                           os.path.join(opt.checkpoint_path,
                                        'encoder-cnn-%d-%d.pkl' % (epoch + 1, total_steps)))

        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        torch.save(decoder.state_dict(),
                   os.path.join(opt.checkpoint_path,
                                'decoder-%d.pkl' % (epoch + 1)))
        torch.save(encoderRNN.state_dict(),
                   os.path.join(opt.checkpoint_path,
                                'encoder-rnn-%d.pkl' % (epoch + 1)))
        torch.save(encoderCNN.state_dict(),
                   os.path.join(opt.checkpoint_path,
                                'encoder-cnn-%d.pkl' % (epoch + 1)))

        print('End of epoch %d \t Time Taken: %d sec' %
            (epoch, time.time() - epoch_start_time))

if __name__ == '__main__':
    opt = TrainOptions().parse()
    train(opt)
