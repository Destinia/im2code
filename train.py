from options.train_options import TrainOptions
from dataloader import UI2codeDataloader

import numpy as np
import math
import torch
import torch.nn as nn
import pickle
import time
import os
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.nn.utils.rnn import pack_padded_sequence
from models.model import EncoderCNN, EncoderRNN, AttentionDecoder, SpatialEncoderRNN
import util.util as utils
from eval_utils import wordErrorRate, tree_distances_multithread
from tensorboardX import SummaryWriter


def validate(opt, loader, models):
    val_accuracy = []
    ted_val_accuracy = []
    encoderCNN, encoderRNN, decoder = models
    for (images, captions, masks) in loader:
        images = Variable(images, requires_grad=False).cuda()
        captions = Variable(captions, requires_grad=False).cuda()
        masks = Variable(masks, requires_grad=False).cuda()
        features = encoderCNN(images)
        encoded_features = encoderRNN(features)
        greedy_outputs = decoder.decode(encoded_features)
        greedy_outputs = greedy_outputs.cpu().numpy()
        gt = captions.data.cpu().numpy()[:, 1:]
        accuracy = wordErrorRate(
            greedy_outputs, gt, opt.eos)
        ted_accuracy = tree_distances_multithread(
            greedy_outputs, gt[:, 1:])
        val_accuracy.append(accuracy)
        ted_val_accuracy.append(np.average(ted_accuracy))
    return sum(val_accuracy) / len(val_accuracy), sum(ted_val_accuracy) / len(ted_val_accuracy)


def train(opt):
    data_loader = UI2codeDataloader(opt)
    val_data_loader = UI2codeDataloader(opt, phase='val')
    dataset = data_loader.load_data()
    writer = SummaryWriter(opt.expr_dir)

    # model = create_model(opt)
    # visualizer = Visualizer(opt)
    encoderCNN = EncoderCNN(opt)
    if opt.spatial:
        encoderRNN = SpatialEncoderRNN(opt)
    else:
        encoderRNN = EncoderRNN(opt)
    decoder = AttentionDecoder(opt)
    if torch.cuda.is_available():
        encoderCNN.cuda()
        encoderRNN.cuda()
        decoder.cuda()

    criterion = utils.LanguageModelCriterion()
    params = list(decoder.parameters()) + \
        list(encoderRNN.parameters()) + list(encoderCNN.parameters())
    opt.current_lr = opt.learning_rate
    optimizer = torch.optim.SGD(params, lr=opt.current_lr)
    total_steps = 0
    infos = {}
    histories = {}
    best_val_loss = math.inf
    best_val_accuracy = 0.0

    for epoch in range(opt.num_epochs):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        print('Epoch [%d/%d]' % (epoch + 1, opt.num_epochs))
        pbar = tqdm(data_loader)
        for (images, captions, masks) in pbar:
            iter_start_time = time.time()
            total_steps += opt.batch_size
            images = Variable(images, requires_grad=False).cuda()
            captions = Variable(captions, requires_grad=False).cuda()
            masks = Variable(masks, requires_grad=False).cuda()
            # targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            features = encoderCNN(images)
            encoded_features = encoderRNN(features)
            outputs = decoder(encoded_features, captions)
            loss = criterion(outputs[:, :-1], captions[:,1:], masks[:,1:])
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(params, opt.norm_grad_clip)
            optimizer.step()
            train_loss = loss.data[0]
            pbar.set_description('Loss: %.4f'
                                 % (loss.data[0]))
            pbar.refresh()
            if total_steps % opt.print_freq == 0:
                # print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                #       % (epoch+1, opt.num_epochs, total_steps, dataset_size,
                #          loss.data[0]))
                writer.add_scalar('train_loss', train_loss, total_steps)
                writer.add_scalar('learning_rate', opt.current_lr, total_steps)
                # add_summary_value(tf_summary_writer, 'scheduled_sampling_prob', model.ss_prob, total_steps)
            ## validation
            # if total_steps % opt.save_latest_freq == 0:
            #     print('saving the latest model (epoch %d, total_steps %d)' %
            #         (epoch, total_steps))
            #     torch.save(decoder.state_dict(), os.path.join(opt.checkpoint_path, 'decoder-%d-%d.pkl' % (epoch + 1, total_steps)))
            #     torch.save(encoderRNN.state_dict(), os.path.join(opt.checkpoint_path, 'encoder-rnn-%d-%d.pkl' % (epoch + 1, total_steps)))
            #     torch.save(encoderCNN.state_dict(), os.path.join(opt.checkpoint_path, 'encoder-cnn-%d-%d.pkl' % (epoch + 1, total_steps)))

        print('the end of epoch %d, iters %d' % (epoch, total_steps))

        cur_val_accuracy, cur_ted_val_accuracy = validate(
            opt, val_data_loader, (encoderCNN, encoderRNN, decoder))
        
        writer.add_scalar('val_accuracy', cur_val_accuracy, epoch)
        writer.add_scalar('val_ted_accuracy', cur_ted_val_accuracy, epoch)
        if cur_val_accuracy > best_val_accuracy:
            best_val_accuracy = cur_val_accuracy
            print('save best model at Epoch: %d'%(epoch))
            torch.save(decoder.state_dict(), os.path.join(
                opt.expr_dir, 'decoder-best.pkl'))
            torch.save(encoderRNN.state_dict(), os.path.join(
                opt.expr_dir, 'encoder-rnn-best.pkl'))
            torch.save(encoderCNN.state_dict(), os.path.join(
                opt.expr_dir, 'encoder-cnn-best.pkl'))

            ## update learning rate
            opt.current_lr = max(opt.current_lr * opt.lr_decay,
                                opt.learning_rate_min)
        print('update learning rate: %.4f' % (opt.current_lr))
        utils.set_lr(optimizer, opt.current_lr)

        print('validation accuracy: %.4f\nBest validation accuracy: %.4f' %
              (cur_val_accuracy, best_val_accuracy))

        torch.save(decoder.state_dict(),
                   os.path.join(opt.expr_dir,
                                'decoder.pkl'))
        torch.save(encoderRNN.state_dict(),
                   os.path.join(opt.expr_dir,
                                'encoder-rnn.pkl'))
        torch.save(encoderCNN.state_dict(),
                   os.path.join(opt.expr_dir,
                                'encoder-cnn.pkl'))
        print('End of epoch %d \t Time Taken: %d sec' %
            (epoch, time.time() - epoch_start_time))

if __name__ == '__main__':
    opt = TrainOptions().parse()
    train(opt)
