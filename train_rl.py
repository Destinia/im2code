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
    start_time = time.time()
    val_accuracy = []
    ted_val_accuracy = []
    encoderCNN, encoderRNN, decoder = models
    for (images, captions, lengths) in loader:
        images = Variable(images, requires_grad=False).cuda()
        captions = Variable(captions, requires_grad=False).cuda()
        features = encoderCNN(images)
        encoded_features = encoderRNN(features)
        greedy_outputs = decoder.decode(encoded_features)
        greedy_outputs = greedy_outputs.cpu().numpy()
        gt = captions.data.cpu().numpy()[:, 1:]
        accuracy = wordErrorRate(
            greedy_outputs, gt, opt.eos)
        ted_accuracy = tree_distances_multithread(
            greedy_outputs, gt)
        val_accuracy.append(accuracy)
        ted_val_accuracy.append(np.average(ted_accuracy))
    # print('validate cost time: %4f'% (time.time()-start_time))
    return sum(val_accuracy)/len(val_accuracy), sum(ted_val_accuracy)/len(ted_val_accuracy)

def train(opt):
    data_loader = UI2codeDataloader(opt)
    val_data_loader = UI2codeDataloader(opt, phase='val')
    dataset = data_loader.load_data()
    writer = SummaryWriter(opt.expr_dir)

    encoderCNN = EncoderCNN(opt)
    if opt.spatial:
        encoderRNN = SpatialEncoderRNN(opt)
    else:
        encoderRNN = EncoderRNN(opt)
    decoder = AttentionDecoder(opt)
    if opt.start_from:
        encoderCNN.load_state_dict(torch.load(os.path.join(
            opt.start_from, 'encoder-cnn.pkl')))
        encoderRNN.load_state_dict(torch.load(os.path.join(
            opt.start_from, 'encoder-rnn.pkl')))
        decoder.load_state_dict(torch.load(os.path.join(
            opt.start_from, 'decoder.pkl')))
    if torch.cuda.is_available():
        encoderCNN.cuda()
        encoderRNN.cuda()
        decoder.cuda()

    weight = torch.ones(opt.target_vocab_size).cuda()
    weight[0] = 0
    criterion = torch.nn.NLLLoss(weight=weight, size_average=False, reduce=False)
    opt.current_lr = opt.learning_rate
    optimizer_ec = torch.optim.SGD(encoderCNN.parameters(), lr=opt.current_lr)
    optimizer_er = torch.optim.SGD(encoderRNN.parameters(), lr=opt.current_lr)
    optimizer_de = torch.optim.SGD(decoder.parameters(), lr=opt.current_lr)
    total_steps = 0
    infos = {}
    histories = {}
    best_val_loss = math.inf
    best_val_accuracy = 0.0
    cur_val_accuracy, cur_ted_val_accuracy = validate(
        opt, val_data_loader, (encoderCNN, encoderRNN, decoder))
    print('Start word error rate: %4f' % (cur_val_accuracy))
    print('Start tree edit distance: %4f' % (cur_ted_val_accuracy))
    prev_ted_accuracy = cur_ted_val_accuracy

    for epoch in range(opt.num_epochs):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        print('Epoch [%d/%d]' % (epoch + 1, opt.num_epochs))
        pbar = tqdm(data_loader)
        val_accuracy = []
        ted_val_accuracy = []
        for (images, gt, num_nonzeros) in pbar:
            batch_size = images.size(0)
            iter_start_time = time.time()
            total_steps += opt.batch_size
            images = Variable(images, requires_grad=False).cuda()
            captions = Variable(gt, requires_grad=False).cuda()
            features = encoderCNN(images)
            encoded_features = encoderRNN(features)
            # outputs, sample_logprobs = decoder.sample_max(encoded_features, {'temperature': 0.5})
            outputs = decoder(encoded_features, captions)
            _, results = torch.max(outputs, 1)
            results = results.cpu().numpy()
            reward = tree_distances_multithread(results, gt[:, 1:])
            avg_reward = np.average(reward)
            loss = 0.0
            optimizer_ec.zero_grad()
            optimizer_er.zero_grad()
            optimizer_de.zero_grad()
            for t in range(captions.size(1) - 1):
                loss += criterion(outputs[:, t], captions[:, 1 + t]) / batch_size
            loss = (loss * (torch.from_numpy(reward-avg_reward).float().cuda())).sum()
            loss.backward()
            utils.grad_clip(encoderCNN.parameters(), opt.norm_grad_clip)
            utils.grad_clip(encoderRNN.parameters(), opt.norm_grad_clip)
            utils.grad_clip(decoder.parameters(), opt.norm_grad_clip)
            optimizer_ec.step()
            optimizer_er.step()
            optimizer_de.step()
            train_loss = loss.data[0] * batch_size / num_nonzeros
            pbar.set_description('Loss: %.4f'
                                 % (np.average(reward)))
            pbar.refresh()
            if total_steps % opt.print_freq == 0:
                writer.add_scalar('train_loss', train_loss, total_steps)
                writer.add_scalar('learning_rate', opt.current_lr, total_steps)
            if  total_steps % opt.save_checkpoint_every == 0:
                cur_val_accuracy, cur_ted_val_accuracy = validate(
                    opt, val_data_loader, (encoderCNN, encoderRNN, decoder))

                writer.add_scalar('val_accuracy', cur_val_accuracy, total_steps)
                writer.add_scalar('val_ted_accuracy', cur_ted_val_accuracy, total_steps)
                if cur_ted_val_accuracy > best_val_accuracy:
                    best_val_accuracy = cur_ted_val_accuracy
                    print('save best model at Epoch: %d'%(epoch))
                    torch.save(decoder.state_dict(), os.path.join(
                        opt.expr_dir, 'decoder-best.pkl'))
                    torch.save(encoderRNN.state_dict(), os.path.join(
                        opt.expr_dir, 'encoder-rnn-best.pkl'))
                    torch.save(encoderCNN.state_dict(), os.path.join(
                        opt.expr_dir, 'encoder-cnn-best.pkl'))
                    print('validation accuracy: %.4f\nBest validation accuracy: %.4f' %
                        (cur_ted_val_accuracy, best_val_accuracy))

        print('the end of epoch %d, iters %d' % (epoch+1, total_steps))

        cur_val_accuracy, cur_ted_val_accuracy = validate(
            opt, val_data_loader, (encoderCNN, encoderRNN, decoder))

        writer.add_scalar('val_accuracy', cur_val_accuracy, total_steps)
        writer.add_scalar('val_ted_accuracy', cur_ted_val_accuracy, total_steps)
        if cur_ted_val_accuracy > best_val_accuracy:
            best_val_accuracy = cur_ted_val_accuracy
            print('save best model at Epoch: %d'%(epoch))
            torch.save(decoder.state_dict(), os.path.join(
                opt.expr_dir, 'decoder-best.pkl'))
            torch.save(encoderRNN.state_dict(), os.path.join(
                opt.expr_dir, 'encoder-rnn-best.pkl'))
            torch.save(encoderCNN.state_dict(), os.path.join(
                opt.expr_dir, 'encoder-cnn-best.pkl'))
            print('validation accuracy: %.4f\nBest validation accuracy: %.4f' %
                (cur_ted_val_accuracy, best_val_accuracy))
            ## update learning rate
        if cur_ted_val_accuracy > prev_ted_accuracy:
            opt.current_lr = max(opt.current_lr * opt.lr_decay,
                                opt.learning_rate_min)
            print('previous epoch validation accuracy: %.4f\ncurrent epoch validation accuracy: %.4f' %
                  (prev_ted_accuracy, cur_ted_val_accuracy))
            print('update learning rate: %.4f' % (opt.current_lr))
            utils.set_lr(optimizer_ec, opt.current_lr)
            utils.set_lr(optimizer_er, opt.current_lr)
            utils.set_lr(optimizer_de, opt.current_lr)



        print('End of epoch %d \t Time Taken: %d sec' %
            (epoch, time.time() - epoch_start_time))

if __name__ == '__main__':
    opt = TrainOptions().parse()
    train(opt)
