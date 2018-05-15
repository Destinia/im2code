import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from .BeamSearch import Beam
from util.util import beam_replicate
import random

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1)

class EncoderCNN(nn.Module):
    def __init__(self, opt):
        super(EncoderCNN, self).__init__()
        self.models = [
            conv3x3(1, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False),
            conv3x3(64, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False),
            conv3x3(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            conv3x3(256, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, ceil_mode=False),
            conv3x3(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, ceil_mode=False),
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        ]
        self.model = nn.Sequential(*self.models)

    def forward(self, x):
        # for l in self.models:
        #     x = l(x)
        output = self.model(x) # batch * 512 * H * W
        output = output.permute(0, 2, 3, 1) # batch * H * W * 512
        # output = torch.unbind(output, 1) # len(H) list of (batch * W * 512)
        return output

class EncoderRNN(nn.Module):
    def __init__(self, opt):
        super(EncoderRNN, self).__init__()
        self.batch_size = opt.batch_size
        self.input_size = opt.cnn_feature_size
        self.max_encoder_l_h = opt.max_encoder_l_h
        self.encoder_num_hidden = opt.encoder_num_hidden
        self.n_layers = opt.encoder_num_layers
        # 4* for bidirectional and (h, c)
        self.pos_embedding_fw = nn.Embedding(
            opt.max_encoder_l_h, self.n_layers * self.encoder_num_hidden * 2)
        self.pos_embedding_bw = nn.Embedding(
            opt.max_encoder_l_h, self.n_layers * self.encoder_num_hidden * 2)
        self.lstm = nn.LSTM(
            self.input_size, self.encoder_num_hidden, self.n_layers, bidirectional=True, batch_first=True)

    def forward(self, img_feats):
        """
        img_feature shape: batch * H * W * hidden_dim
        """
        imgH = img_feats.size(1)
        outputs = []
        for i in range(imgH):  # imgSeq height
            pos = Variable(torch.LongTensor(
                [i] * img_feats.size(0)), requires_grad=False).cuda().contiguous()  # batch * (num_layer * 2) * hidden_dim
            # (num_layer * 2) * batch * hidden_dim
            pos_embedding_fw_h, pos_embedding_fw_c = torch.unbind(self.pos_embedding_fw(
                pos).view(-1, 2 * self.n_layers, self.encoder_num_hidden), 1)
            pos_embedding_bw_h, pos_embedding_bw_c = torch.unbind(self.pos_embedding_bw(
                pos).view(-1, 2 * self.n_layers, self.encoder_num_hidden), 1)
            pos_embedding_h = torch.cat(
                [pos_embedding_fw_h.unsqueeze(0), pos_embedding_bw_h.unsqueeze(0)], 0).contiguous()
            pos_embedding_c = torch.cat(
                [pos_embedding_fw_c.unsqueeze(0), pos_embedding_bw_c.unsqueeze(0)], 0).contiguous()
            source = img_feats[:, i]  # batch * imgW * hidden_dim
            output, _ = self.lstm(source, (pos_embedding_h, pos_embedding_c))
            outputs.append(output)
        return torch.cat(outputs, 1)
class SpatialEncoderRNN(nn.Module):
    def __init__(self, opt):
        super(SpatialEncoderRNN, self).__init__()
        self.batch_size = opt.batch_size
        self.input_size = opt.cnn_feature_size
        self.max_encoder_l_h = opt.max_encoder_l_h
        self.encoder_num_hidden = opt.encoder_num_hidden
        self.n_layers = opt.encoder_num_layers
        # 4* for bidirectional and (h, c)
        self.pos_embedding_w_fw = nn.Embedding(
            opt.max_encoder_l_h, self.n_layers * self.encoder_num_hidden * 2)
        self.pos_embedding_w_bw = nn.Embedding(
            opt.max_encoder_l_h, self.n_layers * self.encoder_num_hidden * 2)
        self.pos_embedding_h_fw = nn.Embedding(
            opt.max_encoder_l_w, self.n_layers * self.encoder_num_hidden * 2)
        self.pos_embedding_h_bw = nn.Embedding(
            opt.max_encoder_l_w, self.n_layers * self.encoder_num_hidden * 2)
        # self.pos_embedding_bw = nn.Embedding(
        #     opt.max_encoder_l_h, self.n_layers * self.encoder_num_hidden * 2)
        self.lstm_w = nn.LSTM(
            self.input_size, self.encoder_num_hidden, self.n_layers, bidirectional=True, batch_first=True)
        self.lstm_h = nn.LSTM(
            self.input_size, self.encoder_num_hidden, self.n_layers, bidirectional=True, batch_first=True)
        
    def forward(self, img_feats):
        """
        img_feature shape: batch * H * W * hidden_dim
        """
        imgH = img_feats.size(1)
        imgW = img_feats.size(2)
        outputs_w = []
        outputs_h = []
        for i in range(imgH):  # imgSeq height
            pos = Variable(torch.LongTensor(
                [i] * img_feats.size(0)), requires_grad=False).cuda().contiguous()  # batch * (num_layer * 2) * hidden_dim
            # (num_layer * 2) * batch * hidden_dim
            pos_embedding_fw_h, pos_embedding_fw_c = torch.unbind(self.pos_embedding_w_fw(
                pos).view(-1, 2 * self.n_layers, self.encoder_num_hidden), 1)
            pos_embedding_bw_h, pos_embedding_bw_c = torch.unbind(self.pos_embedding_w_bw(
                pos).view(-1, 2 * self.n_layers, self.encoder_num_hidden), 1)
            pos_embedding_h = torch.cat(
                [pos_embedding_fw_h.unsqueeze(0), pos_embedding_bw_h.unsqueeze(0)], 0).contiguous()
            pos_embedding_c = torch.cat(
                [pos_embedding_fw_c.unsqueeze(0), pos_embedding_bw_c.unsqueeze(0)], 0).contiguous()
            source = img_feats[:, i]  # batch * imgW * hidden_dim
            output, _ = self.lstm_w(source, (pos_embedding_h, pos_embedding_c))
            outputs_w.append(output)
        for i in range(imgW):  # imgSeq height
            pos = Variable(torch.LongTensor(
                [i] * img_feats.size(0)), requires_grad=False).cuda().contiguous()  # batch * (num_layer * 2) * hidden_dim
            # (num_layer * 2) * batch * hidden_dim
            pos_embedding_fw_h, pos_embedding_fw_c = torch.unbind(self.pos_embedding_h_fw(
                pos).view(-1, 2 * self.n_layers, self.encoder_num_hidden), 1)
            pos_embedding_bw_h, pos_embedding_bw_c = torch.unbind(self.pos_embedding_h_bw(
                pos).view(-1, 2 * self.n_layers, self.encoder_num_hidden), 1)
            pos_embedding_h = torch.cat(
                [pos_embedding_fw_h.unsqueeze(0), pos_embedding_bw_h.unsqueeze(0)], 0).contiguous()
            pos_embedding_c = torch.cat(
                [pos_embedding_fw_c.unsqueeze(0), pos_embedding_bw_c.unsqueeze(0)], 0).contiguous()
            source = img_feats[:, :, i]  # batch * imgW * hidden_dim
            output, _ = self.lstm_h(source, (pos_embedding_h, pos_embedding_c))
            outputs_h.append(output)
        outputs_w_t = torch.cat([_.unsqueeze(1) for _ in outputs_w], 1) # batch * imgH * imgW * num_hidden
        outputs_h_t = torch.cat([_.unsqueeze(1) for _ in outputs_h], 1).transpose(1, 2)

        return torch.cat((outputs_w_t, outputs_h_t), -1).contiguous().view(-1, imgH*imgW, 4*self.encoder_num_hidden)

class AttentionDecoder(nn.Module):
    def __init__(self, opt):
        super(AttentionDecoder, self).__init__()
        self.context_num_hidden = opt.context_num_hidden
        self.decoder_num_hidden = opt.decoder_num_hidden
        self.decoder_num_layers = opt.decoder_num_layers
        self.target_vocab_size = opt.target_vocab_size ## dict size + 4
        self.target_embedding_size = opt.target_embedding_size
        self.max_decoder_l = opt.max_decoder_l
        self.dropout = opt.dropout
        self.output_dropout = nn.Dropout(opt.dropout)
        self.beam_size = opt.beam_size
        self.vocab = opt.vocab
        self.bos = opt.bos
        self.eos = opt.eos
        self.embed = nn.Embedding(self.target_vocab_size, self.target_embedding_size)
        self.core = UI2codeAttention(self.target_embedding_size, self.context_num_hidden, self.decoder_num_hidden, self.decoder_num_layers)
        self.logit = nn.Linear(self.decoder_num_hidden, self.target_vocab_size)
        self.ss_prob = 0.0


    def init_hidden(self, bsz, batch_first=False):
        weight = next(self.parameters()).data
        if batch_first:
            return (Variable(weight.new(bsz, self.decoder_num_layers, self.decoder_num_hidden).zero_()),
                    Variable(weight.new(bsz, self.decoder_num_layers, self.decoder_num_hidden).zero_()))
        return (Variable(weight.new(bsz, self.decoder_num_hidden).zero_()),
                Variable(weight.new(bsz, self.decoder_num_hidden).zero_()),
                Variable(weight.new(bsz, self.context_num_hidden).zero_()))

    def extract_model(self):
        models = dict()
        models['embed'] = self.embed
        models['core'] = self.core
        models['logit'] = self.logit

        return models

    def forward(self, cnn_feats, seq):
        """
        - cnn_feats shape: batch_size * (H * W) * (4 * encoder_num_hidden)
        - seq shape: batch_size * #tokens * vocab_size
        """
        batch_size = cnn_feats.size(0)
        state = self.init_hidden(batch_size)

        outputs = []


        for i in range(seq.size(1) - 1):
            if i == 0:
                it = seq[:, i].clone()
            else:
                if i >= 2 and self.ss_prob > 0.0: # otherwiste no need to sample
                    sample_prob = cnn_feats.data.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, i].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, i].data.clone()
                        #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                        #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                        prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                        it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                        it = Variable(it, requires_grad=False)
                else:
                    it = seq[:, i].clone()

            xt = self.embed(it) # batch * embedding_dim
            output, state = self.core(xt, cnn_feats, state)
            if self.dropout:
                output = self.output_dropout(output)
            output = F.log_softmax(self.logit(output)) # batch * vocab_size
            outputs.append(output)

        return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def decode(self, cnn_feats):
        batch_size = cnn_feats.size(0)
        state = self.init_hidden(batch_size)

        outputs = []
        it = Variable(torch.LongTensor([self.vocab['<s>']]
                      * batch_size)).contiguous().cuda()
        for i in range(self.max_decoder_l):
            xt = self.embed(it)  # batch * embedding_dim
            output, state = self.core(xt, cnn_feats, state)
            output = F.log_softmax(self.logit(output))  # batch * vocab_size
            _, output = torch.max(output, 1)
            it = output
            outputs.append(output)

        return torch.cat([_.unsqueeze(1) for _ in outputs], 1).data

    def vis(self, cnn_feats):
        batch_size = cnn_feats.size(0)
        state = self.init_hidden(batch_size)

        outputs = []
        attn = []
        it = Variable(torch.LongTensor([self.vocab['<s>']]
                      * batch_size)).contiguous().cuda()
        for i in range(self.max_decoder_l):
            xt = self.embed(it)  # batch * embedding_dim
            output, state = self.core.vis(xt, cnn_feats, state)
            output = F.log_softmax(self.logit(output))  # batch * vocab_size
            _, output = torch.max(output, 1)
            it = output
            outputs.append(output)
            attn.append(state[-1])

        return torch.cat([_.unsqueeze(1) for _ in outputs], 1).data, torch.cat([_.unsqueeze(1) for _ in attn], 1).data

    
    def sample(self, cnn_feats, opt={}):
        sample_max = opt.get('sample_max', 0)
        temperature = opt.get('temperature', 1.0)
        batch_size = cnn_feats.size(0)
        state = self.init_hidden(batch_size)

        seq = []
        seqLogprobs = []
        for t in range(self.max_decoder_l + 1):
            if t == 0:  # input <bos>
                it = torch.LongTensor(
                    [self.vocab['<s>']] * batch_size).cuda()
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu() # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False)) # gather the logprobs at sampled positions
                it = it.view(-1).long()

            xt = self.embed(Variable(it, requires_grad=False))

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it)  # seq[t] the input of t+2 time step

                seqLogprobs.append(sampleLogprobs)

            output, state = self.core(xt, cnn_feats, state)
            logprobs = F.log_softmax(self.logit(output))

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)

    def beam(self, context):
        batch_size = context.size(0)
        beam_size = self.beam_size
        beam_context = context.unsqueeze(1).expand(context.size(0), beam_size, context.size(1), context.size(2)).contiguous().view(-1, context.size(1), context.size(2))
        state = self.init_hidden(batch_size)
        seq_len = torch.ones(batch_size, self.beam_size).cuda()
        current_indices_history = []
        beam_parents_history = []
        for t in range(self.max_decoder_l):
            if t == 0:
                beam_input = Variable(torch.LongTensor([self.bos]
                      * batch_size), requires_grad=False).contiguous().cuda()
                xt = self.embed(beam_input)
                out, next_state = self.core(xt, context, state)
            else:
                beam_input = Variable(beam_input, requires_grad=False).contiguous().cuda()
                xt = self.embed(beam_input)
                out, next_state = self.core(xt, beam_context, state)
            probs = F.log_softmax(self.logit(out)).data
            if t == 0:
                beam_score, raw_indices = probs.topk(beam_size, -1)
                current_indices = raw_indices
            else:
                # cond = beam_input.eq(0)+seq_len.eq(1)
                probs.select(1, 0).masked_fill_(beam_input.eq(self.eos).data, 0.0)
                probs.select(1, 0).masked_fill_(beam_input.eq(0).data, 0.0)
                # seq_len.masked_fill_((beam_input.eq(self.eos).data + seq_len.eq(1)).eq(2), t-1)
                # # seq_len.masked_fill_(beam_input.eq(0).data, t-1)
                total_scores = (probs.view(-1, beam_size, self.target_vocab_size) + beam_score.view(batch_size, beam_size, 1).expand(batch_size, beam_size, self.target_vocab_size)).contiguous().view(-1, beam_size * self.target_vocab_size)

                beam_score, raw_indices = total_scores.topk(beam_size, -1)
                current_indices = raw_indices.fmod(self.target_vocab_size)
            beam_parents = raw_indices/self.target_vocab_size
            beam_input = current_indices.view(-1)
            beam_parents_history.append(beam_parents.clone())
            current_indices_history.append(current_indices.clone())
            ## replicate for first timestamp
            if t == 0:
                next_state = beam_replicate(next_state[0], beam_size), beam_replicate(next_state[1], beam_size), beam_replicate(next_state[2], beam_size)
            
            state_h = Variable(next_state[0].data.index_select(
                0, beam_parents.view(-1) + torch.arange(0, batch_size * beam_size, beam_size).long().cuda().contiguous().view(batch_size, 1).expand(batch_size, beam_size).contiguous().view(-1)))
            # state_h = state_h.unsqueeze(0)
            state_c = Variable(next_state[1].data.index_select(
                0, beam_parents.view(-1) + torch.arange(0, batch_size * beam_size, beam_size).long().cuda().contiguous().view(batch_size, 1).expand(batch_size, beam_size).contiguous().view(-1)))
            # state_c = state_c.unsqueeze(0)
            state_o = Variable(next_state[2].data.index_select(
                0, beam_parents.view(-1) + torch.arange(0, batch_size * beam_size, beam_size).long().cuda().contiguous().view(batch_size, 1).expand(batch_size, beam_size).contiguous().view(-1)))
            state = (state_h, state_c, state_o)
            
        scores, indices = torch.max(beam_score, 1)
        scores = scores.view(-1)
        indices = indices.view(-1)
        current_indices = current_indices_history[-1].view(-1).index_select(
            0, indices + torch.arange(0, batch_size * beam_size, beam_size).long().cuda())
        
        results = torch.zeros(batch_size, self.max_decoder_l)
        for t in range(self.max_decoder_l-1 , -1, -1):
            results[:, t].copy_(current_indices)
            indices = beam_parents_history[t].view(-1).index_select(
                0, indices + torch.arange(0, batch_size * beam_size, beam_size).long().cuda())
            if t > 0:
                current_indices = current_indices_history[t - 1].view(-1).index_select(
                    0, indices + torch.arange(0, batch_size * beam_size, beam_size).long().cuda())

        return results

class UI2codeAttention(nn.Module):
    def __init__(self, input_size, context_num_hidden, num_hidden, num_layers=1):
        super(UI2codeAttention, self).__init__()
        """
        input_size: target_embedding_size
        num_hidden: decoder_num_hidden
        """
        self.lstm = nn.LSTMCell(input_size + context_num_hidden, num_hidden)
        self.hidden_mapping = nn.Linear(num_hidden, context_num_hidden, bias=False)
        self.output_mapping = nn.Linear(context_num_hidden+num_hidden, num_hidden, bias=False)
        self.num_layers = num_layers
        # self.input_mapping = nn.Linear(2*num_hidden, num_hidden)
    def forward(self, xt, context, prev_state):
        """
        xt shape: batch * input_size
        context shape: batch * len(feature_map) * decoder_num_hidden
        return:
          context_output: batch_size * decoder_num_hidden
          state: tuple of (ht, ct) each dim - num_layers * batch_size * decoder_num_hidden
        """
        prev_h = prev_state[0]
        prev_c = prev_state[1]
        input = torch.cat([xt, prev_state[2]],-1)
        next_h, next_c = self.lstm(input, (prev_h, prev_c))

            

        top_h= next_h
        mapped_h = self.hidden_mapping(top_h) ## batch * num_hidden
        attn = torch.bmm(context, mapped_h.unsqueeze(2)) ## batch * len(feature) * 1
        attn_weight = F.softmax(attn.squeeze(2)) ## batch * len(feature)
        context_combined = torch.bmm(attn_weight.unsqueeze(1), context).squeeze(1) ## batch * num_hidden
        context_output = F.tanh(self.output_mapping(torch.cat([context_combined, top_h], 1)))
        
        return context_output, (next_h, next_c, context_output)
        
    def vis(self, xt, context, prev_state):
        """
        xt shape: batch * input_size
        context shape: batch * len(feature_map) * decoder_num_hidden
        return:
          context_output: batch_size * decoder_num_hidden
          state: tuple of (ht, ct) each dim - num_layers * batch_size * decoder_num_hidden
        """
        hs = []
        cs = []
        for L in range(self.num_layers):
            prev_h = prev_state[0][L]
            prev_c = prev_state[1][L]
            if L == 0:
                input = torch.cat([xt, prev_state[2]], -1)
                # prev_h = self.input_mapping(torch.cat((prev_h, prev_state[2]), -1))

            else:
                input = hs[-1]
            next_h, next_c = self.lstm_cells[L](input, (prev_h, prev_c))
            hs.append(next_h)
            cs.append(next_c)

        top_h = hs[-1]
        mapped_h = self.hidden_mapping(top_h)  # batch * num_hidden
        attn = torch.bmm(context, mapped_h.unsqueeze(2)
                         )  # batch * len(feature) * 1
        attn_weight = F.softmax(attn.squeeze(2))  # batch * len(feature)
        context_combined = torch.bmm(attn_weight.unsqueeze(
            1), context).squeeze(1)  # batch * num_hidden
        context_output = F.tanh(self.output_mapping(
            torch.cat([context_combined, top_h], 1)))
        return context_output, (torch.stack(hs).contiguous(), torch.stack(cs).contiguous(), context_output, attn_weight)


