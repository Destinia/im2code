import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class EncoderCNN(nn.Module):
    def __init__(self, opt):
        super(EncoderCNN, self).__init__()
        model = [
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
        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x) # batch * 512 * H * W
        output = output.permute([0, 2, 3, 1]) # batch * H * W * 512
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
        self.pos_embedding = nn.Embedding(
            opt.max_encoder_l_h, self.n_layers * self.encoder_num_hidden * 2)
        # self.pos_embedding_bw = nn.Embedding(
        #     opt.max_encoder_l_h, self.n_layers * self.encoder_num_hidden * 2)
        self.lstm_w = nn.LSTM(
            self.input_size, self.encoder_num_hidden, self.n_layers, bidirectional=True)
        self.lstm_h = nn.LSTM(
            self.input_size, self.encoder_num_hidden, self.n_layers, bidirectional=True)
        
    def forward(self, img_feats):
        """
        img_feature shape: batch * H * W * hidden_dim
        """
        assert(len(img_feats) < self.max_encoder_l_h)
        imgH = img_feats.size(1)
        outputs = []
        for i in range(imgH): # imgSeq height
            pos = Variable(torch.LongTensor(
                [i] * img_feats.size(0)), requires_grad=False).cuda().contiguous()  # batch * (num_layer * 2) * hidden_dim
            # (num_layer * 2) * batch * hidden_dim
            pos_embedding = self.pos_embedding(
                pos).view(-1, 2 * self.n_layers, self.encoder_num_hidden).transpose(0, 1).contiguous()
            source = img_feats[:, i].transpose(0, 1) # W * batch * hidden_dim
            output, _ = self.lstm_w(source, (pos_embedding, pos_embedding))
            ## TODO check recoder h or c
            outputs.extend(output)
        return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

class AttentionDecoder(nn.Module):
    def __init__(self, opt):
        super(AttentionDecoder, self).__init__()
        self.decoder_num_hidden = opt.decoder_num_hidden
        self.decoder_num_layers = opt.decoder_num_layers
        self.target_vocab_size = opt.target_vocab_size ## dict size + 4
        self.target_embedding_size = opt.target_embedding_size
        self.max_decoder_l = opt.max_decoder_l
        self.dropout = opt.dropout
        self.embed = nn.Sequential(nn.Embedding(self.target_vocab_size, self.target_embedding_size),
                                   nn.ReLU(),
                                   nn.Dropout(self.dropout))
        self.output_projector = nn.Linear(self.decoder_num_hidden, self.target_vocab_size)
        self.core = UI2codeAttention(self.target_embedding_size, self.decoder_num_hidden, self.decoder_num_layers)
        self.logit = nn.Linear(self.decoder_num_hidden, self.target_vocab_size)


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.decoder_num_layers, bsz, self.decoder_num_hidden).zero_()),
                Variable(weight.new(self.decoder_num_layers, bsz, self.decoder_num_hidden).zero_()))


    def forward(self, cnn_feats, seq):
        """
        - cnn_feats shape: batch_size * (H * W) * (4 * encoder_num_hidden)
        - seq shape: batch_size * #tokens * vocab_size
        """
        batch_size = cnn_feats.size(0)
        state = self.init_hidden(batch_size)

        outputs = []
        for i in range(seq.size(1)):
            it = seq[:, i].clone()

            xt = self.embed(it) # batch * embedding_dim
            output, state = self.core(xt, cnn_feats, state)
            output = F.log_softmax(self.logit(output), dim=1) # batch * vocab_size
            outputs.append(output)

        return torch.cat([_.unsqueeze(1) for _ in outputs], 1)



class UI2codeAttention(nn.Module):
    def __init__(self, input_size ,num_hidden, num_layers=1):
        super(UI2codeAttention, self).__init__()
        """
        input_size: target_embedding_size
        num_hidden: decoder_num_hidden
        """
        self.lstm_cells = nn.ModuleList([nn.LSTMCell(input_size, num_hidden) for _ in range(num_layers)])
        self.hidden_mapping = nn.Linear(num_hidden, num_hidden, bias=False)
        self.output_mapping = nn.Linear(2*num_hidden, num_hidden, bias=False)
        self.num_layers = num_layers
    def forward(self, xt, context, prev_state):
        """
        xt shape: batch * input_size
        context shape: batch * len(feature_map) * decoder_num_hidden
        """
        hs = []
        cs = []
        for L in range(self.num_layers):
            prev_h = prev_state[0][L]
            prev_c = prev_state[1][L]
            if L == 0:
                input = xt
            else:
                input = hs[-1]
            next_h, next_c = self.lstm_cells[L](input, (prev_h, prev_c))
            hs.append(next_h)
            cs.append(next_c)
            

        top_h= hs[-1]
        mapped_h = self.hidden_mapping(top_h) ## batch * num_hidden
        attn = torch.bmm(context, mapped_h.unsqueeze(2)) ## batch * len(feature) * 1
        attn_weight = F.softmax(attn.squeeze(), dim=1) ## batch * len(feature)
        context_combined = torch.bmm(attn_weight.unsqueeze(1), context).squeeze() ## batch * num_hidden
        context_output = self.output_mapping(torch.cat([context_combined, top_h], 1))
        return context_output, (hs, cs)

        
        

# class UI2codeModel(nn.Module):
#     def __init__(self, opt):
#         self.encoderCNN = EncoderCNN(opt)
#         self.encoderRNN = EncoderRNN(opt)
#         self.decoder = 

