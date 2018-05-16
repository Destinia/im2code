import torch
from torch.utils.serialization import load_lua
from models.model import EncoderCNN, EncoderRNN, AttentionDecoder

def load_encoderCNN(opt):
    encoderCNN = EncoderCNN(opt)
    cnn = load_lua('./cnn.t7')
    cnn_weight, _ = cnn.parameters()
    for i, (name, param) in enumerate(encoderCNN.named_parameters()):
        param.data.copy_(cnn_weight[i])
    for i, layer in enumerate(cnn.modules[2:-2]):
        if 'BatchNorm' in str(type(layer)):
            encoderCNN.models[i].running_mean.copy_(layer.running_mean)
            encoderCNN.models[i].running_var.copy_(layer.running_var)

    return encoderCNN

def gate_weight_transform(weight):
    num_hidden = weight.size(-1)
    g1, g2, g3, g4 = weight.chunk(4, 0)
    return torch.cat([g1, g2, g4, g3], 0)

def gate_bias_transform(bias):
    num_hidden = bias.size(-1)
    g1, g2, g3, g4 = bias.chunk(4, 0)
    return torch.cat([g1, g2, g4, g3], 0)

def pos_embedding_transform(weight):
    c, h = weight.chunk(2, 1)
    return torch.cat([h, c], 1)


def load_encoderRNN(opt):
    encoderRNN = EncoderRNN(opt)
    pos_embedding_fw_weight = load_lua('./pos_embedding_fw.t7')
    pos_embedding_bw_weight = load_lua('./pos_embedding_bw.t7')
    encoder_fw_weight = load_lua('./encoder_fw.t7')
    encoder_bw_weight = load_lua('./encoder_bw.t7')
    encoderRNN.pos_embedding_fw.weight.data.copy_(pos_embedding_transform(pos_embedding_fw_weight[0]))
    encoderRNN.pos_embedding_bw.weight.data.copy_(pos_embedding_transform(pos_embedding_bw_weight[0]))
    encoderRNN.lstm.weight_ih_l0.data.copy_(gate_weight_transform(encoder_fw_weight[0]))
    encoderRNN.lstm.bias_ih_l0.data.copy_(gate_bias_transform(encoder_fw_weight[1]))
    encoderRNN.lstm.weight_hh_l0.data.copy_(gate_weight_transform(encoder_fw_weight[2]))
    encoderRNN.lstm.bias_hh_l0.data.copy_(gate_bias_transform(encoder_fw_weight[3]))
    encoderRNN.lstm.weight_ih_l0_reverse.data.copy_(gate_weight_transform(encoder_bw_weight[0]))
    encoderRNN.lstm.bias_ih_l0_reverse.data.copy_(gate_bias_transform(encoder_bw_weight[1]))
    encoderRNN.lstm.weight_hh_l0_reverse.data.copy_(gate_weight_transform(encoder_bw_weight[2]))
    encoderRNN.lstm.bias_hh_l0_reverse.data.copy_(gate_bias_transform(encoder_bw_weight[3]))
    return encoderRNN

def load_decoder(opt):
    decoder = AttentionDecoder(opt)
    decoder_weight = load_lua('./decoder.t7')
    output_projector_weight = load_lua('./output_projector.t7')
    decoder.embed.weight.data.copy_(decoder_weight[0])
    decoder.core.lstm.weight_ih.data.copy_(gate_weight_transform(decoder_weight[1]))
    decoder.core.lstm.bias_ih.data.copy_(gate_bias_transform(decoder_weight[2]))
    decoder.core.lstm.weight_hh.data.copy_(gate_weight_transform(decoder_weight[3]))
    decoder.core.lstm.bias_hh.data.copy_(gate_bias_transform(decoder_weight[4]))
    decoder.core.hidden_mapping.weight.data.copy_(decoder_weight[5])
    decoder.core.output_mapping.weight.data.copy_(decoder_weight[6])
    decoder.logit.weight.data.copy_(output_projector_weight[0])
    decoder.logit.bias.data.copy_(output_projector_weight[1])
    return decoder


