import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from .BeamSearch import Beam
from util.util import Sequence, TopN, beam_replicate
import random

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1)

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
            self.input_size, self.encoder_num_hidden, self.n_layers, bidirectional=True, bias=False, batch_first=True)

    def forward(self, img_feats):
        """
        img_feature shape: batch * H * W * hidden_dim
        """
        assert(len(img_feats) < self.max_encoder_l_h)
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
        self.pos_embedding_w = nn.Embedding(
            opt.max_encoder_l_h, self.n_layers * self.encoder_num_hidden * 2)
        self.pos_embedding_h = nn.Embedding(
            opt.max_encoder_l_w, self.n_layers * self.encoder_num_hidden * 2)
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
        imgW = img_feats.size(2)
        outputs_w = []
        outputs_h = []
        for i in range(imgH): # imgSeq height
            pos = Variable(torch.LongTensor(
                [i] * img_feats.size(0)).zero_(), requires_grad=False).cuda().contiguous()  # batch * (num_layer * 2) * hidden_dim
            # (num_layer * 2) * batch * hidden_dim
            pos_embedding = self.pos_embedding_w(
                pos).view(-1, 2 * self.n_layers, self.encoder_num_hidden).transpose(0, 1).contiguous()
            source = img_feats[:, i].transpose(0, 1) # W * batch * hidden_dim
            output, _ = self.lstm_w(source, (pos_embedding, pos_embedding))
            outputs_w.extend(output)
        for i in range(imgW): # imgSeq width
            pos = Variable(torch.LongTensor(
                [i] * img_feats.size(0)).zero_(), requires_grad=False).cuda().contiguous()  # batch * (num_layer * 2) * hidden_dim
            # (num_layer * 2) * batch * hidden_dim
            pos_embedding = self.pos_embedding_h(
                pos).view(-1, 2 * self.n_layers, self.encoder_num_hidden).transpose(0, 1).contiguous()
            source = img_feats[:, :, i].transpose(0, 1) # H * batch * hidden_dim
            output, _ = self.lstm_h(source, (pos_embedding, pos_embedding))
            outputs_h.append(output) # encoder_l * batch * num_hudden
        outputs_w_t = torch.cat([_.unsqueeze(1) for _ in outputs_w], 1)
        outputs_h_t = torch.stack(outputs_h, 1).view(
            imgH * imgW, -1, self.encoder_num_hidden * 2 * self.n_layers).transpose(0, 1)
        return torch.cat((outputs_w_t, outputs_h_t), -1)

class AttentionDecoder(nn.Module):
    def __init__(self, opt):
        super(AttentionDecoder, self).__init__()
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
        self.core = UI2codeAttention(self.target_embedding_size, self.decoder_num_hidden, self.decoder_num_layers)
        self.logit = nn.Linear(self.decoder_num_hidden, self.target_vocab_size)
        self.ss_prob = 0.0


    def init_hidden(self, bsz, batch_first=False):
        weight = next(self.parameters()).data
        if batch_first:
            return (Variable(weight.new(bsz, self.decoder_num_layers, self.decoder_num_hidden).zero_()),
                    Variable(weight.new(bsz, self.decoder_num_layers, self.decoder_num_hidden).zero_()))
        return (Variable(weight.new(bsz, self.decoder_num_hidden).zero_()),
                Variable(weight.new(bsz, self.decoder_num_hidden).zero_()),
                Variable(weight.new(bsz, self.decoder_num_hidden).zero_()))

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

    
    def sample_max(self, cnn_feats, opt={}):
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

                seqLogprobs.append(sampleLogprobs.view(-1))

            output, state = self.core(xt, cnn_feats, state)
            logprobs = F.log_softmax(self.logit(output))

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)
    def decode_beam(self, context):
        """Decode a minibatch."""

        beam_size = self.beam_size
        batch_size = context.size(0)
        state_h, state_c = self.init_hidden(batch_size)

        #  (1) run the encoder on the src

        # context = context.transpose(0, 1)  # Make things sequence first.

        # Expand tensors for each beam.
        context = Variable(context.data.repeat(beam_size, 1, 1))
        dec_states = (
            Variable(state_h.data.repeat(1, beam_size, 1)),
            Variable(state_c.data.repeat(1, beam_size, 1))
        )

        beam = [
            Beam(beam_size, self.vocab, cuda=True)
            for k in range(batch_size)
        ]

        batch_idx = list(range(batch_size))
        remaining_sents = batch_size

        for i in range(self.max_decoder_l):
            # input (batch * beam)
            input = torch.stack(
                                [b.get_current_state()
                                 for b in beam if not b.done]
                                ).t().contiguous().view(1, -1)

            trg_emb = self.embed(Variable(input).transpose(1, 0))
            trg_h, (trg_h_t, trg_c_t) = self.core(
                trg_emb,
                context,
                (dec_states[0], dec_states[1]),
            )

            dec_states = (trg_h_t, trg_c_t)

            dec_out = trg_h.squeeze(1)
            out = F.softmax(self.logit(dec_out)).unsqueeze(0)

            word_lk = out.view(
                beam_size,
                remaining_sents,
                -1
            ).transpose(0, 1).contiguous()

            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx]):
                    active += [b]

                for dec_state in dec_states:  # iterate over h, c
                    # layers x beam*sent x dim
                    sent_states = dec_state.view(
                        -1, beam_size, remaining_sents, dec_state.size(2)
                    )[:, :, idx]
                    sent_states.data.copy_(
                        sent_states.data.index_select(
                            1,
                            beam[b].get_current_origin()
                        )
                    )

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_idx = torch.cuda.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active(t):
                # select only the remaining active sentences
                view = t.data.view(
                    -1, remaining_sents,
                    self.decoder_num_hidden
                )
                new_size = list(t.size())
                new_size[-2] = new_size[-2] * len(active_idx) \
                    // remaining_sents
                return Variable(view.index_select(
                    1, active_idx
                ).view(*new_size))

            dec_states = (
                update_active(dec_states[0]),
                update_active(dec_states[1])
            )
            dec_out = update_active(dec_out)
            context = update_active(context.transpose(0, 1).contiguous()).transpose(0, 1)

            remaining_sents = len(active)

        #  (4) package everything up

        allHyp, allScores = [], []
        n_best = 1

        for b in range(batch_size):
            scores, ks = beam[b].sort_best()

            allScores += [scores[:n_best]]
            hyps = [beam[b].get_hyp(k) for k in ks[:n_best]][0]
            allHyp += [hyps]

        return allHyp, allScores

    def generate(self, input, state, context, k=1):
        """Run decoder with attention
        Args:
          input: RNN cell input i.e. previous timestamp output (batch_size * 1)
          context: image feature encoder sequence (num_encoder_l * num_encoder_dim)
          state (optional): previous timestamp state (batch * num_decoder_dim, batch * num_decoder_dim)
        Returns:
          words: predict topK id
          logprobs: probabilities of words
          new_states: next state list with len=batch_size element_dim (num_layers * 1 * num_hidden)
        """
        it = Variable(torch.LongTensor(input), volatile=True).cuda()
        xt = self.embed(it)
        logits, new_states = self.core(xt, context, state, batch_first=True)
        logprobs = F.log_softmax(self.logit(logits))
        logprobs, words = logprobs.data.topk(k, 1)
        # print('check 1:', time.time() - start_time)
        # print('check 2:', time.time() - start_time)
        return words, logprobs, new_states

    # def beam_search(self, context):
        """Runs beam search sequence generation on a single image.
        Args:
          initial_input: An initial input for the model -
                         list of batch size holding the first input for every entry.
          initial_state (optional): An initial state for the model -
                         list of batch size holding the current state for every entry.
        Returns:
          A list of batch size, each the most likely sequence from the possible beam_size candidates.
        """
        batch_size = context.size(0)
        partial_sequences = [TopN(self.beam_size) for _ in range(batch_size)]
        complete_sequences = [TopN(self.beam_size) for _ in range(batch_size)]

        initial_input = [self.bos] * batch_size
        initial_state = self.init_hidden(batch_size, batch_first=True)

        words, logprobs, (new_state_h, new_state_c) = self.generate(
            initial_input, initial_state, context,
            k=self.beam_size)
        for b in range(batch_size):
            # Create first beam_size candidate hypotheses for each entry in
            # batch
            for k in range(self.beam_size):
                seq = Sequence(
                    sentence=[words[b][k]],
                    state=(new_state_h[b], new_state_c[b]),
                    logprob=logprobs[b][k],
                    score=logprobs[b][k],
                    context=context[b])
                partial_sequences[b].push(seq)

        # Run beam search.
        for _ in range(self.max_decoder_l - 1):
            partial_sequences_list = [p.extract() for p in partial_sequences]
            for p in partial_sequences:
                p.reset()

            # Keep a flattened list of parial hypotheses, to easily feed
            # through a model as whole batch
            flattened_partial = [
                s for sub_partial in partial_sequences_list for s in sub_partial]

            input_feed = [c.sentence[-1] for c in flattened_partial]
            state_h, state_c = zip(*[c.state for c in flattened_partial])
            state_feed = torch.stack(list(state_h)), torch.stack(list(state_c))
            context_feed = torch.stack([c.context for c in flattened_partial])
            if len(input_feed) == 0:
                # We have run out of partial candidates; happens when
                # beam_size=1
                break

            # Feed current hypotheses through the model, and recieve new outputs and states
            # logprobs are needed to rank hypotheses
            words, logprobs, (new_states_h, new_state_c) \
                = self.generate(
                    input_feed, state_feed, context_feed,
                    k=self.beam_size + 1)
            # print('time2: ', time.time() - start_time)
            idx = 0
            for b in range(batch_size):
                # For every entry in batch, find and trim to the most likely
                # beam_size hypotheses
                for partial in partial_sequences_list[b]:
                    state = (new_states_h[idx], new_state_c[idx])
                    k = 0
                    num_hyp = 0
                    while num_hyp < self.beam_size:
                        w = words[idx][k]
                        sentence = partial.sentence + [w]
                        logprob = partial.logprob + logprobs[idx][k]
                        score = logprob
                        k += 1
                        num_hyp += 1

                        if w == self.eos:
                            ## for NMT normalize length
                            # if self.length_normalization_factor > 0:
                            #     L = self.length_normalization_const
                            #     length_penalty = (L + len(sentence)) / (L + 1)
                            #     score /= length_penalty ** self.length_normalization_factor
                            beam = Sequence(sentence, state,
                                            logprob, score, context[b])
                            complete_sequences[b].push(beam)
                            num_hyp -= 1  # we can fit another hypotheses as this one is over
                        else:
                            beam = Sequence(sentence, state,
                                            logprob, score, context[b])
                            partial_sequences[b].push(beam)
                    idx += 1

        # If we have no complete sequences then fall back to the partial sequences.
        # But never output a mixture of complete and partial sequences because a
        # partial sequence could have a higher score than all the complete
        # sequences.
        for b in range(batch_size):
            if not complete_sequences[b].size():
                complete_sequences[b] = partial_sequences[b]
        seqs = [complete.extract(sort=True)[0]
                for complete in complete_sequences]
        return np.array([s.sentence for s in seqs])

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
                total_scores = (probs.view(-1, beam_size, self.target_vocab_size) + beam_score.view(-1, beam_size, 1).expand(batch_size, beam_size, self.target_vocab_size)).contiguous().view(-1, beam_size * self.target_vocab_size)

                beam_score, raw_indices = total_scores.topk(beam_size, -1)
                current_indices = raw_indices.fmod(self.target_vocab_size)
            beam_parents = raw_indices/self.target_vocab_size
            # if t < 10:
            #     print(t, beam_score, beam_parents)
            beam_input = current_indices.view(-1)
            beam_parents_history.append(beam_parents.clone())
            current_indices_history.append(current_indices.clone())
            state_h = Variable(beam_replicate(next_state[0].squeeze(0).data, self.beam_size).index_select(
                0, beam_parents.view(-1) + torch.arange(0, batch_size * beam_size, beam_size).long().cuda().contiguous().view(batch_size, 1).expand(batch_size, beam_size).contiguous().view(-1)))
            # state_h = state_h.unsqueeze(0)
            state_c = Variable(beam_replicate(next_state[1].squeeze(0).data, self.beam_size).index_select(
                0, beam_parents.view(-1) + torch.arange(0, batch_size * beam_size, beam_size).long().cuda().contiguous().view(batch_size, 1).expand(batch_size, beam_size).contiguous().view(-1)))
            # state_c = state_c.unsqueeze(0)
            state_o = Variable(beam_replicate(next_state[2].data, self.beam_size).index_select(
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

    def get_logprobs_state(self, it, context, state):
        # 'it' is Variable contraining a word index
        xt = self.embed(it)

        output, state = self.core(
            xt, context, state)
        logprobs = F.log_softmax(self.logit(output))

        return logprobs, state

    def beam_search(self, state, logprobs, *args, **kwargs):
        # args are the miscelleous inputs to the core in addition to embedded word and state
        # kwargs only accept opt

        def beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            #INPUTS:
            #logprobsf: probabilities augmented after diversity
            #beam_size: obvious
            #t        : time instant
            #beam_seq : tensor contanining the beams
            #beam_seq_logprobs: tensor contanining the beam logprobs
            #beam_logprobs_sum: tensor contanining joint logprobs
            #OUPUTS:
            #beam_seq : tensor containing the word indices of the decoded captions
            #beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            #beam_logprobs_sum : joint log-probability of each beam

            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols):  # for each column (word, essentially)
                for q in range(rows):  # for each beam expansion
                    # compute logprob of expanding beam q with word in (sorted)
                    # position c
                    local_logprob = ys[q, c]
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    candidates.append(
                        {'c': ix[q, c], 'q': q, 'p': candidate_logprob, 'r': local_logprob})
            candidates = sorted(candidates,  key=lambda x: -x['p'])

            new_state = [_.clone() for _ in state]
            #beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
                #we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                #fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t,
                                      vix] = beam_seq_logprobs_prev[:, v['q']]
                #rearrange recurrent states
                for state_ix in range(len(new_state)):
                    #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:,
                                                                  v['q']]  # dimension one is time step
                #append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c']  # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r']  # the raw logprob here
                # the new (sum) logprob along this beam
                beam_logprobs_sum[vix] = v['p']
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates

        # start beam search
        opt = kwargs['opt']
        beam_size = opt.get('beam_size', 5)

        beam_seq = torch.LongTensor(self.max_decoder_l, beam_size).zero_()
        beam_seq_logprobs = torch.FloatTensor(
            self.max_decoder_l, beam_size).zero_()
        # running sum of logprobs for each beam
        beam_logprobs_sum = torch.zeros(beam_size)
        done_beams = []

        for t in range(self.max_decoder_l):
            """pem a beam merge. that is,
            for every previous beam we now many new possibilities to branch out
            we need to resort our beams to maintain the loop invariant of keeping
            the top beam_size most likely sequences."""
            logprobsf = logprobs.data.float(
            )  # lets go to CPU for more efficiency in indexing operations
            # suppress UNK tokens in the decoding
            logprobsf[:, logprobsf.size(
                1) - 1] = logprobsf[:, logprobsf.size(1) - 1] - 1000

            beam_seq,\
                beam_seq_logprobs,\
                beam_logprobs_sum,\
                state,\
                candidates_divm = beam_step(logprobsf,
                                            beam_size,
                                            t,
                                            beam_seq,
                                            beam_seq_logprobs,
                                            beam_logprobs_sum,
                                            state)

            for vix in range(beam_size):
                # if time's up... or if end token is reached then copy beams
                if beam_seq[t, vix] == 0 or beam_seq[t, vix] == self.eos or t == self.max_decoder_l - 1:
                    final_beam = {
                        'seq': beam_seq[:, vix].clone(),
                        'logps': beam_seq_logprobs[:, vix].clone(),
                        'p': beam_logprobs_sum[vix]
                    }
                    done_beams.append(final_beam)
                    # don't continue beams from finished sequences
                    beam_logprobs_sum[vix] = -1000

            # encode as vectors
            it = beam_seq[t]
            logprobs, state = self.get_logprobs_state(
                Variable(it.cuda()), *(args + (state,)))

        done_beams = sorted(done_beams, key=lambda x: -x['p'])[:beam_size]
        return done_beams

    def sample_beam(self, context, opt={}):
        batch_size = context.size(0)
        beam_size = self.beam_size

        # Project the attention feats first to reduce memory and computation
        # comsumptions.

        assert beam_size <= self.target_vocab_size + \
            1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.max_decoder_l, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.max_decoder_l, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = context[k:k +1].expand(beam_size, context.size(1), context.size(2))

            for t in range(1):
                if t == 0:  # input <bos>
                    it = context.data.new(beam_size).long().zero_().fill_(self.bos)
                    xt = self.embed(Variable(it, requires_grad=False))

                output, state = self.core(xt, tmp_fc_feats, state)
                logprobs = F.log_softmax(self.logit(output))

            self.done_beams[k] = self.beam_search(
                state, logprobs, tmp_fc_feats, opt={'beam_size': beam_size})
            # the first beam has highest cumulative score
            seq[:, k] = self.done_beams[k][0]['seq']
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        # return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)
        return seq.transpose(0, 1)


class UI2codeAttention(nn.Module):
    def __init__(self, input_size ,num_hidden, num_layers=1):
        super(UI2codeAttention, self).__init__()
        """
        input_size: target_embedding_size
        num_hidden: decoder_num_hidden
        """
        self.lstm = nn.LSTMCell(input_size+num_hidden, num_hidden)
        self.hidden_mapping = nn.Linear(num_hidden, num_hidden, bias=False)
        self.output_mapping = nn.Linear(2*num_hidden, num_hidden, bias=False)
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

        
        

# class UI2codeModel(nn.Module):
#     def __init__(self, opt):
#         self.encoderCNN = EncoderCNN(opt)
#         self.encoderRNN = EncoderRNN(opt)
#         self.decoder = 

