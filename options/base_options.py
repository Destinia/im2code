import argparse
import os
import math
from util import util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--data_root', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--model', type=str, default='translation',
                                 help='chooses which model to use. cycle_gan, pix2pix, test')
        self.parser.add_argument('--spatial', action='store_true', help='encoder use spatial LSTM')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--checkpoint_path', type=str, default='save',
                                help='directory to store checkpointed models')
        
        # UI2code
        self.parser.add_argument('--max_image_width', type=int, default=300, help='Maximum length of input feature sequence along width direction') #800/2/2/2
        self.parser.add_argument('--max_image_height', type=int, default=200, help='Maximum length of input feature sequence along width direction') #80 / (2*2*2)
        self.parser.add_argument('--max_num_tokens', type=int, default=150, help='Maximum number of output tokens') # when evaluate, this is the cut-off length.
        self.parser.add_argument('--target_embedding_size', type=int, default=80, help='Token embedding size') # when evaluate, this is the cut-off length.
        self.parser.add_argument('--cnn_feature_size', type=int, default=512, help='CNN embedded feature length') # when evaluate, this is the cut-off length.
        self.parser.add_argument('--encoder_num_hidden', type=int, default=256, help='Number of hidden units in encoder cell')
        self.parser.add_argument('--encoder_num_layers', type=int, default=1, help='Number of hidden layers in encoder cell')
        self.parser.add_argument('--decoder_num_layers', type=int, default=1, help='Number of hidden units in decoder cell')
        self.parser.add_argument('--dropout', type=int, default=0.0, help='Dropout probability') # does support dropout now!!!
        self.parser.add_argument('--train_data_path', type=str, default='data/train.lst', help='The path containing data file names and labels. Format per line: image_path characters')
        self.parser.add_argument('--val_data_path', type=str, default='data/validate.lst', help='The path containing validate data file names and labels. Format per line: image_path characters')
        self.parser.add_argument('--test_data_path', type=str, default='data/test_shuffle.lst', help='The path containing validate data file names and labels. Format per line: image_path characters')
        self.parser.add_argument('--label_path', type=str, default='data/XMLsequence.lst', help='The path containing data file names and labels. Format per line: image_path characters')
        self.parser.add_argument('--vocab_path', type=str, default='data/xml_vocab.txt', help='Vocabulary file. A token per line.')
        self.parser.add_argument('--model_dir', type=str, default='model', help='The directory for saving and loading model parameters (structure is not stored)')
        self.parser.add_argument('--beam_size', type=int, default=5, help='beam search size when validation')
        self.initialized = True



    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test
        self.opt.max_encoder_l_w = math.floor(self.opt.max_image_width / 8.0)
        self.opt.max_encoder_l_h = math.floor(self.opt.max_image_height / 8.0)
        self.opt.decoder_num_hidden = 2 * self.opt.encoder_num_hidden
        if self.opt.spatial:
            # double size decoder num_hidden
            self.opt.decoder_num_hidden = self.opt.decoder_num_hidden * 2
        self.opt.max_decoder_l = self.opt.max_num_tokens + 1
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        # setup vocab
        self.opt.vocab = util.get_vocab(self.opt)
        self.opt.rev_vocab = [t[0]
                              for t in list(sorted(self.opt.vocab.items(), key=lambda x: x[1]))]
        self.opt.eos = self.opt.vocab['</s>']
        self.opt.opening_tag = '{'
        self.opt.closing_tag = '}'
        # save to the disk
        self.opt.expr_dir = os.path.join(self.opt.checkpoint_path, self.opt.name)
        util.mkdirs(self.opt.expr_dir)
        file_name = os.path.join(self.opt.expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
