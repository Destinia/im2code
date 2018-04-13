from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        # self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        # self.parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        # self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        # self.parser.add_argument('--lambda_identity', type=float, default=0.5,
        #                          help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss.'
        #                          'For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        # self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        # self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

    # Optimization
        self.parser.add_argument(
            '--norm_grad_clip', type=float, default=5, help='clip normalized gradients at this value')
        self.parser.add_argument('--num_epochs', type=int, default=10, help='The number of whole data passes')
        self.parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
        self.parser.add_argument('--learning_rate', type=float, default=0.1, help='Initial learning rate')
        self.parser.add_argument('--learning_rate_min', type=float, default=0.00001, help='Initial learning rate')
        self.parser.add_argument('--lr_decay', type=float, default=0.5, help='Decay learning rate by this much if (i) perplexity does not decrease on the validation set or (ii) epoch has gone past the start_decay_at_limit')
        self.parser.add_argument('--start_decay_at', type=float, default=999, help='Start decay after this epoch')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--start_from', type=str, default='', help='start from ongoing model')
        self.parser.add_argument('--save_checkpoint_every', type=int, default=10000, help='save checkpoint frequency')

        self.isTrain = True
