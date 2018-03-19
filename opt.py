import argparse


def parse_opt():
    parser = argparse.ArgumentParser()


    parser.add_argument('--data_base_dir', type=, default='/n/rush_lab/data/image_data/formula_images_crop_pad_down', help='The base directory of the image path in data-path. If the image path in data-path is absolute path, set it to /')
    parser.add_argument('--data_path', type=, default='/n/rush_lab/data/image_data/im2latex_train_large_filter.lst', help='The path containing data file names and labels. Format per line: image_path characters')
    parser.add_argument('--label_path', type=, default='/n/rush_lab/data/image_data/im2latex_formulas.norm4.final.lst', help='The path containing data file names and labels. Format per line: image_path characters')
    parser.add_argument('--val_data_path', type=, default='/n/rush_lab/data/image_data/im2latex_validate_large_filter.lst', help='The path containing validate data file names and labels. Format per line: image_path characters')
    parser.add_argument('--model_dir', type=, default='model', help='The directory for saving and loading model parameters (structure is not stored)')
    parser.add_argument('--log_path', type=, default='log.txt', help='The path to put log')
    parser.add_argument('--output_dir', type=, default='results', help='The path to put visualization results if visualize is set to True')

    # Display
    parser.add_argument('--steps_per_checkpoint', type=, default=100, help='Checkpointing (print perplexity, save model) per how many steps')
    parser.add_argument('--num_batches_val', type=, default=math.huge, help='Number of batches to evaluate.')
    parser.add_argument('--beam_size', type=, default=1, help='Beam size.')
    parser.add_argument('--use_dictionary', type=, default=false, help='Use dictionary during decoding or not.')
    parser.add_argument('--allow_digit_prefix', type=, default=false, help='During decoding, allow arbitary digits before word.')
    parser.add_argument('--dictionary_path', type=, default='/n/rush_lab/data/image_data/train_dictionary.txt', help='The path containing dictionary. Format per line: word')

    # Optimization
    parser.add_argument('--num_epochs', type=, default=15, help='The number of whole data passes')
    parser.add_argument('--batch_size', type=, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=, default=0.1, help='Initial learning rate')
    parser.add_argument('--learning_rate_min', type=, default=0.00001, help='Initial learning rate')
    parser.add_argument('--lr_decay', type=, default=0.5, help='Decay learning rate by this much if (i) perplexity does not decrease on the validation set or (ii) epoch has gone past the start_decay_at_limit')
    parser.add_argument('--start_decay_at', type=, default=999, help='Start decay after this epoch')

    # Network
    parser.add_argument('--dropout', type=, default=0.0, help='Dropout probability') # does support dropout now!!!
    parser.add_argument('--target_embedding_size', type=, default=80, help='Embedding dimension for each target')
    parser.add_argument('--input_feed', type=, default=false, help='Whether or not use LSTM attention decoder cell')
    parser.add_argument('--encoder_num_hidden', type=, default=256, help='Number of hidden units in encoder cell')
    parser.add_argument('--encoder_num_layers', type=, default=1, help='Number of hidden layers in encoder cell') # does not support >1 now!!!
    parser.add_argument('--decoder_num_layers', type=, default=1, help='Number of hidden units in decoder cell')
    parser.add_argument('--vocab_file', type=, default='', help='Vocabulary file. A token per line.')

    # Other
    parser.add_argument('--phase', type=, default='test', help='train or test')
    parser.add_argument('--gpu_id', type=, default=1, help='Which gpu to use. <=0 means use CPU')
    parser.add_argument('--load_model', type=, default=false, help='Load model from model-dir or not')
    parser.add_argument('--visualize', type=, default=false, help='Print results or not')
    parser.add_argument('--seed', type=, default=910820, help='Load model from model-dir or not')
    parser.add_argument('--max_num_tokens', type=, default=150, help='Maximum number of output tokens') # when evaluate, this is the cut-off length.
    parser.add_argument('--max_image_width', type=int, default=300, help='Maximum length of input feature sequence along width direction') #800/2/2/2
    parser.add_argument('--max_image_height', type=int, default=200, help='Maximum length of input feature sequence along width direction') #80 / (2*2*2)
    parser.add_argument('--prealloc', type=, default=false, help='Use memory preallocation and sharing between cloned encoder/decoders')


    args = parser.parse_args()
    return args
