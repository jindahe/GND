import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value

    lowered = value.lower()
    if lowered in {"true", "t", "1", "yes", "y"}:
        return True
    if lowered in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


parser = argparse.ArgumentParser()

par_common = parser.add_argument_group('common parameters')
'''para of code'''
par_common.add_argument('-c_type', type=str, default='sur',
        help='the code type of the original code, one of the labels of code, default: %(default)s')
par_common.add_argument('-n', type=int, default=72,
        help='the number of qubits, one of the labels of code')
par_common.add_argument('-d', type=int, default=5,
        help='the distance of the original code, one of the labels of code')
par_common.add_argument('-k', type=int, default=1,
        help='the number of logical qubits of the code, one of the labels of code, default: %(default)d')
par_common.add_argument('-seed', type=int, default=0,
        help='seed of random removal of stabilizers from the original code, one of the labels of code, default: %(default)d')
'''para of errors'''
par_common.add_argument('-e_model', type=str, default='dep',
        help='error model, default: %(default)s')
par_common.add_argument('-error_seed', type=int, default=51697,
        help='seed of generate errors, default: %(default)d')
par_common.add_argument('-trials', type=int, default=10000,
        help='trials of decoding, default: %(default)d')
par_common.add_argument('-er', type=float, default=0.189,
        help='the error rate for inference, default: %(default)s')
par_common.add_argument('-sweep', type=str2bool, nargs='?', const=True, default=False,
        help='if true, evaluate a log-spaced error-rate sweep; otherwise use -er as a single point, default: %(default)s')
par_common.add_argument('-er_min', type=float, default=None,
        help='lower bound of the error-rate sweep, default: %(default)s')
par_common.add_argument('-er_max', type=float, default=None,
        help='upper bound of the error-rate sweep, default: %(default)s')
par_common.add_argument('-n_points', type=int, default=10,
        help='number of points in an error-rate sweep, default: %(default)d')
par_common.add_argument('-chunk_size', type=int, default=1000,
        help='chunk size used by chunked evaluation scripts, default: %(default)d')
par_common.add_argument('-n_train', type=int, default=10000,
        help='number of training samples for dataset-generation scripts, default: %(default)d')
par_common.add_argument('-n_val', type=int, default=2000,
        help='number of validation samples for dataset-generation scripts, default: %(default)d')
par_common.add_argument('-n_test', type=int, default=2000,
        help='number of test samples for dataset-generation scripts, default: %(default)d')
par_common.add_argument('-shuffle', type=str2bool, nargs='?', const=True, default=True,
        help='if true, shuffle samples before splitting datasets, default: %(default)s')
par_common.add_argument('-split_seed', type=int, default=0,
        help='shuffle seed used by dataset-generation scripts, default: %(default)d')
par_common.add_argument('-train_seed', type=int, default=0,
        help='random seed used for model initialization and training-order stochasticity, default: %(default)d')
par_common.add_argument('-partition_axis', type=str, default='x', choices=['x', 'y'],
        help='spatial axis used for toric syndrome cuts, default: %(default)s')
par_common.add_argument('-cut', type=int, default=None,
        help='cut position used for toric syndrome cuts, default: %(default)s')
par_common.add_argument('-partition_order', type=str, default='none', choices=['none', 'AB', 'BA'],
        help='column order for syndrome-only datasets, default: %(default)s')
par_common.add_argument('-dataset_dir', type=str, default='',
        help='optional output directory for saved datasets, default: %(default)s')
par_common.add_argument('-dataset_path', type=str, default='',
        help='optional explicit path to a syndrome-only dataset, default: %(default)s')
par_common.add_argument('-save_dir', type=str, default='',
        help='optional output directory for saved model checkpoints, default: %(default)s')
par_common.add_argument('-record_dir', type=str, default='',
        help='optional output directory for structured JSON experiment records, default: %(default)s')
par_common.add_argument('-ab_checkpoint', type=str, default='',
        help='optional explicit checkpoint path for the AB-ordered syndrome model, default: %(default)s')
par_common.add_argument('-ba_checkpoint', type=str, default='',
        help='optional explicit checkpoint path for the BA-ordered syndrome model, default: %(default)s')
par_common.add_argument('-mi_samples', type=int, default=10000,
        help='number of Monte Carlo samples used by bipartite-MI evaluation, default: %(default)d')
par_common.add_argument('-eval_seed', type=int, default=0,
        help='random seed used by Monte Carlo sampling during bipartite-MI evaluation, default: %(default)d')
par_common.add_argument('-bootstrap_samples', type=int, default=0,
        help='number of bootstrap resamples used by bipartite-MI evaluation, default: %(default)d')
par_common.add_argument('-bootstrap_seed', type=int, default=0,
        help='bootstrap RNG seed used by bipartite-MI evaluation, default: %(default)d')
par_common.add_argument('-mi_output_path', type=str, default='',
        help='optional path for saving bipartite-MI evaluation results, default: %(default)s')


'''para of made'''
par_common.add_argument('-depth', type=int, default=0,
        help='depth of MADE, default: %(default)d')
par_common.add_argument('-width', type=int, default=1,
        help='width of MADE, default: %(default)d')
par_common.add_argument('-made_activation', type=str, default='tanh',
        choices=['tanh', 'relu', 'sigmoid'],
        help='activation used by MADE hidden layers, default: %(default)s')
par_common.add_argument('-made_residual', type=str2bool, nargs='?', const=True, default=False,
        help='if true, enable MADE residual wrapper, default: %(default)s')
par_common.add_argument('-made_max_params', type=int, default=0,
        help='optional soft cap for MADE parameter count; when positive, training shrinks width to stay within budget, default: %(default)d')

'''para of trade'''
par_common.add_argument('-d_model', type=int, default=256,
        help='d_model of trade, default: %(default)d')
par_common.add_argument('-n_heads', type=int, default=4,
        help='number of heads, default: %(default)d')
par_common.add_argument('-d_ff', type=int, default=256,
        help='dim of forward, default: %(default)d')
par_common.add_argument('-n_layers', type=int, default=1,
        help='number of layers, default: %(default)d')
par_common.add_argument('-hidden_dim', type=int, default=512,
        help='hidden dimension used by NADE training, default: %(default)d')
'''para for training'''
par_common.add_argument('-n_type', type=str, default='made', choices=['made', 'trade', 'nade'],
        help='net type of training , default: %(default)s')

par_common.add_argument('-dtype', type=str, default='float32',
        choices=['float32', 'float64'],
        help='dtypes used during training, default: %(default)s')
par_common.add_argument('-device', type=str, default='cuda:0',
        help='device used during training, default: %(default)s')
par_common.add_argument('-epoch', type=int, default=10000,
        help='epoch of training, default: %(default)d')
par_common.add_argument('-batch', type=int, default=10000,
        help='batch of training, default: %(default)d')
par_common.add_argument('-lr', type=float, default=0.001,
        help='learning rate, default: %(default)s')
par_common.add_argument('-weight_decay', type=float, default=0.0,
        help='weight decay used by Adam/AdamW optimizers, default: %(default)s')
par_common.add_argument('-lr_decay_factor', type=float, default=0.5,
        help='factor used by validation-plateau LR decay, default: %(default)s')
par_common.add_argument('-lr_decay_patience', type=int, default=5,
        help='number of unimproved validation epochs before LR decay, default: %(default)d')
par_common.add_argument('-min_lr', type=float, default=0.0002,
        help='minimum learning rate used by validation-plateau scheduler, default: %(default)s')
par_common.add_argument('-cpe', type=int, default=10000,
        help='correction per cpe epoch, default: %(default)d')
par_common.add_argument('-log_every', type=int, default=10,
        help='print training metrics every this many epochs, default: %(default)d')
par_common.add_argument('-early_stop_patience', type=int, default=20,
        help='stop training after this many unimproved validation epochs; set <=0 to disable, default: %(default)d')
par_common.add_argument('-early_stop_min_delta', type=float, default=0.0,
        help='minimum validation-NLL improvement required to reset early stopping, default: %(default)s')
par_common.add_argument('-save', type=str2bool, nargs='?', const=True, default=False,
        help='save the results if true, default: %(default)s')
        

args = parser.parse_args()

if __name__ == '__main__':
    print(args)
