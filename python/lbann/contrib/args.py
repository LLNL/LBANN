"""Helper functions to add common command-line arguments."""

from typing import Any

import argparse

import lbann
import lbann.core.optimizer


def add_scheduler_arguments(parser):
    """Add command-line arguments for common scheduler settings.

    Adds the following options: `--nodes`, `--procs-per-node`,
    `--partition`, `--account`, `--time-limit`, `--reservation`.
    The caller is responsible for using them.
    `get_scheduler_kwargs` can assist with extracting them.

    Args:
        parser (argparse.ArgumentParser): command-line argument
            parser.

    """
    if not isinstance(parser, argparse.ArgumentParser):
        raise TypeError('expected an argparse.ArgumentParser')
    parser.add_argument(
        '--nodes', action='store', type=int,
        help='number of compute nodes', metavar='NUM')
    parser.add_argument(
        '--procs-per-node', action='store', type=int,
        help='number of processes per compute node', metavar='NUM')
    parser.add_argument(
        '--partition', action='store', type=str,
        help='scheduler partition', metavar='NAME')
    parser.add_argument(
        '--account', action='store', type=str,
        help='scheduler account', metavar='NAME')
    parser.add_argument(
        '--reservation', action='store', type=str,
        help='scheduler reservation', metavar='NAME')
    parser.add_argument(
        '--time-limit', action='store', type=int,
        help='time limit (in minutes)', metavar='MIN')
    parser.add_argument(
        '--setup-only', action='store_true',
        help='set up (but do not run) the experiment')

def get_scheduler_kwargs(args):
    """Generate keyword arguments for a scheduler.

    The parsed arguments must be generated by an
    `argparse.ArgumentParser` that has been processed by
    `add_scheduler_arguments`.

    Args:
        args (Namespace): A namespace returned by
            `argparse.ArgumentParser.parse_args`.

    Return:
        dict

    """
    kwargs = {}
    if args.nodes is not None: kwargs['nodes'] = args.nodes
    if args.procs_per_node is not None:
        kwargs['procs_per_node'] = args.procs_per_node
    if args.partition: kwargs['partition'] = args.partition
    if args.account: kwargs['account'] = args.account
    if args.reservation: kwargs['reservation'] = args.reservation
    if args.time_limit: kwargs['time_limit'] = args.time_limit
    if args.setup_only: kwargs['setup_only'] = True
    return kwargs

def get_distconv_environment(parallel_io=False, num_io_partitions=1, init_nvshmem=False):
    """Return recommended Distconv variables.

    Args:
        parallel_io (bool):
            Whether to read a single sample in parallel.
        num_io_partitions (int):
            The number of processes to read a single sample.
    """
    # TODO: Use the default halo exchange and shuffle method. See https://github.com/LLNL/lbann/issues/1659
    environment = {
        'DISTCONV_WS_CAPACITY_FACTOR': 0.8,
        'LBANN_DISTCONV_HALO_EXCHANGE': 'AL',
        'LBANN_DISTCONV_TENSOR_SHUFFLER': 'AL',
        'LBANN_DISTCONV_CONVOLUTION_FWD_ALGORITHM': 'AUTOTUNE',
        'LBANN_DISTCONV_CONVOLUTION_BWD_DATA_ALGORITHM': 'AUTOTUNE',
        'LBANN_DISTCONV_CONVOLUTION_BWD_FILTER_ALGORITHM': 'AUTOTUNE',
        'LBANN_DISTCONV_RANK_STRIDE': 1,
        'LBANN_DISTCONV_COSMOFLOW_PARALLEL_IO': parallel_io,
        'LBANN_DISTCONV_NUM_IO_PARTITIONS': num_io_partitions,
        "LBANN_KEEP_ERROR_SIGNALS": "1",
    }
    if init_nvshmem:
        environment["LBANN_INIT_NVSHMEM"] = 1

    return environment

def add_optimizer_arguments(parser, default_optimizer='momentum',
                            default_learning_rate=0.01):
    """Add command-line arguments for optimizers.

    Adds the following options: `--optimizer`,
    `--optimizer-learning-rate`. The parsed arguments
    (e.g. `parser.parse_args()`) are the input for `create_optimizer`.

    Args:
        parser (argparse.ArgumentParser): command-line argument
            parser.
        default_optimizer (str): default optimizer to use.
        default_learning_rate (float): default learning rate.

    """
    if not isinstance(parser, argparse.ArgumentParser):
        raise TypeError('expected an argparse.ArgumentParser')
    parser.add_argument(
        '--optimizer', action='store', default=default_optimizer, type=str,
        choices=('momentum', 'sgd', 'adam', 'adagrad', 'rmsprop'),
        help='optimizer (default: {})'.format(default_optimizer))
    parser.add_argument(
        '--optimizer-learning-rate',
        action='store', default=default_learning_rate, type=float,
        help='optimizer learning rate (default: {})'.format(default_learning_rate),
        metavar='VAL')

def create_optimizer(args):
    """Create optimizer from command-line arguments.

    The parsed arguments must be generated by an
    `argparse.ArgumentParser` that has been processed by
    `add_optimizer_arguments`.

    Args:
        args (Namespace): A namespace returned by
           `argparse.ArgumentParser.parse_args`.

    Return:
        lbann.optimizer.Optimizer

    """

    # Get parsed command-line arguments
    try:
        opt = args.optimizer
        lr = args.optimizer_learning_rate
    except AttributeError:
        raise ValueError('parsed arguments have not been processed '
                         'by `add_optimizer_arguments`')

    # Create optimizer
    if opt == 'momentum':
        return lbann.core.optimizer.SGD(learn_rate=lr, momentum=0.9)
    elif opt == 'sgd':
        return lbann.core.optimizer.SGD(learn_rate=lr)
    elif opt == 'adam':
        return lbann.core.optimizer.Adam(learn_rate=lr, beta1=0.9, beta2=0.99,
                                         eps=1e-8)
    elif opt == 'adagrad':
        return lbann.core.optimizer.AdaGrad(learn_rate=lr, eps=1e-8)
    elif opt == 'rmsprop':
        return lbann.core.optimizer.RMSprop(learn_rate=lr, decay_rate=0.99,
                                            eps=1e-8)
    else:
        raise ValueError('invalid optimizer type ({})'.format(opt))


def add_profiling_arguments(parser: argparse.ArgumentParser) -> None:
    """Add command-line arguments for common profiler settings within
    LBANN.

    Adds the following options: `--profile`, `--profile-init`,
    `--caliper` and `--caliper-config`.

    `--caliper-config` implies `--caliper`. `--caliper` without a
    `--caliper-config` will use the default configuration in LBANN.
    These options will only have an effect if LBANN has been built
    with Caliper support.

    The caller is responsible for using them.

    Args:
        parser (argparse.ArgumentParser): command-line argument parser.

    """
    if not isinstance(parser, argparse.ArgumentParser):
        raise TypeError('expected an argparse.ArgumentParser')
    parser.add_argument(
        '--profile', action='store_true', default=False,
        help='enable profiling instrumentation and markers')
    parser.add_argument(
        '--profile-init', action='store_true', default=False,
        help='enable profiling initialization')
    parser.add_argument('--caliper', action='store_true', default=False,
                        help='enable Caliper')
    parser.add_argument(
        '--caliper-config', action='store', default=None, type=str,
        help='Configuration string for Caliper')


def create_profile_callback(args: argparse.Namespace) -> Any:
    """Create a profiler callback from command-line arguments.

    The parsed arguments must be generated by an
    `argparse.ArgumentParser` that has been processed by
    `add_profiling_arguments`.

    Args:
        args (argparse.Namespace): A namespace returned by
            `argparse.ArgumentParser.parse_args`.

    Return:
        None or lbann.CallbackProfiler

    """
    try:
        profile = args.profile
        profile_init = not args.profile_init
    except AttributeError:
        raise ValueError('passed arguments have not been processed by '
                         '`add_profiling_arguments`')

    if profile:
        return lbann.CallbackProfiler(skip_init=profile_init)
    return None


def get_profile_args(args: argparse.Namespace) -> list[str]:
    """Get LBANN command-line arguments for profiling.

    The parsed arguments must be generated by an
    `argparse.ArgumentParser` that has been processed by
    `add_profiling_arguments`.

    Args:
        args (argparse.Namespace): A namespace returned by
            `argparse.ArgumentParser.parse_args`.

    Return:
        list[str]: A list of command-line arguments to add.

    """
    try:
        caliper = args.caliper
        caliper_config = args.caliper_config
    except AttributeError:
        raise ValueError('passed arguments have not been processed by '
                         '`add_profiling_arguments`')

    if lbann.has_feature('CALIPER'):
        if caliper_config:
            return ['--caliper', '--caliper_config', f'"{caliper_config}"']
        if caliper:
            return ['--caliper']
    elif caliper_config or caliper:
        raise RuntimeError('Requested Caliper but LBANN does not have Caliper support.')
    return []
