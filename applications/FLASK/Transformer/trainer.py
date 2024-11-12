"""
Constructs the LBANN distributed training script for transformers.
"""

import argparse
import datetime
import os.path

import lbann
import lbann.models
import lbann.contrib.args
import lbann.contrib.launcher
from lbann.launcher.batch_script import BatchScript
import network
import paths

import dataset_utils
import arg_utils
import network


def construct_training_task(
    model: lbann.Model,
    args: argparse.Namespace,
    learning_rate: float = 0.0001,
    beta1: float = 0.9,
    beta2: float = 0.98,
    eps: float = 1e-9,
    clip_gradient: float = 0.0,
    lr_decay: str = "fixed",
    lr_decay_steps: int = 0,
    end_learning_rate: float = 1e-5,
    warmup_steps: int = 0,
    adamw_decay: float = 0.1,
) -> BatchScript:
    """
    Construct an LBANN trainer batch script for training transformers.

    :param model: An LBANN model.
    :param args: Command-line arguments.
    :param learning_rate: Learning rate.
    :param beta1: Adam beta1 factor.
    :param beta2: Adam beta2 factor.
    :param eps: Adam epsilon factor.
    :param clip_gradient: Clip gradient norm to value (0 disables).
    :param lr_decay: Learning rate decay type (values: fixed, cosine, none).
    :param lr_decay_steps: Steps for the total learning decay process (in cosine
                           decay).
    :param end_learning_rate: Learning rate after decay.
    :param warmup_steps: Learning rate warmup steps.
    :param adamw_decay: Weight decay if using the AdamW optimizer.
    :return: A batch script object that will run distributed training.
    """

    # Setup working directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = f"{timestamp}_{args.job_name}"
    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    # Create batch script
    train_script = make_batch_script(
        model,
        args.dataset,
        work_dir,
        args,
        learning_rate,
        beta1,
        beta2,
        eps,
        clip_gradient,
        lr_decay,
        lr_decay_steps,
        end_learning_rate,
        warmup_steps,
        adamw_decay,
    )

    return train_script


# ----------------------------------------------
# Data reader
# ----------------------------------------------
def make_data_reader(
    dataset_name: str,
    fraction: float,
    validate: bool,
    val_fraction: float,
    always_shuffle: bool,
):
    reader = lbann.reader_pb2.DataReader()
    _reader = reader.reader.add()
    _reader.name = "python"
    _reader.role = "train"
    _reader.shuffle = (
        True if always_shuffle or "pretokenized" not in dataset_name else False
    )
    _reader.fraction_of_data_to_use = fraction
    _reader.python.module = dataset_name
    _reader.python.module_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "datasets",
    )
    _reader.python.sample_function = "get_train_sample"
    _reader.python.num_samples_function = "num_train_samples"
    _reader.python.sample_dims_function = "sample_dims"

    if validate:
        # Validation data reader
        vreader = reader.reader.add()
        vreader.name = "python"
        vreader.role = "validate"
        vreader.shuffle = False
        vreader.fraction_of_data_to_use = val_fraction
        vreader.python.module = _reader.python.module
        vreader.python.module_dir = _reader.python.module_dir
        vreader.python.sample_function = "get_val_sample"
        vreader.python.num_samples_function = "num_val_samples"
        vreader.python.sample_dims_function = "sample_dims"

    return reader


# ----------------------------------------------
# Batch script
# ----------------------------------------------
def make_batch_script(
    model: lbann.Model,
    dataset_name: str,
    work_dir: str,
    args: argparse.Namespace,
    learning_rate: float = 0.0001,
    beta1: float = 0.9,
    beta2: float = 0.98,
    eps: float = 1e-9,
    clip_gradient: float = 0.0,
    lr_decay: str = "fixed",
    lr_decay_steps: int = 0,
    end_learning_rate: float = 1e-5,
    warmup_steps: int = 0,
    adamw_decay: float = 0.1,
):
    # Setup training algorithm
    algo = lbann.BatchedIterativeOptimizer("sgd", epoch_count=args.num_epochs)

    # Create LBANN trainer and data reader
    trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size, training_algo=algo)
    reader = make_data_reader(
        dataset_name,
        args.dataset_fraction,
        not args.skip_validation,
        args.validation_set_fraction,
        args.always_shuffle,
    )

    # Optimizer with learning rate schedule
    if args.optimizer.lower() == "adamw":
        opt = lbann.Adam(
            learn_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            adamw_weight_decay=adamw_decay,
        )
    elif args.optimizer.lower() == "adam":
        opt = lbann.Adam(learn_rate=learning_rate, beta1=beta1, beta2=beta2, eps=eps)

    if lr_decay == "fixed":
        if warmup_steps > 0:
            raise NotImplementedError("Warmup not implemented with fixed learning rate")

        model.callbacks.append(
            lbann.CallbackDropFixedLearningRate(
                drop_epoch=[1],
                amt=2,
            )
        )
        model.callbacks.append(
            lbann.CallbackDropFixedLearningRate(
                drop_epoch=[2, 4, 8, 12],
                amt=0.75,
            )
        )
    elif lr_decay == "cosine":
        model.callbacks.append(
            lbann.CallbackCosineDecayLearningRate(
                lr_max=learning_rate,
                lr_min=end_learning_rate,
                decay_steps=lr_decay_steps,
                initial_warmup_learning_rate=end_learning_rate,
                warmup_steps=warmup_steps,
            )
        )

        print(
            f"Training schedule: warmup to LR={learning_rate:.6f} in "
            f"{warmup_steps} steps, cosine decay to "
            f"LR={end_learning_rate:.6f} in {lr_decay_steps} steps"
        )

    if clip_gradient > 0:
        model.callbacks.append(
            lbann.CallbackClipGradientNorm(global_norm=True, value=clip_gradient)
        )

    # Checkpoint after every epoch
    if args.checkpoint:
        trainer.callbacks.append(
            lbann.CallbackCheckpoint(
                checkpoint_dir=os.path.join(work_dir, "checkpoint"),
                checkpoint_epochs=1,
            )
        )

        # Dump weights after every epoch
        model.callbacks.append(
            lbann.CallbackDumpWeights(
                directory=os.path.join(work_dir, "weights"),
                epoch_interval=1,
            )
        )

    # Print a progress bar
    if args.progress:
        model.callbacks.append(
            lbann.CallbackProgressBar(newline_interval=100, print_mem_usage=True)
        )

    model.callbacks.extend(lbann.contrib.args.create_profile_callbacks(args))

    script_params = lbann.contrib.args.get_scheduler_kwargs(args)
    script_params["work_dir"] = work_dir
    script_params["job_name"] = args.job_name
    script_params["environment"] = {
        "LBANN_NO_INPLACE": 1,
        "LBANN_DISABLE_DISTCONV": 1,
    }

    save_text = args.save_prototext
    filename = "experiment.prototext" if save_text else "experiment.protobin"
    # Create Protobuf file
    protobuf_file = os.path.join(work_dir, filename)

    lbann.proto.save_prototext(
        protobuf_file,
        binary=not save_text,
        trainer=trainer,
        model=model,
        data_reader=reader,
        optimizer=opt,
    )

    # Create batch script
    script_params.pop("setup_only", None)  # Drop this argument.
    script = lbann.contrib.launcher.make_batch_script(**script_params)
    script.add_command('echo "Started training at $(date)"')
    script.add_parallel_command(
        [
            lbann.lbann_exe(),
            f"--prototext={protobuf_file}",
        ]
        + lbann.contrib.args.get_profile_args(args)
    )
    script.add_command("status=$?")
    script.add_command('echo "Finished training at $(date)"')
    script.add_command("exit ${status}")
    return script


def main():
    # Setup command line options
    parser = argparse.ArgumentParser()
    lbann.contrib.args.add_scheduler_arguments(parser, "lbann_transformer")
    lbann.contrib.args.add_profiling_arguments(parser)
    lbann.contrib.args.add_training_arguments(parser)
    arg_utils.add_transformer_architecture_arguments(parser)
    arg_utils.add_training_arguments(parser)
    arg_utils.add_dataset_arguments(parser, default="QM9")

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "adamw"],
        help="Stochastic optimizer used in training",
    )

    # Model parameters
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout ratio in transformer model. 0 disables (default: 0.1)",
    )
    parser.add_argument(
        "--input-dropout",
        type=float,
        default=0.0,
        help="Dropout ratio after input encoding (default: 0.0 = disabled)",
    )

    parser.set_defaults(progress=True, skip_validation=True)
    args = parser.parse_args()

    # Load dataset
    dataset = dataset_utils.load_dataset(args.dataset)

    # Construct model
    model: lbann.Model = network.create_encoder_decoder_transformer(dataset, args)

    # Construct trainer
    train_script: BatchScript = construct_training_task(model, args)

    # Run trainer
    retval = train_script.run(overwrite=True)
    if retval != 0:
        return

    if args.checkpoint:
        print(
            "Training complete, to evaluate the translation with a BLEU score, "
            "run ``evaluate_translation_bleu.py`` with the model checkpoint path "
            "and the same arguments as this training run."
        )
    else:
        print(
            "Training complete, to evaluate the translation with a BLEU score, "
            "run this script with --checkpoint"
        )


if __name__ == "__main__":
    main()
