"""
The QM9 dataset, stored as pre-tokenized binary files for optimized processing.
"""

import os
import os.path

import numpy as np
from pretokenize.SMILES_tokenizer import MolTokenizer
from pretokenize.data_utils import random_zero_array

sequence_length = int(os.getenv("QM9_SEQUENCE_LENGTH", default="32"))

# ----------------------------------------------
# Setup
# ----------------------------------------------

# Load the datasets
data_dir = os.getenv("QM9_DATA_DIR", "/p/vast1/lbann/datasets/FLASK/QM9")

tokenizer = MolTokenizer(os.path.join(data_dir, "QM9_vocab.json"))
tokenizer.load_vocab_file()

dataset_train = np.load(os.path.join(data_dir, "QM9_Pretokenized.npy"))

# dataset_train = np.zeros((140000, 32), dtype=np.float32)
_vocab_size = 46
pad_index = tokenizer.token_to_id("<pad>")
sep_index = tokenizer.token_to_id("<eos>")

# ----------------------------------------------
# Sample access functions
# ----------------------------------------------


def num_train_samples():
    return dataset_train.shape[0]


def get_train_sample(i):
    data = dataset_train[i]

    boundary = np.where(data == sep_index)[0][0]
    masked_data = random_zero_array(
        data[:boundary], 0.15, tokenizer.token_to_id(tokenizer.mask_token)
    )
    output = np.zeros((2 * sequence_length), dtype=np.int32)
    output[0:boundary] = masked_data
    output[boundary] = sep_index
    output[sequence_length:] = data
    return output


def sample_dims():
    return (2 * sequence_length + 1,)


def vocab_size():
    return _vocab_size


if __name__ == "__main__":
    print("Training samples:", num_train_samples())
    print("Training sample 101:")
    print(get_train_sample(0))
