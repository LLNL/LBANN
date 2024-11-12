import numpy as np
from SMILES_tokenizer import MolTokenizer
from data_utils import random_zero_array
import os
import os.path


def main():
    data_dir = os.getenv("QM9_DATA_DIR", "/p/vast1/lbann/datasets/FLASK/QM9")

    tokenizer = MolTokenizer(os.path.join(data_dir, "QM9_vocab.json"))
    tokenizer.load_vocab_file()

    data_file = os.path.join(data_dir, "QM9_smiles.txt")
    with open(data_file, "r") as smiles_data:
        smiles_data = smiles_data.readlines()
        num_samples = len(smiles_data)
        max_length = 32

        tokenized_data = np.ones((num_samples, max_length)) * tokenizer.encode(
            tokenizer.pad_token
        )
        tokenized_data[:, 0] = tokenizer.encode(tokenizer.sep_token)

        for i, smiles in enumerate(smiles_data, start=0):
            tokens = tokenizer.tokenize(smiles)
            tokenized_data[i, : len(tokens)] = tokens
            tokenized_data[i, len(tokens)] = tokenizer.encode(tokenizer.sep_token)
    save_file_loc = os.path.join(data_dir, "QM9_Pretokenized.npy")
    np.save(save_file_loc, tokenized_data)


if __name__ == "__main__":
    main()
