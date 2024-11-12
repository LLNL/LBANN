from multiprocessing import Pool
from SMILES_tokenizer import MolTokenizer
from glob import glob
import argparse
import os
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description="Generate vocab files for different datasets"
)

parser.add_argument(
    "--qm9", action="store_true", help="Generate vocab file for QM9 dataset"
)
parser.add_argument(
    "--zinc", action="store_true", help="Generate vocab file for ZINC dataset"
)
parser.add_argument(
    "--pubchem", action="store_true", help="Generate vocab file for PubChem dataset"
)

args = parser.parse_args()


def join_vocabs(list_of_vocab_dicts):
    """
    Given a list of vocab dictionaries, join them together
    such that all unique tokens are present in the final vocab dictionary
    """
    final_vocab = {}
    counter = 0
    for vocab in list_of_vocab_dicts:
        for token in vocab.keys():
            if token not in final_vocab.keys():
                final_vocab[token] = counter
                counter += 1
    return final_vocab


def generate_zinc_vocab_dict(smi_file):
    tokenizer = MolTokenizer()
    with open(smi_file, "r") as f:
        data = f.readlines()
    for i in tqdm(range(1, len(data))):
        line = data[i].split(" ")
        _ = tokenizer._tokenize(line[0])
    return tokenizer.vocab_dict


def main():

    if args.qm9:
        print("Generating vocab file for QM9 dataset")
        tokenizer = MolTokenizer("QM9_vocab.json")
        default_file = "/p/vast1/lbann/datasets/FLASK/QM9/QM9_smiles.txt"
        qm9_file = os.getenv("QM9_FILE", default_file)
        with open(qm9_file, "r") as smiles_data:
            smiles_data = smiles_data.readlines()
            for line in tqdm(smiles_data):
                tokens = tokenizer.tokenize(line)
        tokenizer.generate_vocab_file("QM9_vocab.json")
        print("QM9 vocab file generated")

    if args.zinc:
        print("Generating vocab file for ZINC dataset")
        default_dir = "/p/vast1/lbann/datasets/FLASK/ZINC"
        zinc_dir = os.getenv("ZINC_DIR", default_dir)
        zinc_files = glob(f"{zinc_dir}/*.smi")

        print(len(zinc_files))

        with Pool(20) as p:
            zinc_vocab_dicts = p.map(generate_zinc_vocab_dict, zinc_files)

        final_vocab = join_vocabs(zinc_vocab_dicts)

        final_tokenizer = MolTokenizer("ZINC_vocab.json")
        final_tokenizer.load_vocab_dict(final_vocab)
        final_tokenizer.generate_vocab_file("ZINC_SMILES_vocab.json")
        print("ZINC vocab file generated")

    if args.pubchem:
        print("Generating vocab file for PubChem dataset")
        default_file = "/p/vast1/lbann/datasets/FLASK/pubchem/CID_SMILES_CANONICAL.smi"
        pubchem_file = os.getenv("PUBCHEM_FILE", default_file)
        with open(pubchem_file, "r") as smiles_data:
            smiles_data = smiles_data.readlines()
            tokenizer = MolTokenizer("PubChem_SMILES_vocab.json")
            for line in tqdm(smiles_data):
                smiles = line.split(" ")[1]
                tokens = tokenizer.tokenize(smiles)

            tokenizer.generate_vocab_file("PubChem_vocab.json")
            print("PubChem vocab file generated")


if __name__ == "__main__":
    main()
