# Pre-Generate Vocabulary

For larger datasets such as PubChem and ZINC, the vocabulary generation process can be time-consuming. To speed up the process, we pre-generate the vocabulary and save it to a file. This allows us to load the vocabulary from the file instead of generating it from scratch.

If for some reason, we need to regenerate the vocabulary:
  
```bash
python GenerateVocabulary.py --qm9 --zinc --pubchem
```

This assumes appropriate data is available and the environment flags, `QM9_FILE`, `ZINC_DIR`, and `PUBCHEM_FILE` are set. For LC, approprate defaults are set. 

## Vocabulary Files

The current repository contains the following vocabulary files:

- `QM9_vocab.json`
- `ZINC_vocab.json`
- `PubChem_vocab.json`

# Pre-Tokenize Data

Currently, we only have a pre-tokenizer for the QM9 dataset. PubChem and ZINC are too large and require too much padding to efficiently pre-tokenize and store and load (for now).
