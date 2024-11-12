import regex as re
import json
import numpy as np
import os


PATTERN = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"


class MolTokenizer:
    def __init__(
        self,
        vocab_file: str = "",
        unk_token="<pad>",
        sep_token="<eos>",
        pad_token="<pad>",
        cls_token="<bos>",
        mask_token="<mask>",
        **kwargs
    ):
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.sep_token = sep_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        self.regex_tokenizer = re.compile(PATTERN)
        self.wordpiece_tokenizer = None
        self.basic_tokenizer = None
        self.vocab_file = vocab_file
        self.vocab_dict = {}
        self.vocab_dict[self.pad_token] = 0
        self.vocab_dict[self.sep_token] = 1
        self.vocab_dict[self.cls_token] = 2
        self.vocab_dict[self.mask_token] = 3
        self.counter = 4

    def load_vocab_file(self):
        if os.path.exists(self.vocab_file):
            with open(self.vocab_file, "r") as f:
                self.vocab_dict = json.load(f)
        else:
            raise NameError(f"Vocab file not found in {self.vocab_file}")

    def load_vocab_dict(self, vocab_dict):
        self.vocab_dict = vocab_dict
        self.counter = len(vocab_dict)

    def _tokenize(self, text):
        split_tokens = self.regex_tokenizer.findall(text)
        return split_tokens

    def tokenize(self, text):
        split_tokens = self._tokenize(text)
        output = np.zeros(len(split_tokens))
        for i, each in enumerate(split_tokens):
            if each not in self.vocab_dict.keys():
                self.vocab_dict[each] = self.counter
                self.counter += 1
            output[i] = self.vocab_dict[each]
        return output

    def encode(self, token):
        return self.vocab_dict[token]

    def convert_tokens_to_string(self, tokens):
        out_string = "".join(tokens).strip()
        return out_string

    def generate_vocab_file(self, vocab_file_name):
        with open(vocab_file_name, "w") as f:
            f.write(json.dumps(self.vocab_dict))

    def token_to_id(self, token):
        return self.vocab_dict[token]
