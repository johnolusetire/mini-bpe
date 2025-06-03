import os
import base64

class BaseTokenizer:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.special_tokens: dict[str, int] = {}  # e.g <|eos|> : 100234
        self.pattern: str = ""
        self.vocab: dict[int, bytes] = self._build_vocab()

    def train(self, text: str, vocab_size: int, verbose: bool=False) -> None:
        raise NotImplementedError("train() must be implemented by subclasses")
    
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError("encode() must be implemented by subclasses")
    
    def decode(self, tokens: list[int]) -> str:
        raise NotImplementedError("decode() must be implemented by subclasses")
    
    def _build_vocab(self) -> dict[int, bytes]:
        """ Builds vocabs from base byte vocab and merges"""
        # initial 256 characters
        vocab = {idx : bytes([idx]) for idx in range(256)}

        # add merges
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
          # add special tokens
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode(encoding="utf-8")
        
        return vocab
    
    def save(self, file_prefix: str, path: str | None = None):
        """
        Saves the tokenizer model to a file.
        - file_prefix (str): The prefix for the model file name.
        - path (str | None): The directory path where the model file will be saved. If None, the file will be saved in the current directory.       
                            
        """

        dir_path = path if path is not None else '.'
        os.makedirs(dir_path, exist_ok=True)

        file_path = os.path.join(dir_path, f"{file_prefix}.json")
        merge_serial = [f"{p0} {p1}" for p0, p1 in self.merges]
        vocab_serial = {str(k) : base64.b64encode(v).decode('ascii') for k,v in self.vocab.items()}

        model_data = {
            "model_type" : "bpe",
            'merges': merge_serial,
            'special_tokens': self.special_tokens,
            'vocab': vocab_serial,
            "version": "minibpe-v1"
        }

        with open(file_path, 'w') as f:
            import json
            json.dump(model_data, f, indent=2)

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileExistsError(f"Path given {path} does not exist does")
        assert path.endswith(".json")
        
        with open(path, "r", encoding="utf-8") as f:
            pass


def count_pairs(ids: list[int], counts: dict | None = None) -> dict[tuple[int, int], int]:
    # assign counts to an empty dictionary or use the passed counts dictionary
    # If counts is None, initialize it as an empty dictionary.
    counts = {} if counts is None else counts

    from itertools import pairwise
    for pair in pairwise(ids): # itertools.pairwise might be faster or a generator type solution zip(ids, ids[1:])
        counts[pair] = counts.get(pair, 0) + 1
    
    return counts

def merge_tokens(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    new_ids = []
    i = 0
    
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

