import os

class BaseTokenizer:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.special_tokens: dict[str, int] = {}  # e.g <|eos|> : 100234
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError("train() must be implemented by subclasses")
    
    def encode(self, text):
        raise NotImplementedError("encode() must be implemented by subclasses")

    def decode(self, ids):
        raise NotImplementedError("decode() must be implemented by subclasses")
    
    def _build_vocab(self):
        """ Builds vocabs from base byte vocab and merges"""
        # initial 256 characters
        vocab = {idx : bytes(idx) for idx in range(256)}

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

        model_data = {
            "model_type" : "bpe",
            'merges': merge_serial,
            'special_tokens': self.special_tokens,
            'vocab': self.vocab,
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


def get_pair_by_frequency(ids: list[int], mode: str = "max", counts: dict | None = None) -> tuple[int, int]:
    # make sure mode is set to max or min
    assert mode == "max" or mode == "min", "Invalid mode: mode must be either 'max' or 'min'"

    # assign counts to an empty dictionary or use the passed counts dictionary
    # If counts is None, initialize it as an empty dictionary.
    counts = {} if counts is None else counts
    
    max_min_pair: tuple[int, int] | None = None
    max_min_seen = float("-inf") if mode == "max" else float("inf")

    for pair in zip(ids, ids[1:]): # itertools.pairwise might be faster or a generator type solution
        counts[pair] = counts.get(pair, 0) + 1
        
        # If the mode is "max", update max_min_pair to the current pair if its count is greater than max_min_seen
        if mode == "max" and counts[pair] > max_min_seen:
            max_min_pair = pair
            max_min_seen = counts[pair]
        # If the mode is "min", update max_min_pair to the current pair if its count is less than max_min_seen
        elif mode == "min" and counts[pair] < max_min_seen:
            max_min_pair = pair
            max_min_seen = counts[pair]

    if max_min_pair is None:
        raise ValueError("No valid pair found in the input list.")
    
    return max_min_pair

def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
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

