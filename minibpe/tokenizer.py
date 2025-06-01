from minibpe.base import *
class BasicTokenizer(BaseTokenizer):
    def __init__(self):
        super.__init__()
        
    """
    To train a tokenizer using bpe. Basic way (no regex)
        - Encode your text using utf-8 encoding
        - Start a loop that only ends when you have reached your desired vocab_size
            - Get the most common pair
            - Merge the pair and assign a new token to it
        - return new vocab
    """
    def train(self, text: str, vocab_size, verbose=False):
        ids = list(text.encode("utf-8"))
        idx = vocab_size - len(self.vocab)

        while idx < vocab_size:
            pair = get_pair_by_frequency(ids)
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"Merging {pair} into new token {idx}")
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
            idx += 1
        return

    def encode(self, text):
        pass

    def decode(self, ids):
        pass