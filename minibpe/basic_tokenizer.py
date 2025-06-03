from .base import *
class BasicTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()
        
    """
    To train a tokenizer using bpe. Basic way (no regex)
        - Encode your text using utf-8 encoding
        - Start a loop that only ends when you have reached your desired vocab_size
            - Get the most common pair
            - Merge the pair and assign a new token to it
        - return new vocab
    """
    def train(self, text: str, vocab_size, verbose=False):
        assert vocab_size >= 256
        tokens = list(text.encode("utf-8"))

        for i in range(len(self.vocab), vocab_size):
            # get pair counts
            counts = count_pairs(tokens)
            # if counts is empty break (edge case that probably won't happen)
            if not counts:
                break
            # get max occuring pair
            pair = max(counts, key=lambda p : counts[p])
            # combine pair and assign new id to it
            self.vocab[i] = self.vocab[pair[0]] + self.vocab[pair[1]]
            # merge pair and get new id/token list
            tokens = merge_tokens(tokens, pair, i)
            self.merges[pair] = i
            
            if verbose:
                print(f"Merging {pair} into new token {i}")

    """
    To encode a given string to a list of tokens
    - encode the string using utf-8
    - merge the tokens using the merge list
    """
    def encode(self, text: str):
        # encode and get list of byte tokens
        tokens = list(text.encode(encoding="utf-8"))

        while len(tokens) >= 2:
            # get the counts
            counts = count_pairs(tokens)
            # get pair that is in merges and is assigned to the lowest token/idx in merges
            pair = min(counts, key=lambda p : self.merges.get(p, float("inf"))) # manual loop is actual faster from benchmark
            if pair not in self.merges:
                break
            # get merge token from self.merges
            idx = self.merges[pair]
            tokens = merge_tokens(ids=tokens, pair=pair, idx=idx)
        
        return tokens

    """
    To decode, get the byte representation for each token and decode with utf-8
    """
    def decode(self, tokens: list[int]):
        text_bytes = b"".join([self.vocab[token] for token in tokens])
        return text_bytes.decode(encoding="utf-8", errors="replace")

def main():
    tokenizer = BasicTokenizer()
    text = "aaab"
    vocab_size = 256 + 20 # Expect one merge: ('a', 'a')
    tokenizer.train(text, vocab_size, verbose=False)
    test = tokenizer.vocab
    tokenizer.save("test")
    
if __name__ == "__main__":
    main()