from .base import *
import regex as re

class RegexTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()
        self.pattern_str = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self._pat = re.compile(self.pattern_str)
    
    def train(self, text: str, vocab_size: int, verbose: bool = False):
        assert vocab_size >= 256
        # pretokenize string using regex pattern
        words = re.findall(self._pat, text)
        text_tokens: list[list[int]] = [list(word.encode("utf-8")) for word in words]

        for i in range(len(self.vocab), vocab_size):
            # get pair counts
            # create a counts dictionary that will be updated for each token in text_tokens
            counts = {}
            for tokens in text_tokens:
                counts = count_pairs(tokens, counts)
            if not counts:
                break
            # get the max occuring pair
            pair = max(counts, key=lambda p : counts[p])

            # combine pair and assign new token id to it
            self.vocab[i] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
            # merge the pair and get new text_tokens list
            text_tokens = [merge_tokens(ids=tokens, pair=pair, idx=i) for tokens in text_tokens]
            
            self.merges[pair] = i  # add pair () to merges dict

            if verbose:
                print(f"Merged {pair} into new token {i}")
            
    def encode(self, text: str) -> list[int]:
        # pretokenize string using regex pattern
        words = re.findall(self._pat, text)
        text_tokens: list[list[int]] = [list(word.encode("utf-8")) for word in words]

        while True:
            counts = {}
            for tokens in text_tokens:
                if len(tokens) >= 2:
                    counts = count_pairs(tokens, counts)
            if len(counts) < 1:
                break
            # get the max occuring pair that has lowest index/rank in merges
            pair: tuple[int, int] = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            
            # for each sub token sequence, merge the current pair
            text_tokens = [merge_tokens(ids=tokens, pair=pair, idx=self.merges[pair]) for tokens in text_tokens]
            # new_text_tokens = []
            # for tokens in text_tokens:
            #    new_tokens = merge(ids=tokens, pair=pair, idx=self.merges[pair])
            #    new_text_tokens.append(new_tokens)
            # text_tokens = new_text_tokens
                

        return [final_token for tokens in text_tokens for final_token in tokens]
    
    def decode(self, tokens: list[int]) -> str:
        text_bytes = b"".join([self.vocab[token] for token in tokens])
        return text_bytes.decode(encoding="utf-8", errors="replace")