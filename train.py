import os
import time
from minibpe import BasicTokenizer
from minibpe import RegexTokenizer

def main():
    corpus = "tests/taylorswift.txt"
    text = open(corpus, "r", encoding="utf-8").read()

    os.makedirs("models", exist_ok=True)
    name = "regex"

    start_time = time.time()

    tokenizer = RegexTokenizer()
    tokenizer.train(text, vocab_size=512, verbose=True)
    file_prefix = os.path.join("models", name)
    tokenizer.save(file_prefix=file_prefix)

    print(f"Training took {time.time() - start_time:.2f} seconds")
    print(tokenizer.decode(tokenizer.encode("My name is John")) == "My name is John")

if __name__ == "__main__":
    main()
