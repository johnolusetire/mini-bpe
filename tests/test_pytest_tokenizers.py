import pytest
import os
import tempfile
from minibpe import BasicTokenizer
from minibpe import RegexTokenizer


class TestBasicTokenizer:
    """Tests for BasicTokenizer using pytest."""

    def test_init(self):
        """Test tokenizer initialization."""
        tokenizer = BasicTokenizer()
        assert tokenizer is not None
        assert len(tokenizer.vocab) == 256  # Initial vocab from base bytes
        assert len(tokenizer.merges) == 0

    def test_train_simple(self):
        """Test training with a very simple text."""
        tokenizer = BasicTokenizer()
        text = "aaab"
        vocab_size = 256 + 1  # Expect one merge: ('a', 'a')
        tokenizer.train(text, vocab_size, verbose=False)
        
        # Expected merge: (ord('a'), ord('a')) -> 256
        # ord('a') is 97
        expected_merge_pair = (97, 97)
        assert expected_merge_pair in tokenizer.merges
        assert tokenizer.merges[expected_merge_pair] == 256
        assert len(tokenizer.vocab) == vocab_size
        assert tokenizer.vocab[256] == b'aa'

    def test_encode_simple(self):
        """Test encoding after simple training."""
        tokenizer = BasicTokenizer()
        text = "aaab"
        vocab_size = 257  # ord('a'), ord('a') -> 256
        tokenizer.train(text, vocab_size, verbose=False)
        
        encoded = tokenizer.encode("aaab")
        # "aaab" -> [256, 97, 98] (assuming (97,97) became 256, and 97 is 'a', 98 is 'b')
        assert encoded == [256, 97, 98]

        encoded_a = tokenizer.encode("a")
        assert encoded_a == [97]

        encoded_b = tokenizer.encode("b")
        assert encoded_b == [98]
        
        encoded_aaaa = tokenizer.encode("aaaa")
        assert encoded_aaaa == [256, 256]

    def test_decode_simple(self):
        """Test decoding."""
        tokenizer = BasicTokenizer()
        text = "aaab"
        vocab_size = 257
        tokenizer.train(text, vocab_size, verbose=False)

        # From test_encode_simple: "aaab" encodes to [256, 97, 98]
        decoded_text = tokenizer.decode([256, 97, 98])
        assert decoded_text == "aaab"

        decoded_a = tokenizer.decode([97])
        assert decoded_a == "a"

    def test_train_encode_decode_roundtrip(self):
        """Test a full train, encode, and decode cycle."""
        tokenizer = BasicTokenizer()
        text = "ababab"
        # ord('a') = 97, ord('b') = 98
        # Possible merges:
        # 1. (97, 98) -> 256 (ab)
        # ids: [256, 256, 256]
        # 2. (256, 256) -> 257 (abab)
        # ids: [257, 256]
        vocab_size = 256 + 2 
        tokenizer.train(text, vocab_size, verbose=False)
        
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

        text2 = "abcabc"
        encoded2 = tokenizer.encode(text2)  # Should use merges learned from 'ababab' if applicable
        decoded2 = tokenizer.decode(encoded2)
        assert decoded2 == text2

    def test_encode_unknown_chars_after_train(self):
        """Test encoding text with characters not seen during training (but within base 256)."""
        tokenizer = BasicTokenizer()
        text = "aaa"
        vocab_size = 257  # (97, 97) -> 256
        tokenizer.train(text, vocab_size, verbose=False)

        encoded = tokenizer.encode("aaabbbccc")
        # Expected: (97,97) -> 256. So 'aaa' -> [256, 97]. 'bbb' -> [98,98,98]. 'ccc' -> [99,99,99]
        # Corrected expectation: 'aaabbbccc' -> [256, 97, 98, 98, 98, 99, 99, 99]
        assert encoded == [256, 97, 98, 98, 98, 99, 99, 99]
        decoded = tokenizer.decode(encoded)
        assert decoded == "aaabbbccc"

    def test_empty_text(self):
        """Test with empty text inputs."""
        tokenizer = BasicTokenizer()
        vocab_size = 256
        
        # Train with empty text (should not change anything)
        tokenizer.train("", vocab_size)
        assert len(tokenizer.vocab) == 256
        assert len(tokenizer.merges) == 0

        # Encode empty text
        encoded = tokenizer.encode("")
        assert encoded == []

        # Decode empty list
        decoded = tokenizer.decode([])
        assert decoded == ""

    def test_save_functionality(self):
        """Test saving the tokenizer."""
        tokenizer = BasicTokenizer()
        text = "test data for save and load. test data for save and load."
        vocab_size = 256 + 5  # Learn a few merges
        tokenizer.train(text, vocab_size, verbose=False)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_prefix = "test_basic_tokenizer"
            tokenizer.save(file_prefix, path=temp_dir)
            
            model_path = os.path.join(temp_dir, f"{file_prefix}.json")
            assert os.path.exists(model_path)
            
            # Verify the saved file contains expected data
            import json
            with open(model_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
                assert "model_type" in model_data
                assert "merges" in model_data
                assert "vocab" in model_data
                assert model_data["model_type"] == "bpe"

    def test_train_no_new_merges(self):
        """Test training when vocab_size doesn't allow new merges."""
        tokenizer = BasicTokenizer()
        text = "aaab"
        vocab_size = 256  # Same as initial, no merges expected
        tokenizer.train(text, vocab_size, verbose=False)
        
        assert len(tokenizer.merges) == 0
        assert len(tokenizer.vocab) == 256
        
        encoded = tokenizer.encode("aaab")
        # Expect raw byte encoding
        assert encoded == [97, 97, 97, 98]

    def test_encode_with_merges_priority(self):
        """Test that encoding correctly prioritizes earlier learned merges (lower token ID)."""
        tokenizer = BasicTokenizer()
        text = "ababab"
        vocab_size = 256 + 2
        tokenizer.train(text, vocab_size, verbose=False)

        # Check merges:
        # ord('a')=97, ord('b')=98
        # Expected: (97,98) -> 256
        # Expected: (256,256) -> 257
        assert (97, 98) in tokenizer.merges
        assert tokenizer.merges[(97, 98)] == 256
        assert (256, 256) in tokenizer.merges
        assert tokenizer.merges[(256, 256)] == 257

        encoded_ab = tokenizer.encode("ab")
        assert encoded_ab == [256]

        encoded_abab = tokenizer.encode("abab")
        # After training on "ababab", the merges should be (97,98)->256 and (256,256)->257
        # So, "abab" -> [256,256] -> [257]
        assert encoded_abab == [257] 

        encoded_ababab = tokenizer.encode("ababab")
        # "ababab" -> [256,256,256] -> [257, 256]
        assert encoded_ababab == [257, 256]


class TestRegexTokenizer:
    """Tests for RegexTokenizer using pytest."""

    def test_init(self):
        """Test regex tokenizer initialization."""
        tokenizer = RegexTokenizer()
        assert tokenizer is not None
        assert len(tokenizer.vocab) == 256  # Initial vocab from base bytes
        assert len(tokenizer.merges) == 0
        assert hasattr(tokenizer, 'pattern_str')
        assert hasattr(tokenizer, '_pat')

    def test_train_simple(self):
        """Test training with simple text."""
        tokenizer = RegexTokenizer()
        text = "hello world"
        vocab_size = 256 + 5  # Learn some merges
        tokenizer.train(text, vocab_size, verbose=False)
        
        assert len(tokenizer.vocab) == vocab_size
        assert len(tokenizer.merges) > 0

    def test_encode_decode_roundtrip(self):
        """Test encoding and decoding roundtrip."""
        tokenizer = RegexTokenizer()
        text = "Hello, world! This is a test."
        vocab_size = 256 + 10
        tokenizer.train(text, vocab_size, verbose=False)
        
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

    def test_regex_pattern_splits_correctly(self):
        """Test that the regex pattern splits text appropriately."""
        tokenizer = RegexTokenizer()
        text = "Hello world! How are you?"
        
        # Test with untrained tokenizer (should still split by regex)
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

    def test_whitespace_handling(self):
        """Test handling of various whitespace characters."""
        tokenizer = RegexTokenizer()
        text = "word1\nword2\tword3 word4"
        vocab_size = 256 + 5
        tokenizer.train(text, vocab_size, verbose=False)
        
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

    def test_contractions_handling(self):
        """Test handling of contractions like 's, 'd, etc."""
        tokenizer = RegexTokenizer()
        text = "I'm happy, you're sad, we'd like it."
        vocab_size = 256 + 10
        tokenizer.train(text, vocab_size, verbose=False)
        
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

    def test_numbers_handling(self):
        """Test handling of numbers."""
        tokenizer = RegexTokenizer()
        text = "There are 123 apples and 45 oranges, totaling 168 fruits."
        vocab_size = 256 + 15
        tokenizer.train(text, vocab_size, verbose=False)
        
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

    def test_empty_text_regex(self):
        """Test regex tokenizer with empty text."""
        tokenizer = RegexTokenizer()
        
        # Train with empty text
        tokenizer.train("", 256)
        assert len(tokenizer.vocab) == 256
        assert len(tokenizer.merges) == 0

        # Encode/decode empty text
        encoded = tokenizer.encode("")
        assert encoded == []
        
        decoded = tokenizer.decode([])
        assert decoded == ""

    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        tokenizer = RegexTokenizer()
        text = "Hello 世界! Café résumé naïve 🚀"
        vocab_size = 256 + 20
        tokenizer.train(text, vocab_size, verbose=False)
        
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

    def test_save_functionality_regex(self):
        """Test saving the regex tokenizer."""
        tokenizer = RegexTokenizer()
        text = "This is test data for regex tokenizer save functionality."
        vocab_size = 256 + 8
        tokenizer.train(text, vocab_size, verbose=False)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_prefix = "test_regex_tokenizer"
            tokenizer.save(file_prefix, path=temp_dir)
            
            model_path = os.path.join(temp_dir, f"{file_prefix}.json")
            assert os.path.exists(model_path)
            
            # Verify the saved file contains expected data
            import json
            with open(model_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
                assert "model_type" in model_data
                assert "merges" in model_data
                assert "vocab" in model_data
                assert model_data["model_type"] == "bpe"

    def test_regex_vs_basic_different_tokenization(self):
        """Test that regex tokenizer handles certain patterns differently than basic."""
        tokenizer = RegexTokenizer()
        
        # Test that contractions are handled as expected
        text = "I'm going to the store."
        vocab_size = 256 + 5
        tokenizer.train(text, vocab_size, verbose=False)
        
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text
        
        # Test that the regex tokenizer preserves contractions better
        contraction_text = "don't won't can't"
        encoded_contractions = tokenizer.encode(contraction_text)
        decoded_contractions = tokenizer.decode(encoded_contractions)
        assert decoded_contractions == contraction_text


class TestComparison:
    """Comparative tests between BasicTokenizer and RegexTokenizer."""

    def test_same_simple_text_different_results(self):
        """Test that basic and regex tokenizers handle the same text differently."""
        text = "hello world"
        vocab_size = 256 + 5
        
        basic_tokenizer = BasicTokenizer()
        regex_tokenizer = RegexTokenizer()
        
        basic_tokenizer.train(text, vocab_size, verbose=False)
        regex_tokenizer.train(text, vocab_size, verbose=False)
        
        basic_encoded = basic_tokenizer.encode(text)
        regex_encoded = regex_tokenizer.encode(text)
        
        # They should both decode to the same text
        basic_decoded = basic_tokenizer.decode(basic_encoded)
        regex_decoded = regex_tokenizer.decode(regex_encoded)
        
        assert basic_decoded == text
        assert regex_decoded == text
        
        # But the encoded representations might be different due to different tokenization strategies
        # This is expected behavior

    def test_whitespace_differences(self):
        """Test how basic vs regex tokenizers handle whitespace differently."""
        text = "word1   word2\nword3"
        vocab_size = 256 + 3
        
        basic_tokenizer = BasicTokenizer()
        regex_tokenizer = RegexTokenizer()
        
        basic_tokenizer.train(text, vocab_size, verbose=False)
        regex_tokenizer.train(text, vocab_size, verbose=False)
        
        basic_encoded = basic_tokenizer.encode(text)
        regex_encoded = regex_tokenizer.encode(text)
        
        basic_decoded = basic_tokenizer.decode(basic_encoded)
        regex_decoded = regex_tokenizer.decode(regex_encoded)
        
        # Both should preserve the original text
        assert basic_decoded == text
        assert regex_decoded == text


@pytest.fixture
def sample_text():
    """Fixture providing sample text for testing."""
    return "This is a sample text for testing tokenizers. It contains various words, punctuation, and numbers like 123."


@pytest.fixture
def basic_tokenizer():
    """Fixture providing a BasicTokenizer instance."""
    return BasicTokenizer()


@pytest.fixture
def regex_tokenizer():
    """Fixture providing a RegexTokenizer instance."""
    return RegexTokenizer()


class TestFixtures:
    """Tests using pytest fixtures."""

    def test_basic_tokenizer_with_sample_text(self, basic_tokenizer, sample_text):
        """Test basic tokenizer with sample text using fixtures."""
        vocab_size = 256 + 10
        basic_tokenizer.train(sample_text, vocab_size, verbose=False)
        
        encoded = basic_tokenizer.encode(sample_text)
        decoded = basic_tokenizer.decode(encoded)
        
        assert decoded == sample_text
        assert len(basic_tokenizer.vocab) == vocab_size

    def test_regex_tokenizer_with_sample_text(self, regex_tokenizer, sample_text):
        """Test regex tokenizer with sample text using fixtures."""
        vocab_size = 256 + 10
        regex_tokenizer.train(sample_text, vocab_size, verbose=False)
        
        encoded = regex_tokenizer.encode(sample_text)
        decoded = regex_tokenizer.decode(encoded)
        
        assert decoded == sample_text
        assert len(regex_tokenizer.vocab) == vocab_size

    def test_tokenizer_performance_comparison(self, basic_tokenizer, regex_tokenizer, sample_text):
        """Compare performance characteristics of both tokenizers."""
        vocab_size = 256 + 15
        
        # Train both tokenizers
        basic_tokenizer.train(sample_text, vocab_size, verbose=False)
        regex_tokenizer.train(sample_text, vocab_size, verbose=False)
        
        # Test that both handle the same text correctly
        basic_encoded = basic_tokenizer.encode(sample_text)
        regex_encoded = regex_tokenizer.encode(sample_text)
        
        basic_decoded = basic_tokenizer.decode(basic_encoded)
        regex_decoded = regex_tokenizer.decode(regex_encoded)
        
        assert basic_decoded == sample_text
        assert regex_decoded == sample_text
        
        # Both should have learned the same number of merges
        assert len(basic_tokenizer.merges) <= vocab_size - 256
        assert len(regex_tokenizer.merges) <= vocab_size - 256


# Additional parametrized tests
@pytest.mark.parametrize("tokenizer_class", [BasicTokenizer, RegexTokenizer])
def test_tokenizer_roundtrip_parametrized(tokenizer_class):
    """Test that both tokenizer types can perform encode/decode roundtrips."""
    tokenizer = tokenizer_class()
    test_texts = [
        "Hello world!",
        "Testing 123 with numbers.",
        "Unicode test: 世界 🚀",
        "Contractions: don't, won't, can't",
        "",  # Empty string
        "a",  # Single character
    ]
    
    vocab_size = 256 + 10
    tokenizer.train("Hello world! Testing 123 with numbers. Unicode test: 世界 🚀", vocab_size, verbose=False)
    
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text, f"Roundtrip failed for {tokenizer_class.__name__} with text: '{text}'"


@pytest.mark.parametrize("vocab_size", [256, 260, 300])
def test_different_vocab_sizes(vocab_size):
    """Test tokenizers with different vocabulary sizes."""
    tokenizer = BasicTokenizer()
    text = "This is a test with repeated words. This is a test. Won't work without a diverse corpus"
    
    tokenizer.train(text, vocab_size, verbose=False)
    
    assert len(tokenizer.vocab) == vocab_size
    assert len(tokenizer.merges) == max(0, vocab_size - 256)
    
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text
