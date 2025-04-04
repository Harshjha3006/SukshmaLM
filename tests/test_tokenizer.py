import pytest
from tokenizer.tokenizer import LLMTokenizer
import re

@pytest.fixture
def sample_text(): 
    return "Helloworld!<|endoftext|>Mynameis<|im_start|>harshjha."

@pytest.fixture
def tokenizer(): 
    return LLMTokenizer(vocab_size=300,special_tokens=["<|endoftext|>","<|im_start|>"])


def test_init_validations():

    # testing vocab_size type validation
    with pytest.raises(TypeError,match = "vocab_size must be an integer"): 
        LLMTokenizer(vocab_size="vocab",special_tokens=["endoftext"])

    # testing special_tokens list validation
    with pytest.raises(TypeError,match = "special_tokens must be a list"): 
        LLMTokenizer(vocab_size=500, special_tokens="special")

    # testing special_tokens token type validation
    with pytest.raises(TypeError,match = "each token in special_tokens must be a str"): 
        LLMTokenizer(vocab_size=500, special_tokens=[30,"hello"])

    # testing vocab_size value validation
    with pytest.raises(ValueError,match = re.escape("vocab_size must be >= 256 + num_special_tokens")): 
        LLMTokenizer(vocab_size=257,special_tokens=["hello","hi"])

    # testing special tokens uniqueness validation 
    with pytest.raises(ValueError,match = "all special tokens must be unique"): 
        LLMTokenizer(vocab_size=300, special_tokens=["<|endoftext|>","<|endoftext|>"])

    # testing special tokens non emptiness validation
    with pytest.raises(ValueError,match = "all special tokens must be non empty"): 
        LLMTokenizer(vocab_size=300, special_tokens=["hello",""])

    # testing special tokens encoding validation 
    with pytest.raises(ValueError,match = "Special token \ud800 can't be encoded in utf-8"):
        LLMTokenizer(vocab_size=300,special_tokens=["hello", "\ud800"])


def test_chunk_test(tokenizer,sample_text): 
    tokens = tokenizer._chunk_text(sample_text)
    assert len(tokens) == 3
    assert all(isinstance(chunk,list) for chunk in tokens)
    assert all(isinstance(token,int) for chunk in tokens for token in chunk)

    chunks = tokenizer._chunk_text("")
    assert chunks == []

    chunks = tokenizer._chunk_text("<|endoftext|><|im_start|>")
    assert chunks == []

def test_get_most_freq_bigram(tokenizer): 
    tokens = [[1,2,1,3,1,2],[1,3,4,3,1,3]]
    bigram = tokenizer._get_most_freq_bigram(tokens)
    assert bigram == (1,3)
    with pytest.raises(ValueError,match = "No bigrams were found in the input tokens"): 
        bigram = tokenizer._get_most_freq_bigram([[1],[2]])

    with pytest.raises(ValueError,match = "No bigrams were found in the input tokens"): 
        bigram = tokenizer._get_most_freq_bigram([[],[]])


def test_replace_with_token_id(tokenizer): 
    tokens = [[1,2,1,3,1,2],[1,3,4,3,1,3]]
    new_tokens = tokenizer._replace_with_token_id(tokens,400,(1,3))
    assert new_tokens == [[1,2,400,1,2],[400,4,3,400]]

    new_tokens = tokenizer._replace_with_token_id([[1],[2]],400,(1,3))
    assert new_tokens == [[1],[2]]

    new_tokens = tokenizer._replace_with_token_id([[],[]],400,(1,2))
    assert new_tokens == [[],[]]


def test_train_validation(tokenizer): 
    with pytest.raises(TypeError,match = "input_file_path must be a str"):
        tokenizer.train(2)

    with pytest.raises(ValueError,match = "input file does not exist at the specified path"):
        tokenizer.train("random.txt")


def test_special_token_handling(tokenizer): 
    assert tokenizer.special_token_idMap["<|endoftext|>"] == 256
    assert tokenizer.special_token_idMap["<|im_start|>"] == 257

    assert tokenizer.tokenToByte[256].decode('utf-8') == "<|endoftext|>"
    assert tokenizer.tokenToByte[257].decode('utf-8') == "<|im_start|>"



def test_train(tokenizer): 
    train_file = "data/sample.txt"
    tokenizer.train(train_file)

    assert len(tokenizer.merges) == tokenizer.vocab_size - (256 + len(tokenizer.special_tokens)) 
    assert len(tokenizer.tokenToByte) == tokenizer.vocab_size

    # validate merged tokens
    for bigram, token_id in tokenizer.merges.items(): 
        assert token_id in tokenizer.tokenToByte
        assert bigram[0] in tokenizer.tokenToByte
        assert bigram[1] in tokenizer.tokenToByte

        byte_rep = tokenizer.tokenToByte[token_id]
        
        expected_byte_rep = tokenizer.tokenToByte[bigram[0]] + tokenizer.tokenToByte[bigram[1]]

        assert byte_rep == expected_byte_rep

    # validate special tokens
    for i,token in enumerate(tokenizer.special_tokens): 
        token_id = 256 + i
        assert token_id in tokenizer.tokenToByte
        assert tokenizer.tokenToByte[token_id].decode("utf-8") == token

    # valid single byte tokens
    for i in range(256): 
        assert i in tokenizer.tokenToByte
        assert tokenizer.tokenToByte[i] == bytes([i])
