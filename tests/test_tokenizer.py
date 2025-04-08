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
    with pytest.raises(TypeError) as excinfo: 
        LLMTokenizer(vocab_size="vocab",special_tokens=["endoftext"])
    assert str(excinfo.value) == "vocab_size must be an integer"

    # testing special_tokens list validation
    with pytest.raises(TypeError) as excinfo: 
        LLMTokenizer(vocab_size=500, special_tokens="special")
    assert str(excinfo.value) == "special_tokens must be a list"

    # testing special_tokens token type validation
    with pytest.raises(TypeError) as excinfo: 
        LLMTokenizer(vocab_size=500, special_tokens=[30,"hello"])
    assert str(excinfo.value) == "each token in special_tokens must be a str"

    # testing vocab_size value validation
    with pytest.raises(ValueError) as excinfo: 
        LLMTokenizer(vocab_size=257,special_tokens=["hello","hi"])
    assert str(excinfo.value) == "vocab_size must be >= 256 + num_special_tokens"

    # testing special tokens uniqueness validation 
    with pytest.raises(ValueError) as excinfo: 
        LLMTokenizer(vocab_size=300, special_tokens=["<|endoftext|>","<|endoftext|>"])
    assert str(excinfo.value) == "all special tokens must be unique"

    # testing special tokens non emptiness validation
    with pytest.raises(ValueError) as excinfo: 
        LLMTokenizer(vocab_size=300, special_tokens=["hello",""])
    assert str(excinfo.value) == "all special tokens must be non empty"

    # testing special tokens encoding validation 
    with pytest.raises(ValueError) as excinfo:
        LLMTokenizer(vocab_size=300,special_tokens=["hello", "\ud800"])
    assert str(excinfo.value) == "Special token \ud800 can't be encoded in utf-8"

    # testing special tokens overlap validation 
    with pytest.raises(ValueError) as excinfo:
        LLMTokenizer(vocab_size=500,special_tokens=["<|endoftext|>", "<|endoftext|>extra"])
    assert str(excinfo.value) == "Special tokens <|endoftext|> and <|endoftext|>extra overlap"


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
    with pytest.raises(ValueError) as excinfo: 
        bigram = tokenizer._get_most_freq_bigram([[1],[2]])
    assert str(excinfo.value) == "No bigrams were found in the input tokens"

    with pytest.raises(ValueError) as excinfo: 
        bigram = tokenizer._get_most_freq_bigram([[],[]])
    assert str(excinfo.value) == "No bigrams were found in the input tokens"


def test_replace_with_token_id(tokenizer): 
    tokens = [[1,2,1,3,1,2],[1,3,4,3,1,3]]
    new_tokens = tokenizer._replace_with_token_id(tokens,400,(1,3))
    assert new_tokens == [[1,2,400,1,2],[400,4,3,400]]

    new_tokens = tokenizer._replace_with_token_id([[1],[2]],400,(1,3))
    assert new_tokens == [[1],[2]]

    new_tokens = tokenizer._replace_with_token_id([[],[]],400,(1,2))
    assert new_tokens == [[],[]]


def test_train_validation(tokenizer): 
    with pytest.raises(TypeError) as excinfo:
        tokenizer.train(2)
    assert str(excinfo.value) == "input_file_path must be a str"

    with pytest.raises(ValueError) as excinfo:
        tokenizer.train("random.txt")
    assert str(excinfo.value) == "input file does not exist at the specified path"


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

def test_encode_validation(tokenizer): 
    with pytest.raises(TypeError) as excinfo: 
        tokenizer.encode(2)
    assert str(excinfo.value) == "Input text should be in string form"

def test_decode_validation(tokenizer): 

    # testing whether all input tokens are in the tokenizer vocab 
    with pytest.raises(ValueError) as excinfo: 
        tokenizer.decode([1,2000])
    assert str(excinfo.value) == f"Token id 2000 is out of valid range [0 - {tokenizer.vocab_size - 1}]"

    # testing whether tokens are a list
    with pytest.raises(TypeError) as excinfo: 
        tokenizer.decode(200)
    assert str(excinfo.value) == "tokens must be a list"

    # testing input tokens type 
    with pytest.raises(TypeError) as excinfo: 
        tokenizer.decode([200,"10"])
    assert str(excinfo.value) == "Each token must be an int"


def test_encode_decode(tokenizer,sample_text):

    encoded_tokens = tokenizer.encode(sample_text)
    assert tokenizer.decode(encoded_tokens) == sample_text