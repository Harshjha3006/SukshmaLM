A BPE tokenizer for tokenizing text data for consumption by LLMs. 


### Running Tests

Before running the tests, make sure there are 2 tokenizer configs saved as "sample1" and "sample2" with vocab_sizes 300 and 500 respectively, special_tokens as "|<im_start>|" and "|<im_end>|" respectively so that tests pass successfully. 
To run the test suite, use the following command from the project root directory:

```bash
pytest tests/
```

This will run all tests in the `tests` directory. For more detailed output, you can use:

```bash
pytest -v tests/
```

## Usage

### Training the Tokenizer

To train the tokenizer on your text data, use the following command:

```bash
python tokenizer/tokenizer.py --vocab_size <VOCAB_SIZE> --special_tokens <TOKEN1> <TOKEN2> ... --file_path <TRAINING_FILE_PATH> --config_name <CONFIG_NAME> --eos_token <EOS_TOKEN> [--verbose]
```

Parameters:
- `--vocab_size`: Total number of tokens in the vocabulary (must be ≥ 259 + number of user-provided special tokens), default value = 259
- `--special_tokens`: List of special tokens to be included in the vocabulary, default value = ["<|bos|>", eos_token] which are beginning of text and end of text tokens respectively  
- `--file_path`(required): Path to the training text file
- `--config_name`: Name of the tokenizer config, tokenizer-specific output files will be stored in the directory "tokenizer/{config_name}" and you can later load a specific tokenizer config as shown later in this readme, default value = "test"
- `--eos_token`: String representation of the end of text token you want to use for your tokenizer 
- `--verbose`: (Optional) Enable detailed training progress logs

Example:
```bash
python tokenizer/tokenizer.py --vocab_size 1000 --special_tokens "<|endoftext|>" "<|im_start|>" --file_path training_data.txt --config_name "sample_config" --eos_token="<|eos|>" --verbose
```

### Importing the Tokenizer

You can import the tokenizer class in your Python code in two ways:

1. If you have the repository in your project directory:
```python
from tokenizer.tokenizer import LLMTokenizer
```

2. If you want to use it as a standalone package, you can add the repository root to your Python path:
```python
import sys
sys.path.append('/path/to/SukshmaLM')  # Add the repository root to Python path
from tokenizer.tokenizer import LLMTokenizer
```


Here's a complete example of how to use the tokenizer in your Python code:

```python
from tokenizer.tokenizer import LLMTokenizer

# Initialize the tokenizer
tokenizer = LLMTokenizer()

# Load your desired config
tokenizer.load_config("my_config")

# Encode text
encoded = tokenizer.encode("Your text here")

# Decode tokens
decoded = tokenizer.decode(encoded)

# Print results
print(f"Encoded tokens: {encoded}")
print(f"Decoded text: {decoded}")
```


## How It Works

1. The tokenizer starts with a base vocabulary of 256 single-byte tokens
2. Special tokens are added to the vocabulary including padding token, bos and eos tokens 
3. During training, it:
   - Reads the input text file
   - Chunks the text using special tokens
   - Iteratively merges the most frequent token pairs until reaching the desired vocabulary size
4. The trained tokenizer can then encode text into tokens and decode tokens back into text

## Output Files

After training, the tokenizer generates the following files in the `tokenizer/{config_name}` directory:
- `merges.pkl`: Binary file containing the merge operations
- `tokenToByte.pkl`: Binary file mapping tokens to their byte representations
- `config.json`: A JSON file containing tokenizer's properties like vocab_size, special_tokens, etc. 
- `merges.json`: Human-readable version of merge operations
- `tokenToByte.json`: Human-readable version of token-to-byte mappings

## Notes

- The input text file must be UTF-8 encodable
- Special tokens must be unique, non-empty and non-overlapping
- Vocabulary size must be at least 259 + number of user-provided special tokens
- The tokenizer automatically creates a `tokenizer/{config_name}` directory to store its output files 