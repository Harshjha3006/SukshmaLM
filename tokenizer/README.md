A BPE tokenizer for tokenizing text data for consumption by LLMs. 

## Installation

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/Harshjha3006/SukshmaLM.git
cd SukshmaLM
```

### 2. Install Dependencies

The tokenizer requires Python 3.x. While most dependencies are from Python's standard library, you'll need to install pytest for running tests.

#### Using pip

```bash
pip install -r requirements.txt
```
or you can manually install the required package

```bash
pip install pytest==8.3.5
```

Standard library packages used (no installation needed):
- os
- json
- re
- argparse
- pickle

### 3. Running Tests

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
python tokenizer/tokenizer.py --vocab_size <VOCAB_SIZE> --special_tokens <TOKEN1> <TOKEN2> ... --file_path <TRAINING_FILE_PATH> --config_name <CONFIG_NAME> [--verbose]
```

Parameters:
- `--vocab_size`: Total number of tokens in the vocabulary (must be ≥ 256 + number of special tokens), default value = 256
- `--special_tokens`: List of special tokens to be included in the vocabulary, default value = None 
- `--file_path`(required): Path to the training text file
- `--config_name`: Name of the tokenizer config, tokenizer-specific output files will be stored in the directory "tokenizer/{config_name}" and you can later load a specific tokenizer config as shown later in this readme, default value = "test"
- `--verbose`: (Optional) Enable detailed training progress logs

Example:
```bash
python tokenizer/tokenizer.py --vocab_size 1000 --special_tokens "<|endoftext|>" "<|im_start|>" --file_path training_data.txt --config_name "sample_config" --verbose
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
2. Special tokens are added to the vocabulary
3. During training, it:
   - Reads the input text file
   - Chunks the text using special tokens (if any special_tokens are specified)
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
- Vocabulary size must be at least 256 + number of special tokens
- The tokenizer automatically creates a `tokenizer/{config_name}` directory to store its output files 