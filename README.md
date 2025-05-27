# SukshmaLM

SukshmaLM (where "Sukshma" is a Hindi word which means "very small") is a lightweight implementation of a decoder-only transformer language model, inspired by the GPT-2 architecture. This project provides code for training and inference of language models with a focus on simplicity and educational value. <br>This project also includes code for training a Byte pair Encoding Tokenizer from scratch and using it for encoding and decoding. More info about the tokenizer is available at [tokenizer/README.md](tokenizer/README.md). <br>
<br>

## Features

- Decoder-only transformer architecture
- Customizable model parameters (embedding dimension, number of layers, attention heads, etc.)
- Training with configurable hyperparameters
- Inference with various generation strategies (top-k sampling, temperature control, skip_special_tokens)
- TensorBoard integration for training monitoring
- Checkpoint saving and loading


## Installation

1. Clone the repository:
```bash
git clone https://github.com/Harshjha3006/SukshmaLM.git
cd SukshmaLM
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Steps to follow for training your own LLM and using it 

- Choose a text file as training data for training your tokenizer, train it on the data and save it. More Details at [tokenizer/README.md](tokenizer/README.md)
- Choose a text file as training data for training your LLM, tokenize it using your trained tokenizer. More Details at [Encoding data](#encoding-data)
- Train your own LLM using the tokenized data (a pickle file), set appropriate hyperparameters. More Details at [Training](#training)
- Generate completions using the trained LLM. More Details at [Inference](#inference)

## Project Structure

- `model/`: Contains the transformer model implementation
- `data/`: Directory for training data for both the model and the tokenizer (You can place your training data here)
- `checkpoints/`: Saved model checkpoints
- `tensorboard_logs/`: Training logs for visualization
- `scripts/`: Contains a utility script for encoding a text file into a pkl file containing its tokenized version. More info at [Encoding Data](#encoding-data)
- `tests/`: Unit tests (There are only tests for the tokenizer for now)
- `tokenizer/`: Custom tokenizer implementation (see [tokenizer/README.md](tokenizer/README.md) for details)

## Usage

### Training And Usage of Tokenizer 

See [tokenizer/README.md](tokenizer/README.md) for more information 

### Encoding Data

To encode your text data into tokens for training:

```bash
python -m scripts.encode_data --input_file path/to/text/file --output_dir path/to/output/directory --tokenizer_config name_of_tokenizer_config
```

Key encoding parameters:
- `--input_file`: Path to the input text file to be tokenized (required)
- `--output_dir`: Directory where the tokenized data will be saved (required)
- `--tokenizer_config`: Name of the tokenizer config to use for encoding (required)

The script will create a pickle file containing the tokenized data in the specified output directory. The output file will be named `{input_filename}_tokens.pkl` and will contain both the tokenized data and the vocabulary size.

### Training

To train the model, use the following command:

```bash
python trainer.py --training_data_path path/to/data --vocab_size 50000 [other options]
```

Key training parameters:
- `--training_data_path`: Path to the tokenized training data (it expects a single pickle file containing tokenized data) (required)
- `--batch_size`: Batch size for training (default: 512)
- `--context_len`: Context window size (default: 1024)
- `--embed_dim`: Embedding dimension (default: 512)
- `--vocab_size`: Vocab size of the model and the tokenizer used to create the tokenized data (required)
- `--num_layers`: Number of decoder blocks (default: 6)
- `--num_heads`: Number of attention heads (default: 4)
- `--num_epochs`: Number of training epochs (default: 5)
- `--lr`: Base or Max Learning rate (default: 1e-3)
- `--warmup_steps`: Number of warmup steps for the cosine lr scheduler (default: 100)
- `--l2reg`: L2 regularization rate (default: 1e-3)
- `--exp_name`: name of the experiment or training run, model checkpoints will be saved in the checkpoints/{exp_name} folder (default: "test")
- `--seed`: Seed value for better reproducibility (default: 248)
- `--logging_steps`: Loss will be logged every {logging_steps} steps (default: 100)
- `--eval_steps`: Model's current best state will be saved every {eval_steps} steps (default: 500)

This will create a model checkpoint in checkpoints/{exp_name}/model.pth and also save few of the model's important properties in checkpoints/{exp_name}/config.json. <br> TensorBoard logs will be stored at tensorboard_logs/{exp_name} folder.

### Inference

To generate text using a trained model:

```bash
python inference.py --model_path path/to/checkpoint --tokenizer_config name_of_tokenizer_config [other options]
```

Key inference parameters:
- `--model_path`: Path to the model's .pth file (required) 
- `--max_new_tokens`: Maximum number of new tokens to generate (default: 512)
- `--tokenizer_config`: Name of the saved tokenizer config you will use to decode the generated text, it should have the same vocab_size as the model. For more information on this see [tokenizer/README.md](tokenizer/README.md). (required)
- `--prefix`: Starting text for generation (default: "Hi, I'm a language model")
- `--topk`: Number of top tokens to sample from (default: 50)
- `--temperature`: Controls randomness in generation (default: 1.0)
- `--skip_special_tokens`: bool flag which if set will not print special tokens to the screen
- `--seed`: Seed value for better reproducibility (default: 248)


## Monitoring Training

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir tensorboard_logs/{exp_name}
```

## License

This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.
