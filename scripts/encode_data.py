from tokenizer.tokenizer import LLMTokenizer
import argparse
import os
import pickle

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",type = str,required=True,help = "path of the input text file")
    parser.add_argument("--output_dir",type = str,required = True, help = "path of the output directory where tokens will be stored")
    parser.add_argument("--tokenizer_config",type = str, required=True,help = "name of tokenizer config you want to use")
    parser.add_argument("--context_len", type = int, required=True, help = "context length of the LLM")
    args, _ = parser.parse_known_args()
    args_dict = vars(args)

    # Loading the input file 
    if not os.path.isfile(args_dict["input_file"]): 
        raise FileNotFoundError("Input file not found at specified path")
    
    print("Reading input data ...")
    try: 
        with open(args_dict["input_file"],'r',encoding = 'utf-8') as f: 
            text = f.read()
    except UnicodeDecodeError: 
        raise UnicodeDecodeError("Input file couldn't be decoded by utf-8")
    
    print("Encoding Data ...")
    # Loading the tokenizer and encoding the input text
    tokenizer = LLMTokenizer()
    tokenizer.load_config(args_dict["tokenizer_config"])
    tokens = tokenizer.encode(text)

    # Adding padding tokens if required
    block_size = args_dict["context_len"] + 1
    padding_needed = (block_size - (len(tokens) % block_size)) % block_size
    if padding_needed > 0:
        tokens = tokens + ([tokenizer.PAD_TOKEN_ID] * padding_needed)

    # storing the tokens in the output_dir 
    os.makedirs(args_dict["output_dir"],exist_ok=True)
    input_file_name = os.path.splitext(os.path.basename(args_dict["input_file"]))[0]
    output_file = os.path.join(args_dict["output_dir"],f"{input_file_name}_tokens.pkl")

    tokenized_data = {
        "vocab_size": tokenizer.vocab_size, 
        "data": tokens
    }

    with open(output_file,'wb') as file: 
        pickle.dump(tokenized_data, file)

    print(f"Tokenized Data Serialized to disk at {output_file}")        

