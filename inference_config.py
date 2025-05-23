import torch
import argparse 


# The main config class for LLM inference. It contains various important configuration options required for LLM inference

class LLMInferenceConfig: 

    def __init__(self, **kwargs): 
        """
        Config class for LLM inference
        """

        # setting the device type, 'cuda:0' or 'cpu'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # setting the config properties parsed through the command line parser
        for key, value in kwargs.items(): 
            setattr(self, key, value)


# method for getting the config properties from the user via command line 

def get_inference_config(**optional_kwargs): 

    # defining the parser
    parser = argparse.ArgumentParser()

    # define various command line arguements 

    parser.add_argument("--model_path", type=str, required=True, help = "path to the model checkpoint")
    parser.add_argument("--max_new_tokens", type=int, default=512, help = "max number of tokens that the LLM can generate for one prompt")
    parser.add_argument("--tokenizer_config",type = str, required=True, help = "The tokenizer used should have the same vocab_size as the one used in model's architecture")
    parser.add_argument("--prefix", type = str, default = "Hi, I'm a language model", help = "Starting text that will be feeded into the model")
    parser.add_argument("--topk", type = int, default = 50, help = "Number of Most likely tokens from which to sample the next token")
    parser.add_argument("--seed", type = int, default = 248, help = "Seed value for better reproducibility")
    parser.add_argument("--skip_special_tokens", action = 'store_true', help = "if set, special tokens will not be printed")

    # parsing the known args
    args, _ = parser.parse_known_args()
    # converting args to dict form
    kwargs = vars(args)

    # adding any optional kwargs 
    kwargs.update(optional_kwargs)

    # return the LLM config 
    return LLMInferenceConfig(**kwargs)