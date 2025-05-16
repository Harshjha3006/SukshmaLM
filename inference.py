from inference_config import get_inference_config, LLMInferenceConfig
import torch 
from model.gpt import GPT
from tokenizer.tokenizer import LLMTokenizer
import random
import numpy as np 
import torch.nn.functional as F
import time

class LLM: 
    def __init__(self, config: LLMInferenceConfig): 
        """
        Main class for using an LLM in inference mode
        Args: 
            config (LLMInferenceConfig): Contains all the configuration options and hyperparameters required 
            to use an LLM in inference mode 
        Raises: 
            ValueError: if tokenizer and model's vocab_size don't match 
        """

        self.config = config 
        self.device = config.device
        self.model_path = config.model_path                 # path to model checkpoint
        self.max_new_tokens = config.max_new_tokens         # Maximum number of tokens llM can generate for one prompt
        self.tokenizer_config = config.tokenizer_config     # Specific Tokenizer used 
        self.prefix = config.prefix                         # Starting text that will be feeded into LLM for completion 
        self.topk = config.topk                             # The number of most likely tokens from which to sample the next token 

        # set the seed 
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Load the model from model path 
        self.model, model_training_config = GPT.from_checkpoint(self.model_path, self.device)
        self.context_len = model_training_config.context_len

        # set the model to evaluation mode 
        self.model.eval()

        # Load the tokenizer for the specific tokenizer config
        self.tokenizer = LLMTokenizer()
        self.tokenizer.load_config(self.tokenizer_config)

        tokenizer_vocab_size = self.tokenizer.vocab_size
        model_vocab_size = model_training_config.vocab_size

        if tokenizer_vocab_size != model_vocab_size: 
            raise ValueError("Tokenizer and Model's vocab_size should match, Use a different tokenizer")



    def generate(self): 

        """
        Generates completion based on text prefix provided through cmd line parser
        """

        # tokenize the prefix text and add a batch dimension 
        tokens = torch.tensor(self.tokenizer.encode(self.prefix), dtype = torch.long).unsqueeze(0) # (Batch, tokens_len)

        print()
        # print the initial text on screen 
        print(self.prefix, end = '', flush = True)

        # stores how much text has been printed 
        printed_len = len(self.prefix)

        # Keep generating tokens till max_new_tokens have not been generated 
        while tokens.shape[-1] < self.max_new_tokens:

            with torch.no_grad():  
            
                # get logits from LLM model 
                logits = self.model(tokens[:, -self.context_len:].to(self.device)) # (Batch, context_len, vocab_size)
                
                # get the logits for the next token to be generated 
                logits = logits[:, -1, :]   # (Batch, vocab_size)

                # Normalize the logits using softmax 
                probs = F.softmax(logits, dim = -1)  # (Batch, vocab_size)

                # Take out top_k most likely tokens 
                top_k_probs, top_k_idx = torch.topk(probs, self.topk, dim = -1) # (Batch, top_k)

                # Sample the next token to be generated from the top_k tokens 
                ix = torch.multinomial(top_k_probs, num_samples = 1)   # (Batch, 1)

                # Gather the indices of the sampled token 
                next_token = torch.gather(top_k_idx, -1, ix)   # (Batch, 1)

                # append the next token to the tokens list 
                tokens = torch.cat([tokens, next_token], dim = 1)  # (Batch, tokens_len)

                # Decode the tokens
                decoded_text = self.tokenizer.decode(tokens.squeeze(0).tolist())

                # print only the new token 
                print(decoded_text[printed_len:], end = '', flush= True)

                # update the printed_len 
                printed_len = len(decoded_text)

                
                time.sleep(0.1)


        print()
        print()
        

if __name__ == "__main__": 

    config = get_inference_config()

    llm = LLM(config)
    llm.generate()