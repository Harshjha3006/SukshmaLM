from inference_config import get_config, LLMInferenceConfig
import torch 
from model.gpt import GPT
from tokenizer.tokenizer import LLMTokenizer
import random
import numpy as np 
import torch.nn.functional as F

class LLM: 
    def __init__(self, config: LLMInferenceConfig): 
        """
        Main class for using an LLM in inference mode
        Args: 
            config (LLMInferenceConfig): Contains all the configuration options and hyperparameters required 
            to use an LLM
        """

        self.config = config 
        self.device = config.device
        self.model_path = config.model_path 
        self.max_new_tokens = config.max_new_tokens
        self.tokenizer_config = config.tokenizer_config 
        self.prefix = config.prefix 
        self.context_len = config.context_len 

        # set the seed 
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Load the model from model path 
        self.model = GPT.from_checkpoint(self.model_path, self.device)

        # set the model to evaluation model 
        self.model.eval()

        # Load the tokenizer for the specific tokenizer config
        self.tokenizer = LLMTokenizer()
        self.tokenizer.load_config(self.tokenizer_config)



    def generate(self): 

        tokens = torch.tensor(self.tokenizer.encode(self.prefix),dtype = torch.long).unsqueeze(0)

        print()
        print("Language Output: ")
        print(self.tokenizer.decode(tokens.squeeze(0).tolist()), end = '', flush = True)

        last_written = tokens.shape[-1]

        while tokens.shape[-1] < self.max_new_tokens:

            with torch.no_grad():  
            
                logits = self.model(tokens[:, -self.context_len:].to(self.device))

                logits = logits[:, -1, :]

                probs = F.softmax(logits, dim = -1)  # (Batch, vocab_size)

                top_k, top_k_idx = torch.topk(probs, 50, dim = -1) # (Batch, vocab_size)

                ix = torch.multinomial(top_k, num_samples= 1)   # (Batch, 1)

                res_tokens = torch.gather(top_k_idx, -1, ix)   # (Batch, 1)

                tokens = torch.cat([tokens, res_tokens], dim = 1)  # (Batch, tokens_len)

                decoded_text = self.tokenizer.decode(tokens.squeeze(0).tolist())

                print(decoded_text[last_written:], end = '', flush= True)

                last_written = tokens.shape[-1]


        print()
        print()
        

if __name__ == "__main__": 

    config = get_config()

    llm = LLM(config)
    llm.generate()