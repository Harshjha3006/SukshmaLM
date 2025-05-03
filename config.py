import torch
import argparse 


# The main config class for training LLMs. It contains various important properties regarding LLM training 

class LLMTrainerConfig: 

    def __init__(self, **kwargs): 
        """
        Initializing various properties related to LLM Training 
        """

        # setting the device type, 'cuda:0' or 'cpu'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # setting the config properties parsed through the command line parser
        for key,value in kwargs.items(): 
            setattr(self,key, value)


# method for getting the config properties from the user via command line 

def get_config(**optional_kwargs): 

    # defining the parser
    parser = argparse.ArgumentParser()

    # define various command line arguements 

    parser.add_argument("--training_data_path", type=str, required=True,help = "path to the training data")
    parser.add_argument("--batch_size",type = int, default = 512, help = "batch size for training")
    parser.add_argument("--context_len", type = int, default=1024, help = "context window size of the LLM")
    parser.add_argument("--verbose", action="store_true", help = "Enables detailed training logs")

    # parsing the known args
    args, _ = parser.parse_known_args()
    # converting args to dict form
    kwargs = vars(args)

    # adding any optional kwargs 
    kwargs.update(optional_kwargs)

    # return the LLM config 
    return LLMTrainerConfig(**kwargs)