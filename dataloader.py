import os
import torch 
from tokenizer.tokenizer import LLMTokenizer
from torch.utils.data import Dataset, DataLoader

# Class for encapsulating text data for the LLM

class LLMDataset(Dataset):
    def __init__(self, data_path: str,context_length: int,tokenizer_config: str): 
        """
        Initializes the LLMDataset 
        Args: 
            data_path (str): path to the raw dataset 
            context_length (int): Maximum number of tokens an LLM can process simultaneously, i.e. its context window 
            tokenizer_config (int): Name of the specific tokenizer config you want to use 
        Raises: 
            FileNotFoundError: if data_path is not a valid path
            UnicodeDecodeError: it is raised if the training data can't be decoded into utf-8
        """
        
        # Raising error if data_path is not valid
        if not os.path.isfile(data_path): 
            raise FileNotFoundError("Data does not exist at the specified path")
        
        # Reading raw text data
        try: 
            with open(data_path,'r',encoding = 'utf-8') as file: 
                self.raw_data = file.read()
        except UnicodeDecodeError: 
            raise UnicodeDecodeError("text couldn't be decoded into utf-8")
        

        # Loading tokenizer 
        self.tokenizer = LLMTokenizer()
        self.tokenizer.load_config(tokenizer_config)

        # Encoding the raw data into tokens
        self.data = self.tokenizer.encode(self.raw_data)
        # Store context length
        self.context_length = context_length
        # block_size represents one training example
        self.block_size = context_length + 1
        
        # Calculate padding needed if len(data) does not equally divided by block_size 
        padding_needed = (self.block_size - (len(self.data) % self.block_size)) % self.block_size
        if padding_needed > 0:
            self.data = self.data + ([self.tokenizer.PAD_TOKEN_ID] * padding_needed)


    def __len__(self) -> int: 
        """
        Returns the total size of the dataset
        """
        return len(self.data) // self.block_size


    def __getitem__(self, ind: int) -> tuple: 
        """
        Returns a data element at a specific index
        Args:
            ind (int): Index of the dataset to return
        Returns:
            tuple: (input_sequence, target_sequence)
        """
        # Calculate start and end indices
        start_idx = ind * self.block_size
        end_idx = start_idx + self.block_size
        
        # Get the input sequence
        x = torch.tensor(self.data[start_idx:end_idx - 1])

        y = torch.tensor(self.data[start_idx + 1:end_idx])
        
        return x, y


def get_dataloader(config): 

    # initializing the dataset 
    llmDataset = LLMDataset(config.training_data_path, config.context_len, config.tokenizer_config)
    # wrapping the dataset inside a dataloader and returning it 
    return DataLoader(llmDataset, batch_size=config.batch_size, shuffle= True, pin_memory=True,drop_last=True)