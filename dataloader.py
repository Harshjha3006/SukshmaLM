import os
import torch 
from torch.utils.data import Dataset, DataLoader
import pickle

# Class for encapsulating text data for the LLM

class LLMDataset(Dataset):
    def __init__(self, data_path: str,context_length: int): 
        """
        Initializes the LLMDataset 
        Args: 
            data_path (str): path to the tokenized dataset 
            context_length (int): Maximum number of tokens an LLM can process simultaneously, i.e. its context window 
        Raises: 
            FileNotFoundError: if data_path is not a valid path
        """
        
        # Raising error if data_path is not valid
        if not os.path.isfile(data_path): 
            raise FileNotFoundError("Data does not exist at the specified path")
        
        # Reading tokenized data 
        with open(data_path, 'rb') as file: 
            self.data = pickle.load(file)

        # block_size represents the size of one training example
        self.block_size = context_length + 1

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
    llmDataset = LLMDataset(config.training_data_path, config.context_len)
    # wrapping the dataset inside a dataloader and returning it 
    return DataLoader(llmDataset, batch_size=config.batch_size, shuffle=True, pin_memory=True,drop_last=True)