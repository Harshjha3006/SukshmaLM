import os
import json 
import re
import argparse
import pickle

class LLMTokenizer: 

    def __init__(self,vocab_size: int, special_tokens: list[str],config_name: str = "test", verbose: bool = False): 
        """
        Initializes the tokenizer class 
        Token Arrangement ->  0 - 255                                           (single byte tokens)
                              256 - (256 + num_special_tokens - 1)              (special tokens)
                              (256 + num_special_tokens) - (vocab_size - 1)     (merged tokens created during tokenization)
          
        Args: 
            vocab_size (int): The vocabulary size i.e. total number of tokens of the tokenizer 
            special_tokens (list[str]): list of special_tokens in string form , eg -> ["<|endoftext|>", "<|im_start|>"]
            config_name (str): name of the saved config for this specific trained tokenizer
            verbose (bool): Used to enable detailed training logs 
        Raises: 
            ValueError: if vocab_size is too small or special tokens are invalid or they overlap 
            TypeError: if vocab_size is not an integer or special tokens is not a list of strings or config_name is not a str
        """

        # Type validations 
        if not isinstance(vocab_size,int): 
            raise TypeError("vocab_size must be an integer")
        if not isinstance(special_tokens,list):
            raise TypeError("special_tokens must be a list")
        if not all(isinstance(token, str) for token in special_tokens):
            raise TypeError("each token in special_tokens must be a str")
        if not isinstance(config_name, str): 
            raise TypeError("config_name must be a str")
    

        # Value validations
        # if vocab_size = 256 + len(special_tokens) then no merges will be performed and it will be equivalent
        # to a single byte level tokenizer with special tokens 
        if vocab_size < 256 + len(special_tokens):
            raise ValueError("vocab_size must be >= 256 + num_special_tokens")
        if len(special_tokens) != len(set(special_tokens)):
            raise ValueError("all special tokens must be unique")
        if any(not token for token in special_tokens):
            raise ValueError("all special tokens must be non empty")
        for i,token in enumerate(special_tokens):
            try: 
                token.encode("utf-8")
            except UnicodeEncodeError:
                raise ValueError(f"Special token {token} can't be encoded into utf-8")
            for token2 in special_tokens[i + 1:]:
                if token in token2 or token2 in token: 
                    raise ValueError(f"Special tokens {token} and {token2} overlap")
            

        # initializations
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.config_name = config_name
        self.verbose = verbose

        # path of directory where output files of the tokenizer would be stored
        self.storage_dir = os.path.join("tokenizer", self.config_name)
        # config_file will store the tokenizer's state 
        self.config_file_path = os.path.join(self.storage_dir,"config.json")

        # merges dict stores the mapping from a bigram of token ids to a new token id 
        self.merges = {}
        self.merges_store_path = os.path.join(self.storage_dir,"merges.pkl")
        self.pretty_merges_store_path = os.path.join(self.storage_dir,"merges.json")

        # tokenToByte stores the byte representation of each token 

        # compute the byte representation of all single byte tokens 
        self.tokenToByte = {i : bytes([i]) for i in range(256)}
        self.tokenToByte_store_path = os.path.join(self.storage_dir,"tokenToByte.pkl")
        self.pretty_tokenToByte_store_path = os.path.join(self.storage_dir,"tokenToByte.json")

        # special_token_idMap stores the token ids of the special tokens
        self.special_token_idMap = {}
        for i,token in enumerate(self.special_tokens):
            self.special_token_idMap[token] = 256 + i
            self.tokenToByte[self.special_token_idMap[token]] = token.encode('utf-8') 

    def train(self, input_file_path: str): 
        """
        Trains the LLM tokenizer based on the training data provided in the input file  
        Args: 
            input_file_path (str): path of the training data 
        Raises: 
            TypeError: if input_file_path is not a str
            ValueError: if the input file does not exist at the specified input path or if the training data can't be decoded into utf-8
        """

        print("Performing Validations ...")
        if not isinstance(input_file_path,str):
            raise TypeError("input_file_path must be a str")
        if not os.path.isfile(input_file_path):
            raise ValueError("input file does not exist at the specified path")

        print("Reading training data ...")
        # reading the input file and validating if it can be decoded into utf-8
        try: 
            with open(input_file_path,'r',encoding = 'utf-8') as f: 
                text = f.read()
        except UnicodeDecodeError: 
            raise ValueError("Input text file could not be decoded into utf-8")

        print("Chunking text ...")        
        # chunk the text using special_tokens and encode the chunks using utf-8
        tokens = self._chunk_text(text)

        # set the curr_vocab_size
        curr_vocab_size = 256 + len(self.special_tokens)

        print("Starting to Merge tokens ...")
        # iterate till the required vocab_size has been achieved 
        while curr_vocab_size < self.vocab_size: 
            # get the most frequent token bigram from the chunks
            bigram = self._get_most_freq_bigram(tokens)

            # assign a new token id to that bigram
            new_token_id = curr_vocab_size

            # update the curr_vocab_size 
            curr_vocab_size += 1

            # replace the bigram with that new token id in the tokens list
            tokens = self._replace_with_token_id(tokens, new_token_id, bigram)

            # update the merges dict
            self.merges[bigram] = new_token_id

            # update the tokenToByte dict
            self.tokenToByte[new_token_id] = self.tokenToByte[bigram[0]] + self.tokenToByte[bigram[1]]

            # print progress if verbose is enabled
            if self.verbose:
                print(f"{self.tokenToByte[bigram[0]].decode('utf-8',errors='replace')} and {self.tokenToByte[bigram[1]].decode('utf-8',errors='replace')} got merged into {self.tokenToByte[new_token_id].decode('utf-8',errors='replace')}")
                print(f"vocab_size got updated to {curr_vocab_size}")
                print()

        print("Saving output files ...")
        # create storage_dir if not already created
        os.makedirs(self.storage_dir, exist_ok=True)
        # save the merges and tokenToByte dicts to disk 
        with open(self.merges_store_path,'wb') as f: 
            pickle.dump(self.merges,f)

        with open(self.tokenToByte_store_path,'wb') as f: 
            pickle.dump(self.tokenToByte,f)

        # save the tokenizer's state to config_file_path
        config = {
            "vocab_size": self.vocab_size, 
            "special_tokens": self.special_tokens, 
            "special_token_idMap" : self.special_token_idMap
        }

        with open(self.config_file_path, 'w') as f: 
            json.dump(config,f)

        if self.verbose:
            # also save their human readable form to disk for visualization

            # transform the merges and tokenToByte dicts for making them suitable for json format
            pretty_merges = {str(k) : v for k,v in self.merges.items()}
            pretty_tokenToByte = {k : v.decode('utf-8',errors = 'replace') for k,v in self.tokenToByte.items()}
            
            with open(self.pretty_merges_store_path, 'w') as f: 
                json.dump(pretty_merges,f)

            with open(self.pretty_tokenToByte_store_path,'w') as f: 
                json.dump(pretty_tokenToByte,f)

        print("Training Completed ...")


    def load_config(self,config_name: str):
        """
        Loads a specific tokenizer config
        Args: 
            config_name (str): name of the config 
        Raises: 
            TypeError: if config name is not a str
            ValueError: if tokenizer's specific files are not found at their specified path 
        """

        if not isinstance(config_name,str): 
            raise TypeError("config_name must be a str")

        self.storage_dir = os.path.join("tokenizer", config_name)
        self.config_name = config_name
        self.merges_store_path = os.path.join(self.storage_dir,"merges.pkl")
        self.tokenToByte_store_path = os.path.join(self.storage_dir,"tokenToByte.pkl")
        self.config_file_path = os.path.join(self.storage_dir,"config.json")

        
        if not os.path.isfile(self.config_file_path): 
            raise ValueError(f"tokenizer config file not found at {self.config_file_path}")
        if not os.path.isfile(self.merges_store_path): 
            raise ValueError(f"tokenizer's merges dict file not found at {self.merges_store_path}")
        if not os.path.isfile(self.tokenToByte_store_path):
            raise ValueError(f"tokenizer's tokenToByte dict file not found at {self.tokenToByte_store_path}")
        
        with open(self.config_file_path, 'r') as f:
            config = json.load(f)

        self.vocab_size = config["vocab_size"]
        self.special_tokens = config["special_tokens"]
        self.special_token_idMap = config["special_token_idMap"]

        with open(self.merges_store_path,'rb') as f: 
            self.merges = pickle.load(f)

        with open(self.tokenToByte_store_path, 'rb') as f: 
            self.tokenToByte = pickle.load(f)


    def encode(self,text: str) -> list[int]: 
        """
        Encodes the given text (in string form) into a list of token ids 
        Args: 
            text (str): input text in string form
        Returns: 
            a list of token ids representing the encoded text
        Raises: 
            TypeError if input text is not in string form
        """
        # input validation 
        if not isinstance(text,str): 
            raise TypeError("Input text should be in string form")
        
        # regex pattern to divide text into chunks by special_tokens
        pattern = '|'.join(map(re.escape,self.special_tokens))

        # find the special tokens in the text
        matches = list(re.finditer(pattern, text))

        # starting index of current chunk 
        start = 0

        # output tokens list
        tokens = []

        # iterate over chunks
        for match in matches: 
            # indices of special_token [special_start,special_end)
            special_start,special_end = match.span()

            # encode the current chunk and append it to the output tokens list 
            self._encode_chunk(text[start:special_start],tokens)

            # append the special token 
            tokens.append(self.special_token_idMap[match.group()])

            # update start to move on the next chunk 
            start = special_end

        # encode the last chunk 
        if start < len(text): 
            self._encode_chunk(text[start:],tokens)


        return tokens
    
    def decode(self,tokens: list[int]) -> str: 
        """
        Decodes a list of numerical tokens back into string form 
        Args: 
            tokens (list[int]): the list of input tokens
        Returns: 
            Decoded form of the tokens
        Raises: 
            TypeError if the tokens is not a list of ints
            ValueError if tokens are out of valid range 
        """

        # input validation 
        if not isinstance(tokens,list): 
            raise TypeError("tokens must be a list")
        if not all(isinstance(token,int) for token in tokens): 
            raise TypeError("Each token must be an int")
        for token in tokens: 
            if not (0 <= token < self.vocab_size):
                raise ValueError(f"Token id {token} is out of valid range [0 - {self.vocab_size - 1}]")
                

        byte_string = b''.join(self.tokenToByte[token] for token in tokens)

        return byte_string.decode('utf-8',errors='replace')

    def _encode_chunk(self,chunk: str, tokens: list[int]):
        """
        Encodes an individual chunk (which does not contain special tokens)

        Args: 
            chunk (str): individual chunk to be encoded
            tokens (list[int]): list of tokens in which the encoded tokens are to be appended
        """    

     
        # first encode the chunk using utf-8 
        try: 
            chunk_tokens = list(chunk.encode('utf-8'))
        except UnicodeEncodeError: 
            raise ValueError("Input text can't be encoded into utf-8")

        # simply append the encoded first token only if the chunk's length is 1
        if len(chunk_tokens) == 1: 
            tokens.append(chunk_tokens[0])
            return 
        elif not chunk_tokens: 
            return
        
        # merge the bigrams in increasing order of their token ids 
        while True: 
            bigram = self._get_bigram_with_min_token_id(chunk_tokens)
            if bigram not in self.merges: 
                break
            # taking the element at 0th index because the below method returns a list of lists
            chunk_tokens = self._replace_with_token_id([chunk_tokens],self.merges[bigram],bigram)[0]
        
        tokens.extend(chunk_tokens)
                
    
    def _get_bigram_with_min_token_id(self,tokens: list[int]) -> tuple[int,int]: 
        """
        Returns a bigram from the text which has the lowest token id in the merges dict
        Args: 
            tokens (list[int]): list of tokens
        Returns: 
            bigram with the lowest token id in the merges dict
        """

        if len(tokens) < 2: 
            return (-1,-1)

        bigram_set = set()
        for i in range(len(tokens) - 1): 
            bigram_set.add((tokens[i],tokens[i + 1]))

        return min(bigram_set,key = lambda bigram : self.merges.get(bigram,float('inf')))

    def _replace_with_token_id(self, tokens: list[list[int]], token_id: int, bigram: tuple[int,int]) -> list[list[int]]: 

        """
        Replaces the bigram with the provided token id and returns a new tokens list
        Args: 
            tokens (list[list[int]]): the original list of tokens
            token_id (int) : the token_id which will replace the provided bigram
            bigram (tuple[int,int]): the bigram to be replaced 
        Returns: 
            returns a new list of tokens with the replaced token id 
        """

        new_tokens = []

        for chunk in tokens: 
            new_chunk_tokens = []
            i = 0 
            while i < len(chunk): 
                if i < len(chunk) - 1 and chunk[i] == bigram[0] and chunk[i + 1] == bigram[1]: 
                    new_chunk_tokens.append(token_id)
                    i += 2
                else: 
                    new_chunk_tokens.append(chunk[i])
                    i += 1
            new_tokens.append(new_chunk_tokens)
        
        return new_tokens

    def _get_most_freq_bigram(self, tokens: list[list[int]]) -> tuple[int,int]: 
        """
        Returns the most frequent token bigram in the tokens list
        Args: 
            tokens (list[list[int]]): the list of input tokens 
        Returns: 
            the most frequent token bigram
        Raises: 
            ValueError: if no bigrams are found in the input tokens
        """

        freq_map = {}
        for chunk in tokens: 
            for i in range(len(chunk) - 1): 
                pair = (chunk[i],chunk[i + 1])
                freq_map[pair] = freq_map.get(pair,0) + 1

        # raise error if no bigram was found
        if not freq_map: 
            raise ValueError("No bigrams were found in the input tokens")

        return max(freq_map,key = freq_map.get)
    
    def _chunk_text(self,text: str) -> list[list[int]]: 
        """
        Divides the text into various chunks with the help of special tokens 
        Args: 
            text (str): the input text 
        Returns: 
            a list of chunks where each chunk contains a list of tokens encoded using utf-8
        """
        
        # define the regex pattern used for splitting text
        pattern = '|'.join(map(re.escape,self.special_tokens))

        # divide the text into chunks 
        chunks = re.split(pattern, text)

        tokens = []

        for chunk in chunks: 
            # ignore empty chunks
            if not chunk: 
                continue
            tokens.append(list(chunk.encode('utf-8')))

        return tokens


        
        
            
if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size",type = int, required=True,help = "vocab_size of the vocabulary")
    parser.add_argument("--special_tokens",nargs = '+',required = True, help = "list of special tokens in string form")
    parser.add_argument("--file_path",type = str, required=True, help = "path of training text file")
    parser.add_argument("--verbose", action="store_true",help = "will print more information related to training process")
    parser.add_argument("--config_name",type = str, required=True, help = "name of this specific tokenizer")
    args, _ = parser.parse_known_args()
    
    # transform args to dict form
    args = vars(args)

    tokenizer = LLMTokenizer(args["vocab_size"], args["special_tokens"],args["config_name"],args["verbose"])
    tokenizer.train(args["file_path"])
