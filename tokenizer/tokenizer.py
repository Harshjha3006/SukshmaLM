"""
PseudoCode

encoding 

    create chunks of input text 
    encoded_chunks = []
    for each chunk 
        if len(chunk) < 2 
            encoded_chunks.add(list(chunk.encode("utf-8"))[0])
            continue
        get the pair from merges dict that has the least token id and is present in the chunk 
        while you can get this pair 
            replace this pair by the corresponding token id in the merges dict 

        chunk.add(endoftext_token)
        encoded_chunks.add(chunk)

    flatten the encoded chunks
    return them

decoding 

    bytes_rep = b''.join([vocab[idx] for idx in input_tokens])
    return bytes_rep.decode("utf-8",error = "replace")

"""
import os
import json 
import re
import argparse
import pickle

class LLMTokenizer: 

    def __init__(self,vocab_size: int, special_tokens: list[str],verbose: bool = False): 
        """
        Initializes the tokenizer class 
        Token Arrangement ->  0 - 255                                           (single byte tokens)
                              256 - (256 + num_special_tokens - 1)              (special tokens)
                              (256 + num_special_tokens) - (vocab_size - 1)     (merged tokens formed during tokenization)
          
        Args: 
            vocab_size (int): The vocabulary size i.e. total number of tokens of the tokenizer 
            special_tokens (list[str]): list of special_tokens in string form , eg -> ["<|endoftext|>", "<|im_start|>"]
        Raises: 
            ValueError: if vocab_size is too small or special tokens are invalid
            TypeError: if vocab_size is not an integer or special tokens is not a list of strings 
        """

        # Type validations 
        if not isinstance(vocab_size,int): 
            raise TypeError("vocab_size must be an integer")
        if not isinstance(special_tokens,list):
            raise TypeError("special_tokens must be a list")
        if not all(isinstance(token, str) for token in special_tokens):
            raise TypeError("each token in special_tokens must be a str")
    

        # Value validations
        # if vocab_size = 256 + len(special_tokens) then no merges will be performed and it will be equivalent
        # to a char level tokenizer with special tokens 
        if vocab_size < 256 + len(special_tokens):
            raise ValueError("vocab_size must be >= 256 + num_special_tokens")
        if len(special_tokens) != len(set(special_tokens)):
            raise ValueError("all special tokens must be unique")
        if any(not token for token in special_tokens):
            raise ValueError("all special tokens must be non empty")
        for token in special_tokens:
            try: 
                token.encode("utf-8")
            except UnicodeEncodeError:
                raise ValueError(f"Special token {token} can't be encoded in utf-8")
            

        # initializations
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.verbose = verbose

        # merges dict stores the mapping from a bigram of token ids to a new token id 
        self.merges = {}
        self.merges_store_path = "tokenizer/merges.pkl"
        self.pretty_merges_store_path = "tokenizer/merges.json"

        # tokenToByte stores the byte representation of each token 
        # compute the byte representation of all single byte tokens 
        self.tokenToByte = {i : bytes([i]) for i in range(256)}
        self.tokenToByte_store_path = "tokenizer/tokenToByte.pkl"
        self.pretty_tokenToByte_store_path = "tokenizer/tokenToByte.json"

        # special_token_ids stores the token ids of the special tokens
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
            ValueError: if the input file does not exist at the specified input path 
        """

        if not isinstance(input_file_path,str):
            raise TypeError("input_file_path must be a str")
        if not os.path.isfile(input_file_path):
            raise ValueError("input file does not exist at the specified path")

        with open(input_file_path,'r',encoding = 'utf-8') as f: 
            text = f.read()

        # chunk the text using special_tokens and encode the chunks using utf-8
        tokens = self._chunk_text(text)

        # set the curr_vocab_size
        curr_vocab_size = 256 + len(self.special_tokens)

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

        # save the merges and tokenToByte dicts to disk 
        with open(self.merges_store_path,'wb') as f: 
            pickle.dump(self.merges,f)

        with open(self.tokenToByte_store_path,'wb') as f: 
            pickle.dump(self.tokenToByte,f)

        if self.verbose:
            # also save their human readable form to disk for visualization

            # transform the merges and tokenToByte dicts for making them suitable for json format
            pretty_merges = {str(k) : v for k,v in self.merges.items()}
            pretty_tokenToByte = {k : v.decode('utf-8',errors = 'replace') for k,v in self.tokenToByte.items()}
            
            with open(self.pretty_merges_store_path, 'w') as f: 
                json.dump(pretty_merges,f)

            with open(self.pretty_tokenToByte_store_path,'w') as f: 
                json.dump(pretty_tokenToByte,f)


    def _replace_with_token_id(self, tokens: list[list[int]], token_id: int, bigram: tuple[int,int]) -> list[list[int]]: 

        """
        Replaces the bigram with the provided token id 
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
    parser.add_argument("--special_tokens",nargs = '+',help = "list of special tokens in string form")
    parser.add_argument("--file_path",type = str, required=True, help = "path of training text file")
    parser.add_argument("--verbose", action="store_true",help = "will print more information related to training process")
    args, _ = parser.parse_known_args()
    
    # transform args to dict form
    args = vars(args)

    tokenizer = LLMTokenizer(args["vocab_size"], args["special_tokens"],verbose = args["verbose"])
    tokenizer.train(args["file_path"])
