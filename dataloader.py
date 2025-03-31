
"""
PseudoCode
2 million rows of text
each row can fit inside the context window of 1024

so each row will be given to the model at once 

let's say batch size is b 

then one batch will contain b rows , input to model - (b, num_tokens, embed_dim) -> (b, 1024, 1024)

endoftext will be appended to each row at the end, and the extra tokens will be treated as padding tokens. 

"""


import tiktoken 

def encode_text_to_tokens(text: str) -> list[int]: 
    """
    converts given text(in string form) to list of tokens 
    """ 

    

