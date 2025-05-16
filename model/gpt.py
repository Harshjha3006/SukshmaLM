import torch 
import torch.nn as nn
import torch.nn.functional as F
from trainer_config import LLMTrainerConfig

class MaskedSelfAttention(nn.Module): 
    
    def __init__(self, embed_dim: int, num_heads: int, context_len: int, dropout_rate: float): 
        """
        Initializes a Multi Head Masked Self Attention Block

        Args: 
            embed_dim (int): embedding dimension of the input 
            num_heads (int): number of attention heads 
            context_len (int): size of LLM's context window 
            dropout_rate (int): dropout rate for the dropout layer 
        """

        # Initializing parent class 
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # computing embed_dim for each head i.e. head_dim
        self.head_dim = embed_dim // num_heads

        # query, key and value matrices 
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Attention mask of shape (context_len, context_len) to prevent tokens from interacting with future tokens
        self.register_buffer("tril", torch.tril(torch.ones(context_len,context_len)))

        # Dropout layer for regularization 
        self.dropout = nn.Dropout(dropout_rate)

        # Projection layer for projecting the results into the residual pathway 
        self.proj = nn.Linear(embed_dim, embed_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor: 
            
        # Unpacking dimensions of input tensor 
        Batch, Context_len, Embed_dim = x.shape 

        # Computing the key, query and value vectors for each token embedding 
        
        # keys give info related "what a token offers"
        k = self.key(x) # (Batch, Context_len, Embed_dim)
        # queries give info related to "what a token is looking for"
        q = self.query(x) # (Batch, Context_len, Embed_dim)
        # values give related to "what a token will actually communicate with the other token" 
        v = self.value(x) # (Batch, Context_len, Embed_dim)

        # Reshape the vectors so that multiple heads are created 
        k = torch.permute(k.view(Batch, Context_len, self.num_heads, self.head_dim),(0,2,1,3)) # (Batch, Num_heads, Context_len, Head_dim)
        q = torch.permute(q.view(Batch, Context_len, self.num_heads, self.head_dim),(0,2,1,3)) # (Batch, Num_heads, Context_len, Head_dim)
        v = torch.permute(v.view(Batch, Context_len, self.num_heads, self.head_dim),(0,2,1,3)) # (Batch, Num_heads, Context_len, Head_dim)

        # Compute scaled attention matrices for each head 
        # Compute affinity between tokens 
        attention = torch.matmul(q, k.transpose(-1,-2)) / (Embed_dim ** 0.5) # (Batch, Num_heads, Context_len, Context_len)

        # Mask the attention matrix so that tokens don't attend to future tokens 
        attention = torch.masked_fill(attention, self.tril[:Context_len, :Context_len] == 0, float('-inf')) # (Batch, Num_heads, Context_len, Context_len)

        # compute softmax for normalization 
        attention = F.softmax(attention, dim = -1) # (Batch, Num_heads, Context_len, Context_len)

        # add dropout layer for regularization
        # will randomly shut off communications between some tokens 
        attention = self.dropout(attention)

        # Compute output by doing matrix product of attention with value vectors 
        # output will contain token embeddings which have incorporated information from other tokens in the context window
        output = torch.matmul(attention, v) # (Batch, Num_heads, Context_len, Head_dim)

        # reshape and permute the output so that all heads get concatenated for each token
        output = torch.permute(output, (0,2,1,3)) # (Batch, Context_len, Num_heads, Head_dim)
        output = output.reshape(Batch, Context_len, Embed_dim) # (Batch, Context_len, Embed_dim)

        # project the output to the residual pathway 
        output = self.proj(output) # (Batch, Context_len, Embed_dim)

        return output 



class FeedForward(nn.Module): 
    def __init__(self, embed_dim: int, dropout_rate: float): 
        """
        Initializes a Feed forward layer 
        Args: 
            embed_dim (int): embedding dimension of the input vector 
            dropout_rate (float): dropout rate for the dropout layer 
        """

        # Initializing the parent class 
        super().__init__()

        self.fflayer = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),  # expand the dimensions for increasing expressiveness of model
            nn.ReLU(),                            # non linearity 
            nn.Linear(4 * embed_dim, embed_dim),  # projection into residual pathway 
            nn.Dropout(dropout_rate)              # dropout for regularization 
        )   


    def forward(self, x: torch.Tensor) -> torch.Tensor: 

        return self.fflayer(x)



class TransformerBlock(nn.Module): 
    def __init__(self, embed_dim: int, num_heads: int, context_len: int, dropout_rate: float, num_layers: int):
        """
        Initializes a single Transformer Block 
        2 Main components -> - MaskedSelfAttention 
                          -> - FeedForwardLayer 

        Args: 
            embed_dim (int): embedding dimension of input vectors 
            num_heads (int): number of multi head attention units 
            context_len (int): size of the context window of the LLM 
            dropout_rate (float): dropout rate for the dropout layer 
            num_layers (int): number of transformer blocks 
        """

        # Initializing parent class 
        super().__init__()

        # masked self attention block 
        self.attention = MaskedSelfAttention(embed_dim, num_heads, context_len, dropout_rate)

        # feed forward layer 
        self.ffd = FeedForward(embed_dim, dropout_rate)

        # layer norm layers 
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # residual scaling factor 
        self.res_scaling_factor = 1 / (num_layers ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 

        x = x + self.res_scaling_factor * self.attention(self.ln1(x))

        x = x + self.res_scaling_factor * self.ffd(self.ln2(x))

        return x

    


# a GPT style decoder only transformer
class GPT(nn.Module): 

    def __init__(self, config: LLMTrainerConfig): 
        """
        A Decoder-only Transformer model based on the GPT-2 Architecture  
        """

        # initializing the parent class  
        super().__init__()

        self.embed_dim = config.embed_dim
        self.vocab_size = config.vocab_size
        self.context_len = config.context_len  
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.dropout_rate = config.dropout_rate


        # Initial Embedding Layer
        # (Batch, context_len) -> (Batch, context_len, embed_dim)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)

        # Positional Encoding layer 
        # an embedding for every relative position in the context window 
        self.positional_encoding = nn.Embedding(self.context_len, self.embed_dim)
        self.register_buffer("pos_tensor", torch.arange(self.context_len))

        # Multiple Transformer Blocks 
        # (Batch, context_len, embed_dim) -> (Batch, context_len, embed_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(self.embed_dim, self.num_heads, self.context_len, self.dropout_rate, self.num_layers)
                                      for _ in range(self.num_layers)])

        # Reverse Embedding 
        # (Batch, context_len, embed_dim) -> (Batch, context_len, vocab_size)
        # Logits are generated for every token which will then be used to compute cross entropy loss
        self.reverse_embedding = nn.Linear(self.embed_dim, self.vocab_size)
        # Weight sharing with the initial embedding layer
        self.reverse_embedding.weight = self.embedding.weight

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str):
        """
        Initialize a GPT model from a checkpoint file.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file (.pth)
            device (str): Device to load the model on ('cuda:0' or 'cpu')
            
        Returns:
            model (GPT): Initialized model with loaded weights
        """
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Get the config (LLMTrainingConfig) from checkpoint
        config = checkpoint['config']
        
        # Create a new model instance
        model = cls(config)
        
        # Load the model state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move model to specified device
        model = model.to(device)
        
        return model, config

    def forward(self, x: torch.Tensor) -> torch.Tensor:  

        # x of shape (Batch, Context_len)
        Batch, Context_len = x.shape

        # Convert tokens to their embeddings 
        x = self.embedding(x) # (Batch, Context_len, Embed_dim)

        # Add Positional Encodings to the Embeddings 
        x = x + self.positional_encoding(self.pos_tensor) # (Batch, Context_len, Embed_dim)

        # Feed the input vectors x through the Transformer Blocks 
        x = self.blocks(x) # (Batch, Context_len, Embed_dim)

        # Compute logits over the vocabulary
        logits = self.reverse_embedding(x) # (Batch, Context_len, vocab_size)

        # return logits for computing loss 
        return logits 



