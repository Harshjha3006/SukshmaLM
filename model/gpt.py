import torch 
import torch.nn as nn
import torch.nn.functional as F

class MaskedSelfAttention(nn.Module): 
    
    def __init__(self,embed_dim,num_heads,dropout_rate,context_len): 
        """
        Initializes a Multi Head Masked Self Attention Block 
        """

        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads

        # query, key and value matrices 
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Attention mask to prevent tokens from interacting with future tokens
        self.register_buffer("tril", torch.tril(torch.ones(context_len,context_len)))

        # Dropout layer for regularization 
        self.dropout = nn.Dropout(dropout_rate)

        # Projection layer for projecting into the residual pathway 
        self.proj = nn.Linear(embed_dim, embed_dim)


    def forward(self, x): 
        
        Batch, Context_len, Embed_dim = x.shape 

        # Computing the key, query and value vectors for each token embedding 

        k = self.key(x) # (Batch, Context_len, Embed_dim)
        q = self.query(x) # (Batch, Context_len, Embed_dim)
        v = self.value(x) # (Batch, Context_len, Embed_dim)

        # Reshape the vectors so that multiple heads are created 
        k = torch.permute(k.view(Batch, Context_len, self.num_heads, self.head_dim),(0,2,1,3)) # (Batch, Num_heads, Context_len, Head_dim)
        q = torch.permute(q.view(Batch, Context_len, self.num_heads, self.head_dim),(0,2,1,3)) # (Batch, Num_heads, Context_len, Head_dim)
        v = torch.permute(v.view(Batch, Context_len, self.num_heads, self.head_dim),(0,2,1,3)) # (Batch, Num_heads, Context_len, Head_dim)

        # Compute scaled attention matrices for each head 
        # Attention Matrices -> (Batch, Num_heads, Context_len, Context_len)    
        attention = torch.matmul(q, k.transpose(-1,-2)) / (Embed_dim ** 0.5)

        # Mask the attention matrix so that tokens don't attend to their future tokens 
        attention = torch.masked_fill(attention, self.tril[:Context_len, :Context_len] == 0, float('-inf'))

        # compute softmax for normalization 
        attention = F.softmax(attention, dim = -1)

        # add dropout layer for regularization
        # will randomly shut off communications between some tokens 
        attention = self.dropout(attention)

        # Compute output by doing matrix product of attention with value vectors 
        # (Batch, Num_heads, Context_len, Context_len) with (Batch, Num_heads, Context_len, Head_dim) = (Batch, Num_heads, Context_len, Head_dim)
        output = torch.matmul(attention, v)

        # reshape and permute the output 
        output = torch.permute(output, (0,2,1,3))
        output = output.reshape(Batch, Context_len, Embed_dim)

        # project the output to the residual pathway 
        output = self.proj(output) # (Batch, Context_len, Embed_dim)

        return output 



class FeedForward(nn.Module): 
    def __init__(self,embed_dim,dropout_rate): 
        """
        Initializes a Feed forward layer 
        """

        super().__init__()

        self.fflayer = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),  # expand the dimensions for increasing expressiveness of model
            nn.ReLU(),                            # non linearity 
            nn.Linear(4 * embed_dim, embed_dim),  # projection into residual pathway 
            nn.Dropout(dropout_rate)              # dropout for regularization 
        )   


    def forward(self, x): 

        return self.fflayer(x)



class TransformerBlock(nn.Module): 
    def __init__(self,embed_dim, num_heads, context_len, dropout_rate):
        """
        Initializes a single Transformer Block 
        2 Main components -> - MaskedSelfAttention 
                          -> - FeedForwardLayer 
        """

        super().__init__()

        # masked attention layer
        self.attention = MaskedSelfAttention(embed_dim, num_heads, dropout_rate, context_len)

        # feed forward layer 
        self.ffd = FeedForward(embed_dim, dropout_rate)

        # layer norm layers 
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x): 

        x = x + self.attention(self.ln1(x))

        x = x + self.ffd(self.ln2(x))

        return x

    


# a GPT style decoder only transformer
class GPT(nn.Module): 

    def __init__(self,config): 
        """
        Initializes all the layers and hyperparameters 
        """

        super().__init__()
        self.embed_dim = config.embed_dim
        self.vocab_size = config.vocab_size
        self.context_len = config.context_len  
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.dropout_rate = config.dropout_rate


        # Initial Embedding Layer
        # (Batch, context_len, vocab_size) -> (Batch, context_len, embed_dim)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)

        # Positional Encoding layer 
        # an embedding for every relative position in the context window 
        self.positional_encoding = nn.Embedding(self.context_len, self.embed_dim)


        # Multiple Transformer Blocks 
        # (Batch, context_len, embed_dim) -> (Batch, context_len, embed_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(self.embed_dim, self.num_heads,self.context_len,self.dropout_rate)
                                      for _ in range(self.num_layers)])

        # Reverse Embedding 
        # (Batch, context_len, embed_dim) -> (Batch, context_len, vocab_size)
        # Logits are generated for every token which will then used to compute cross entropy loss
        self.reverse_embedding = nn.Linear(self.embed_dim, self.vocab_size)
        # Weight sharing with initial embedding layer
        self.reverse_embedding.weight = self.embedding.weight

    



    def forward(self, x): 

        # x of shape (Batch, Context_len)
        Batch, Context_len = x.shape

        # Convert tokens to their embeddings -> (Batch, Context_len, Embed_dim)
        x = self.embedding(x)

        # Add Positional Encodings to the Embeddings 
        x = x + self.positional_encoding(torch.arange(0,Context_len))

        # Feed the input vectors x through the Transformer Blocks 
        # output shape after all layers -> (Batch, Context_len, Embed_dim)
        x = self.blocks(x)

        # Compute logits over the vocab 
        # logits -> (Batch, Context_len, vocab_size)
        logits = self.reverse_embedding(x)

        # return logits for computing loss 
        return logits 



