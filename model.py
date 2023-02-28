import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
  '''PositionalEncoding: This class implements the positional encoding used in the Transformer architecture. 
  The purpose of this module is to add positional information to the input embeddings, 
  so that the Transformer can distinguish between tokens based on their position in the sequence. 
  The implementation in this class uses sine and cosine functions of different frequencies and phases
  to create a fixed set of positional embeddings that are added to the input embeddings.'''
    def __init__(self, emb_size, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, emb_size) # initialize a tensor of zeros with shape (max_len, emb_size) to hold positional encodings
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # create a tensor of shape (max_len, 1) with values [0, 1, 2, ..., max_len - 1]
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size)) # calculate a tensor of shape (emb_size // 2,) with values that will be used to compute sine and cosine values for the positional encodings
        pe[:, 0::2] = torch.sin(position * div_term) # compute sine values for even indices in the last dimension of pe
        pe[:, 1::2] = torch.cos(position * div_term) # compute cosine values for odd indices in the last dimension of pe
        pe = pe.unsqueeze(0) # add a new dimension at the beginning of the tensor to represent the batch size
        self.register_buffer('pe', pe) # register the positional encoding tensor as a buffer so that it is saved and loaded with the model

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)] # add the positional encodings to the input embeddings
        return self.dropout(x) # apply dropout to the output


class TransformerBlock(nn.Module):
  '''TransformerBlock: This class implements a single Transformer block, which consists of a multi-head attention layer 
  followed by a position-wise feedforward layer. The purpose of this block is to allow the model 
  to capture complex interactions between tokens in the input sequence. 
  The implementation in this class applies layer normalization and residual connections around each of the two layers, 
  and also includes dropout and skip connections between the layers.'''
    def __init__(self, emb_size, num_heads, dropout_rate=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(emb_size, num_heads, dropout_rate) # create a multi-head attention module
        '''MultiHeadAttention: This class implements the multi-head attention mechanism used in the Transformer architecture. 
        The purpose of this module is to allow the model to attend to different parts of the input sequence in parallel, 
        by splitting the input into multiple "heads" and computing the attention weights separately for each head. 
        The implementation in this class uses linear layers to project the input to separate "query", "key", and "value" representations for each head,
        and then applies the scaled dot-product attention formula to compute the attention weights and values for each head.'''
        
        self.norm1 = nn.LayerNorm(emb_size) # create a layer normalization module
        self.norm2 = nn.LayerNorm(emb_size) # create a second layer normalization module for the residual connection after the feedforward layer
        self.dropout1 = nn.Dropout(dropout_rate) # create a dropout module
        self.dropout2 = nn.Dropout(dropout_rate) # create a second dropout module
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size), # create a linear layer that projects the input to a higher-dimensional space
            nn.ReLU(), # apply a ReLU activation function
            nn.Linear(4 * emb_size, emb_size), # create a second linear layer that projects the output back to the original dimensionality
            nn.Dropout(dropout_rate), # apply dropout to the output
        )

    def forward(self, x):
        att_output, _ = self.attention(x, x, x) # apply multi-head attention to the input
        x = self.norm1(x + self.dropout1(att_output)) # apply a residual connection with layer normalization and dropout to the attention output
        ff_output = self.feed_forward(x) # apply the feedforward layer to the output of the first residual connection
        x = self.norm2(x + self.dropout2(ff_output)) # apply a second residual connection with layer normalization and dropout to the output of the feedforward layer
        return x


class GPT3(nn.Module):
  '''GPT3: This class implements the full GPT-3 model, 
  which consists of a token embedding layer, a positional encoding layer, multiple Transformer blocks, 
  a pooling layer, a dropout layer, and a linear output layer. 
  The purpose of this model is to generate a probability distribution over the vocabulary for each token in an input sequence, 
  conditioned on the preceding tokens. The implementation in this class uses PyTorch modules to define each of the layers, 
  and includes a forward method that applies each layer in sequence to the input sequence and returns the final output logits.'''
    def __init__(self, vocab_size, emb_size, num_layers, num_heads, dropout_rate=0.1):
        super().__init__()
                self.token_emb = nn.Embedding(vocab_size, emb_size) # create an embedding module to convert token indices to embeddings
        self.pos_enc = PositionalEncoding(emb_size) # create a positional encoding module
        self.transformer_blocks = nn.ModuleList([TransformerBlock(emb_size, num_heads, dropout_rate) for _ in range(num_layers)]) # create a list of transformer blocks
        self.dropout = nn.Dropout(dropout_rate) # create a dropout module
        self.fc = nn.Linear(emb_size, vocab_size) # create a linear layer to project the final hidden state to the output vocabulary size

    def forward(self, x):
        token_emb = self.token_emb(x) # convert the input token indices to embeddings
        pos_enc = self.pos_enc(token_emb) # apply the positional encodings to the embeddings
        transformer_output = pos_enc # initialize the transformer output to the positional encodings
        for transformer_block in self.transformer_blocks: # apply each transformer block in sequence
            transformer_output = transformer_block(transformer_output)
        hidden_state = transformer_output # the final transformer output is the hidden state
        pooled_output = hidden_state.mean(dim=1) # compute the mean of the hidden state over the sequence length to get a pooled representation
        pooled_output = self.dropout(pooled_output) # apply dropout to the pooled representation
        logits = self.fc(pooled_output) # project the pooled representation to the output vocabulary size
        return logits
