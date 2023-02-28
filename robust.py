import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GPT3(nn.Module):
  '''GPT3: This class implements the full GPT-3 model, which consists of a token embedding layer, 
  a positional encoding layer, multiple Transformer blocks, a pooling layer, a dropout layer, and a linear output layer. 
  The purpose of this model is to generate a probability distribution over the vocabulary for each token in an input sequence, 
  conditioned on the preceding tokens. The implementation in this class uses PyTorch modules to define each of the layers, 
  and includes a forward method that applies each layer in sequence to the input sequence and returns the final output logits.'''
    def __init__(self, num_tokens, emb_size, num_heads, num_layers, max_len=512, dropout_rate=0.1):
        super().__init__()
        
        self.token_embedding = nn.Embedding(num_tokens, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, max_len=max_len, dropout_rate=dropout_rate)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(emb_size, num_heads, dropout_rate=dropout_rate) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(emb_size)
        self.fc = nn.Linear(emb_size, num_tokens)
        
        self.max_len = max_len
        
    def forward(self, x):
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.ln_f(x[:, -1])
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
  '''PositionalEncoding: This class implements the positional encoding used in the Transformer architecture. 
  The purpose of this module is to add positional information to the input embeddings, 
  so that the Transformer can distinguish between tokens based on their position in the sequence. 
  The implementation in this class uses sine and cosine functions of different frequencies and phases 
  to create a fixed set of positional embeddings that are added to the input embeddings.'''
    def __init__(self, emb_size, max_len=512, dropout_rate=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        
        pos_enc = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pos_enc', pos_enc)
        
    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pos_enc.size(0):
            # Extend positional encoding if necessary
            pos_enc = self.pos_enc.repeat(seq_len // self.pos_enc.size(0) + 1, 1, 1)
            self.register_buffer('pos_enc', pos_enc)
        x = x + self.pos_enc[:seq_len, :]
        return self.dropout(x)
    
class TransformerBlock(nn.Module):
  '''TransformerBlock: This class implements a single Transformer block, 
  which consists of a multi-head attention layer followed by a position-wise feedforward layer. 
  The purpose of this block is to allow the model to capture complex interactions between tokens in the input sequence. 
  The implementation in this class applies layer normalization and residual connections around each of the two layers, 
  and also includes dropout and skip connections between the layers.'''
    def __init__(self, emb_size, num_heads, dropout_rate=0.1):
        super().__init__()
        
        self.multi_head_attention = MultiHeadAttention(emb_size, num_heads, dropout_rate=dropout_rate)
        self.ln1 = nn.LayerNorm(emb_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.GELU(),
            nn.Linear(4 * emb_size, emb_size)
        )
        self.ln2 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        residual = x
        x = self.multi_head_attention(x)
        x = self.dropout(self.ln1(x + residual))
        residual = x
        x = self.feed_forward(x)
        x = self.dropout(self.ln2(x + residual))
        return x
    
class MultiHeadAttention(nn.Module):
  '''MultiHeadAttention: This class implements the multi-head attention mechanism used in the Transformer architecture. 
  The purpose of this module is to allow the model to attend to different parts of the input sequence in parallel, 
  by splitting the input into multiple "heads" and computing the attention weights separately for each head. 
  The implementation in this class uses linear layers to project the input to separate "query", "key", and "value" representations for each head, 
  and then applies the scaled dot-product attention formula to compute the attention weights and values for each head.'''
    def __init__(self, emb_size, num_heads, dropout_rate=0.1):
      super().__init__()
    self.num_heads = num_heads
    self.head_size = emb_size // num_heads
    self.emb_size = emb_size
    self.q_linear = nn.Linear(emb_size, emb_size)
    self.k_linear = nn.Linear(emb_size, emb_size)
    self.v_linear = nn.Linear(emb_size, emb_size)
    self.fc = nn.Linear(emb_size, emb_size)
    self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        b, l, _ = x.size()
        q = self.q_linear(x).view(b, l, self.num_heads, self.head_size).transpose(1, 2)
        k = self.k_linear(x).view(b, l, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_linear(x).view(b, l, self.num_heads, self.head_size).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        x = torch.matmul(att, v).transpose(1, 2).contiguous().view(b, l, self.emb_size)
        x = self.fc(x)
        return x

