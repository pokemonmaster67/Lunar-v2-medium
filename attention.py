import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )

class LunarAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.sliding_window = config.sliding_window
        
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings
        )
        
        self.dropout = nn.Dropout(config.attention_dropout_prob)
        self.use_flash_attention = config.use_flash_attention

    def _rotary_embed(self, q, k, seq_len):
        cos, sin = self.rotary_emb(q, seq_len)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def _rotate_half(self, x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length = hidden_states.shape[:2]
        
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        
        query = query.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        key = key.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        value = value.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        
        query, key = self._rotary_embed(query, key, seq_length)
        
        if self.use_flash_attention and torch.cuda.is_available():
            try:
                from flash_attn import flash_attn_func
                output = flash_attn_func(
                    query.transpose(1, 2),
                    key.transpose(1, 2),
                    value.transpose(1, 2),
                    dropout_p=self.dropout.p if self.training else 0.0,
                    softmax_scale=1.0 / math.sqrt(self.head_dim),
                    causal=True
                )
            except ImportError:
                output = self._vanilla_attention(query, key, value, attention_mask)
        else:
            output = self._vanilla_attention(query, key, value, attention_mask)
            
        output = output.reshape(batch_size, seq_length, self.hidden_size)
        output = self.out(output)
        return output

    def _vanilla_attention(self, query, key, value, attention_mask):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores = scores + attention_mask
            
        if self.sliding_window:
            window_mask = torch.ones_like(scores)
            window_mask = torch.triu(window_mask, diagonal=self.sliding_window)
            window_mask = torch.tril(window_mask, diagonal=-self.sliding_window)
            scores = scores.masked_fill(window_mask.bool(), float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        return torch.matmul(attn_weights, value)
