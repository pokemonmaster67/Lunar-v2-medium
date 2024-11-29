from dataclasses import dataclass
from typing import Optional

@dataclass
class LunarConfig:
    """Configuration class for Lunar-v2-medium model"""
    
    vocab_size: int = 50257  # GPT-2 vocabulary size
    max_position_embeddings: int = 2048
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    max_sequence_length: int = 2048
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-5
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = True
    use_cache: bool = True
    use_flash_attention: bool = True
    sliding_window: int = 256
    rope_scaling: Optional[dict] = None
    
    @property
    def num_parameters(self) -> int:
        """Calculate approximate number of parameters"""
        embedding_params = self.vocab_size * self.hidden_size
        attention_params = self.num_hidden_layers * (
            4 * self.hidden_size * self.hidden_size +  # QKV projections + output
            2 * self.hidden_size  # Layer norms
        )
        ffn_params = self.num_hidden_layers * (
            2 * self.hidden_size * self.intermediate_size +  # FFN layers
            self.hidden_size + self.intermediate_size  # Biases
        )
        
        total_params = embedding_params + attention_params + ffn_params
        return total_params

    @classmethod
    def get_150m_config(cls):
        """Returns configuration for 150M parameter model"""
        return cls(
            hidden_size=1024,
            num_hidden_layers=14,
            num_attention_heads=16,
            intermediate_size=4096
        )
