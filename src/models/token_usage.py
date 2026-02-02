from dataclasses import dataclass


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0  # Tokens written to cache
    cache_read_input_tokens: int = 0  # Tokens read from cache
    model_name: str = None
    turn_name: str = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_creation_input_tokens

    @property
    def total_cached_tokens(self) -> int:
        return self.cache_creation_input_tokens + self.cache_read_input_tokens
