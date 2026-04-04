
from pathlib import Path
from tokenizers import Tokenizer


class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tk = tokenizer
        vocab = tokenizer.get_vocab()
        self.stoi = vocab
        self.itos = {i: tok for tok, i in vocab.items()}

    def encode(self, s: str) -> list[int]:
        return self.tk.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=True)

    def token_to_id(self, token: str) -> int:
        token_id = self.tk.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Token not found in tokenizer vocab: {token}")
        return token_id

    @property
    def vocab_size(self) -> int:
        return self.tk.get_vocab_size()

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: Path) -> None:
        self.tk.save(str(path))
