
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


class SentencePieceTokenizer:
    def __init__(self, path: Path) -> None:
        import sentencepiece as spm
        self._path = path
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(str(path))

    def encode(self, s: str) -> list[int]:
        return self._sp.encode(s, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self._sp.decode(ids)

    def token_to_id(self, token: str) -> int:
        idx = self._sp.piece_to_id(token)
        unk = self._sp.unk_id()
        if idx == unk and self._sp.id_to_piece(unk) != token:
            raise ValueError(f"Token not found in tokenizer vocab: {token}")
        return idx

    @property
    def vocab_size(self) -> int:
        return self._sp.get_piece_size()

    @property
    def eos_id(self) -> int | None:
        eid = self._sp.eos_id()
        return eid if eid >= 0 else None

    @classmethod
    def load(cls, path: Path) -> "SentencePieceTokenizer":
        return cls(path)

    def save(self, path: Path) -> None:
        import shutil
        shutil.copy2(self._path, path)


AnyTokenizer = BPETokenizer | SentencePieceTokenizer
