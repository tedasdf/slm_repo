#!/usr/bin/env python3
"""Train a SentencePiece (unigram) tokenizer on C4.

Streams allenai/c4 en split, collects sentences into a temp file,
then trains SentencePiece in-process and saves the .model file.

Usage:
    python src/slm/data/train_sentencepiece.py
    python src/slm/data/train_sentencepiece.py --vocab_size 32101 --num_sentences 5000000
    python src/slm/data/train_sentencepiece.py --output artifacts/tokenizer/my_sp/tokenizer.model
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size",      type=int, default=32101)
    parser.add_argument("--num_sentences",   type=int, default=5_000_000,
                        help="Sentences streamed from C4 for training")
    parser.add_argument("--output", type=str,
                        default="artifacts/tokenizer/c4_sentencepiece_32101/tokenizer.model")
    parser.add_argument("--model_type", type=str, default="unigram",
                        choices=["unigram", "bpe"])
    args = parser.parse_args()

    try:
        import sentencepiece as spm
    except ImportError:
        sys.exit("sentencepiece not installed — run: pip install sentencepiece")

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets not installed — run: pip install datasets")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Streaming allenai/c4 (en) …")
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True, trust_remote_code=True)

    total = 0
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                     encoding="utf-8") as fh:
        tmp_path = fh.name
        for example in ds:
            for line in example["text"].split("\n"):
                line = line.strip()
                if line:
                    fh.write(line + "\n")
                    total += 1
                    if total >= args.num_sentences:
                        break
            if total >= args.num_sentences:
                break
            if total % 500_000 == 0 and total > 0:
                print(f"  {total:,} / {args.num_sentences:,} sentences collected …")

    print(f"Collected {total:,} sentences → {tmp_path}")
    print(f"Training SentencePiece ({args.model_type}, vocab={args.vocab_size}) …")

    model_prefix = str(output_path.with_suffix(""))

    spm.SentencePieceTrainer.train(
        input=tmp_path,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=0.9995,
        # token ids: 0=<pad>, 1=<unk>, 2=</s>, no <s>
        pad_id=0,   pad_piece="<pad>",
        unk_id=1,   unk_piece="<unk>",
        bos_id=-1,                       # disabled
        eos_id=2,   eos_piece="</s>",
        num_threads=os.cpu_count() or 4,
        input_sentence_size=args.num_sentences,
        shuffle_input_sentence=True,
    )

    os.unlink(tmp_path)
    print(f"Saved: {output_path}")

    # Quick verification
    sp = spm.SentencePieceProcessor()
    sp.load(str(output_path))
    print(f"vocab_size : {sp.get_piece_size()}")
    print(f"eos_id     : {sp.eos_id()}")
    sample = "The quick brown fox jumps over the lazy dog."
    ids = sp.encode(sample)
    print(f"encode     : {ids}")
    print(f"decode     : {sp.decode(ids)}")


if __name__ == "__main__":
    main()
