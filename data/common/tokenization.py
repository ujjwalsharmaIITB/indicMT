"""Source-side tagging and SentencePiece encoding, shared by every stage that feeds the model.

Training data: the data pipeline writes each source as `tag_source(tag, text)` ("<xx> text",
data/bpcc-*/create_multilingual_dataset.py), and training/encode_spm.py encodes those whole lines
with `sp.encode(lines)`. Everything that feeds the model later (evaluation sources, CTranslate2
evaluation and serving) must produce exactly the same pieces, so it calls `encode_sources` /
`encode_source` here, which build the same tagged line and make the same call.

With the 128k model's `add_dummy_prefix` (SentencePiece's default) and every `<xx>` tag a
user-defined piece, a tagged line always encodes as ['▁', '<xx>', '▁word', ...]: a lone '▁'
first, then the tag. `encode_sources` checks that prefix. Prepending the tag to `sp.encode(text)`
instead drops that '▁', which gives the model a source format it never saw in training.

deployment/stage_common.py copies this file into deployment/api/common/ for the Docker build
context; keep it dependency-free (it only calls the SentencePieceProcessor it is given).
"""

PREFIX = "▁"


def tag_source(tag: str, text: str) -> str:
    """The tagged source line, as written into the training data: '<xx> text'."""
    return f"{tag} {text}"


def encode_sources(sp, tag: str, texts: list, check: bool = True) -> list:
    """Encode raw source sentences for target tag `tag` (e.g. '<hi>') exactly as the training
    sources were encoded. Returns one list of pieces per sentence, each starting ['▁', tag]."""
    pieces = sp.encode([tag_source(tag, t) for t in texts], out_type=str)
    if check:
        for p, t in zip(pieces, texts):
            if p[:2] != [PREFIX, tag]:
                raise ValueError(f"source does not start with the training prefix {[PREFIX, tag]}: "
                                 f"{p[:3]} for {t[:40]!r}")
    return pieces


def encode_source(sp, tag: str, text: str, check: bool = True) -> list:
    """Single-sentence form of encode_sources."""
    return encode_sources(sp, tag, [text], check)[0]
