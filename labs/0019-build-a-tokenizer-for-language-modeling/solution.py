def train_tokenizer(corpus, vocab_size):
    """
    Character-level tokenizer.

    Args:
        corpus: list[str]
        vocab_size: maximum number of token IDs

    Returns:
        encode: str -> list[int]
        decode: list[int] -> str
    """

    # Collect every unique character appearing in the corpus
    chars = sorted(set("".join(corpus)))

    if len(chars) > vocab_size:
        raise ValueError(
            f"Corpus contains {len(chars)} unique characters, "
            f"but vocab_size is only {vocab_size}."
        )

    # character -> integer
    stoi = {ch: i for i, ch in enumerate(chars)}

    # integer -> character
    itos = {i: ch for ch, i in stoi.items()}

    def encode(text):
        return [stoi[ch] for ch in text]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    return encode, decode