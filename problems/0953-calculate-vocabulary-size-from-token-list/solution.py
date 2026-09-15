def vocab_size(tokens, special_tokens=None):
    """Return the vocabulary size from a list of preprocessed tokens.

    Args:
        tokens: list[str] of preprocessed tokens
        special_tokens: optional list[str] of special tokens to include
    Returns:
        int: number of unique tokens in the combined vocabulary
    """
    st=set()
    for e in tokens:
        st.add(e)
    if special_tokens is not None:
        for e in special_tokens:
            st.add(e)
    return len(st)