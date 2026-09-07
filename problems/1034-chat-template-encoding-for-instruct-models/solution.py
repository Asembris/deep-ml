def encode_chat(user_message: str, system_message: str = None) -> list:
    """
    Build a Llama-3-style chat template token sequence.

    Args:
        user_message: the user's message (string)
        system_message: optional system message (string). If None, use
            "You are a helpful assistant.".

    Returns:
        A list of integer token IDs.
    """
    if system_message is None:
        system_message = "You are a helpful assistant."

    def encode(text):
        return [ord(c) for c in text]

    def header(role):
        return [2] + encode(role) + [3] + encode("\n\n")

    tokens = [1]

    tokens += header("system")
    tokens += encode(system_message)
    tokens += [4]

    tokens += header("user")
    tokens += encode(user_message)
    tokens += [4]

    tokens += header("assistant")

    return tokens