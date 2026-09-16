class CharTokenizer:
    def __init__(self, text: str):
        """
        Build a character-level tokenizer from the input text.
        
        Args:
            text: A string used to build the vocabulary.
        """
        chars = sorted(set(text.lower()))
        self.stoi={c: i for i, c in enumerate(chars, start=2)}
        self.stoi["<BOS>"]=0
        self.stoi["<EOS>"]=1
        self.itos={i:c for i, c in enumerate(chars, start=2)}
        self.itos[0]="<BOS>"
        self.itos[1]="<EOS>"
        self.vocab_size=len(self.stoi)


    def encode(self, text: str) -> list:

        """
        Encode a string into a list of token indices.
        
        Args:
            text: The string to encode.
        Returns:
            List of integer indices.
        """
        res=[0]+[self.stoi[e] for e in list(text.lower())]+[1]
        return res

    def decode(self, indices: list) -> str:
        """
        Decode a list of token indices back into a string.
        
        Args:
            indices: List of integer indices.
        Returns:
            Decoded string.
        """
        res=[self.itos[e] for e in indices]
        return "".join(res)


