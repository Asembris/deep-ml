import numpy as np

class DropoutLayer:
    def __init__(self, p: float):
        """Initialize the dropout layer.
        
        Attributes to set:
            self.p: the dropout rate
            self.mask: stores the dropout mask (initially None)
        """
        # Your code here
        self.p=p 
        self.mask=None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of the dropout layer.
        
        Generate a new mask on each training forward pass and store it in self.mask.
        """
        # Your code here
        if not training:
            return x

        self.mask = np.random.binomial(
            1, 1 - self.p, size=x.shape
        ) / (1 - self.p)

        return x * self.mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass of the dropout layer.
        
        Use the stored self.mask from the most recent forward pass.
        """
        # Your code here
        if self.mask is None:
            return grad

        return grad * self.mask

