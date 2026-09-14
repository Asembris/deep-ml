def count_parameters(layers: list) -> int:
    """
    Count the total number of trainable parameters in a neural network.
    
    Args:
        layers: A list of dictionaries describing each layer.
                Each dict contains 'type' and layer-specific parameters.
    
    Returns:
        Total number of trainable parameters as an integer.
    """
    # Your code here
    res=0
    for layer in layers:
        if layer["type"]=="dense":
            res+=layer["input_size"]*layer["output_size"]
            res+=layer["output_size"] if layer["use_bias"] else 0
        elif layer["type"]=="conv2d":
            kernel_size=layer["kernel_size"]
            if isinstance(kernel_size, int):
                kh = kw = kernel_size
            else:
                kh, kw = kernel_size
            res+=layer["in_channels"]*layer["out_channels"] *kh * kw
            res+=layer["out_channels"] if layer["use_bias"] else 0

        else:
            res+=layer["num_embeddings"] *layer["embedding_dim"]
    return res




