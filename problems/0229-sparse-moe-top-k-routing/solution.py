import numpy as np

def moe_topk_routing(
    router_logits: np.ndarray,
    expert_outputs: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Perform top-k expert routing for a Mixture-of-Experts layer.
    
    For each token:
    1. Select the top-k experts based on router_logits
    2. Compute softmax weights over only the selected experts
    3. Return weighted combination of the selected expert outputs
    
    Args:
        router_logits: Shape (batch_size, num_experts)
                      Raw scores from the router for each expert
        expert_outputs: Shape (batch_size, num_experts, hidden_dim)
                       Output from each expert for each input
        k: Number of experts to select per token
        
    Returns:
        Shape (batch_size, hidden_dim) - weighted combination of expert outputs
    """
    # Your code here
    top_experts = np.argsort(
        -router_logits,
        axis=-1,
        kind="stable"
    )[:, :k]

    top_logits = np.take_along_axis(
        router_logits,
        top_experts,
        axis=-1
    )

    mx = np.max(top_logits, axis=-1, keepdims=True)

    probs = np.exp(top_logits - mx)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
   
    selected_outputs = np.take_along_axis(
        expert_outputs,
        top_experts[:, :, None],
        axis=1
    )

    weighted_outputs = probs[:, :, None] * selected_outputs


    
    res = np.sum(weighted_outputs, axis=1)
    return res 
