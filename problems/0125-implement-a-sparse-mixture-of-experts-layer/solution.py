import numpy as np

def moe(x: np.ndarray, We: np.ndarray, Wg: np.ndarray, n_experts: int, top_k: int) -> np.ndarray:
    """
    Args:
        x: Input tensor of shape (n_batch, l_seq, d_model)
        We: Expert weights of shape (n_experts, d_model, d_model)
        Wg: Gating weights of shape (d_model, n_experts)
        n_experts: Number of experts
        top_k: Number of experts to route each token to
    Returns:
        Output tensor of shape (n_batch, l_seq, d_model)
    """
   
    logits= x @ Wg 
    mx=np.max(logits,axis=-1,keepdims=True)
    expert_prob=np.exp(logits-mx)/np.sum(np.exp(logits-mx),axis=-1,keepdims=True)

    expert_prob_mask=np.zeros_like(expert_prob)
    top_k_experts=np.argsort(expert_prob,axis=-1)[:,:,-top_k:]

    np.put_along_axis(
        expert_prob_mask,
        top_k_experts,
        np.take_along_axis(expert_prob, top_k_experts, axis=-1),
        axis=-1
    )
    expert_prob_mask/=np.sum(expert_prob_mask,axis=-1,keepdims=True)

    res=np.zeros_like(x,dtype=float)
    for expert in range(n_experts):
        o=x @ We[expert]
        res+= expert_prob_mask[:,:,expert,None] * o
    return res



   












