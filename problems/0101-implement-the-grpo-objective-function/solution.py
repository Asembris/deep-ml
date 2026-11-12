import numpy as np

def grpo_objective(rhos, A, pi_theta_old, pi_theta_ref, epsilon=0.2, beta=0.01) -> float:
    """
    Compute the GRPO objective function.

    Args:
        rhos: List of likelihood ratios (pi_theta / pi_theta_old).
        A: List of advantage estimates.
        pi_theta_old: List of old policy probabilities (per-sample, not normalized).
        pi_theta_ref: List of reference policy probabilities (per-sample, not normalized).
        epsilon: Clipping parameter for the surrogate objective.
        beta: KL divergence penalty coefficient.

    Returns:
        The computed GRPO objective value.
    """
    # Your code here
    def kl(ratio):
        return ratio-np.log(ratio)-1
    
    rhos=np.array(rhos)
    A=np.array(A)
    pi_theta = rhos * pi_theta_old
    kl_loss = np.mean(kl(pi_theta_ref / pi_theta)*rhos)

    l_policy=np.mean(np.minimum(rhos* A,np.clip(rhos,1-epsilon,1+epsilon)*A))

    return l_policy-beta*kl_loss


