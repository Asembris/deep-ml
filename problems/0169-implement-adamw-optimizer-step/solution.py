import numpy as np

def adamw_update(w, g, m, v, t, lr, beta1, beta2, epsilon, weight_decay):
    """
    Perform one AdamW optimizer step.
    Args:
      w: parameter vector (np.ndarray)
      g: gradient vector (np.ndarray)
      m: first moment vector (np.ndarray)
      v: second moment vector (np.ndarray)
      t: integer, current time step
      lr: float, learning rate
      beta1: float, beta1 parameter
      beta2: float, beta2 parameter
      epsilon: float, small constant
      weight_decay: float, weight decay coefficient
    Returns:
      w_new, m_new, v_new
    """
    # Your code here
    m_new=m*beta1 + (1-beta1)*g 
    v_new=v*beta2 + (1-beta2)*g *g 
    m_c=m_new/(1-beta1**t)
    v_c=v_new/(1-beta2**t)
    w-= lr * weight_decay * w 

    w-= lr * m_c/(np.sqrt(v_c)+epsilon)

    return (w,m_new,v_new)



