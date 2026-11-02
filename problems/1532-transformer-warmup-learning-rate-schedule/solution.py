def transformer_lr(step, d_model, warmup_steps, factor=1.0):
    """
    Learning rate at a single update step (step >= 1) following section 5.3 of the paper:
    factor * d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5).
    Returns a float.
    """
    return factor *(d_model**(-0.5)) * min(1/(step**0.5),step*(warmup_steps**(-1.5)))


def lr_schedule(num_steps, d_model, warmup_steps, factor=1.0):
    """
    List of learning rates for steps 1..num_steps, each rounded to 8 decimals.
    """
    res=[]
    for step in range(1,num_steps+1):
        lr=transformer_lr(step,d_model,warmup_steps,factor)
        res.append(round(lr,8))
    return res