def linear_lr_decay(initial_lr: float, end_lr: float, num_steps: int) -> list:
    """
    Generate a linear learning rate decay schedule.
    
    Args:
        initial_lr: Starting learning rate
        end_lr: Final learning rate
        num_steps: Total number of training steps
    
    Returns:
        List of learning rates for each step
    """
    # Your code here
    if num_steps==0:
        return []
    if num_steps==1:
        return [initial_lr]

    learning_rates=[initial_lr]
    step=(end_lr-initial_lr)/num_steps
    step = (end_lr - initial_lr) / (num_steps - 1)
    learning_rates = []

    for i in range(num_steps):
        learning_rates.append(initial_lr + i * step)

    return learning_rates







