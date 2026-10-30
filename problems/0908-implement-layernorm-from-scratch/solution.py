import torch

def layer_norm(x, gamma, beta, eps=1e-5):
    # TODO: normalize over the last dim, then affine-transform with gamma and beta
    mean=torch.mean(x,keepdim=True,dim=-1)
    var=torch.var(x,keepdim=True,unbiased=False,dim=-1)
    x_=(x-mean)/torch.sqrt(var+eps)
    res= gamma * x_ + beta 
    return res
