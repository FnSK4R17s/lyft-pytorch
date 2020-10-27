import torch
import config


def shapefy( xy_pred, xy, xy_av):
    NDIM = 3
    xy_pred = xy_pred.view(-1, config.HFORWARD, NDIM, 2)
    xy = xy.view(-1, config.HFORWARD, 2)[:,:,None]
    xy_av = xy_av.view(-1, config.HFORWARD)[:,:,None]
    return xy_pred, xy,xy_av

def LyftLoss(c, xy_pred, xy, xy_av):
    c = c.view(-1,c.shape[-1])
    xy_pred, xy, xy_av  = shapefy(xy_pred, xy, xy_av)
    
    c = torch.softmax(c, dim=1)
    
    l = torch.sum(torch.mean(torch.square(xy_pred-xy), dim=3)*xy_av, dim=1)
    
    # The LogSumExp trick for better numerical stability
    # https://en.wikipedia.org/wiki/LogSumExp
    m = l.min(dim=1).values
    l = torch.exp(m[:, None]-l)
    
    l = m - torch.log(torch.sum(l*c, dim=1))
    denom = xy_av.max(2).values.max(1).values
    l = torch.sum(l*denom)/denom.sum()
    return 3*l # I found that my loss is usually 3 times smaller than the LB score


def MSE(xy_pred, xy, xy_av):
    xy_pred, xy, xy_av = shapefy(xy_pred, xy, xy_av)
    return 9*torch.mean(torch.sum(torch.mean(torch.square(xy_pred-xy), 3)*xy_av, dim=1))

def MAE(xy_pred, xy, xy_av):
    xy_pred, xy, xy_av = shapefy(xy_pred, xy, xy_av)
    return 9*torch.mean(torch.sum(torch.mean(torch.abs(xy_pred-xy), 3)*xy_av, dim=1))