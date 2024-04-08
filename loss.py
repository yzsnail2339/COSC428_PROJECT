import torch
import torch.nn.functional as F


def cross_entropy2d(predict, target, weight=None, size_average=True):
    assert predict.dim() == 4, "预测的维度必须是4 (n, c, h, w)"
    assert target.dim() == 3, "目标的维度必须是3 (n, h, w)"
    n, c, h, w = predict.size()
    assert target.size(0) == n and target.size(1) == h and target.size(2) == w, "预测和目标的空间尺寸必须匹配"
    
    log_p = F.log_softmax(predict, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    
    # 更新loss计算方式以适应新的API
    if size_average:
        reduction = 'mean'
    else:
        reduction = 'sum'
    
    loss = F.nll_loss(log_p, target, weight=weight, reduction=reduction)
    return loss