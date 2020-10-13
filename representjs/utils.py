import torch
from torch.optim.lr_scheduler import LambdaLR


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k,
    from https://github.com/facebookresearch/moco/blob/master/main_moco.py"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def count_parameters(model):
    """From https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
