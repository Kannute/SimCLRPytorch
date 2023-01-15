import torch


def accuracy(output, target, top_k=(1,)) -> list:
    """Computes the accuracy over the k top predictions for the specified values of k.

    :param output: Model output
    :param target: Target model output
    :param top_k: K top predictions
    :returns list: List of top K predictions
    """

    with torch.no_grad():
        max_k = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
