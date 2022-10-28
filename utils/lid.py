import torch
import numpy as np
from scipy.spatial.distance import cdist
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters_in_MB(model):
    return sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary_head" not in name)/1e6

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)

    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(1/batch_size))
    return res

# calculate local intrinsic dimension
def track_latent_lid(model, loader, k=128):
    lids = []
    if isinstance(model, torch.nn.DataParallel):
        model.module.get_features = True
    else:
        model.get_features = True
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        b = images.shape[0]
        batch_lid = []
        k = images.shape[0]
        with torch.no_grad():
            fs, logits = model(images)
            # print(len(fs))
            for f in fs:
                f = f.view(b, -1)
                lid = mle_batch_pt(f, f, k).detach().cpu()
                batch_lid.append(lid)
        lids.append(batch_lid)
    if isinstance(model, torch.nn.DataParallel):
        model.module.get_features = False
    else:
        model.get_features = False
    lids = np.array(lids)
    shape = lids.shape
    return lids.sum(axis=0)


def mle_batch_pt(data, batch, k=20):
    b = data.shape[0]
    # data = data.view(b, -1)
    # batch = batch.view(b, -1)
    r = torch.cdist(data, batch, p=2)
    k = min(k, b-1)
    lids = []
    for i in range(data.shape[0]):
        a = torch.topk(r[i], k=k+1, dim=0, largest=False)[0]
        # [1:k+1]
        lid = -k / torch.sum(torch.log(a/a[-1]))
        lids.append(lid)
    return torch.stack(lids)


# lid of a single query point x
def mle_single(data, x, k=20, dist=True, metric='euclidean'):
    data = np.asarray(data, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape((-1, x.shape[0]))
    if dist:
        k = min(k, len(data)-1)

    def f(v): return - k / np.sum(np.log(v/v[-1]))
    if dist:
        a = cdist(x, data, metric=metric)
        a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k+1]
    else:
        a = data
        a = np.apply_along_axis(np.sort, axis=1, arr=a)
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a[0]

# lid of a batch of query points X


def mle_batch(data, batch, k):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data)-1)
    def f(v): return - k / np.sum(np.log(v/v[-1]))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a
