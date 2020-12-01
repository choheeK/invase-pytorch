import torch.nn as nn


class KWTAMask(nn.Module):
    def __init__(self, n_pick=50):
        super(KWTAMask, self).__init__()
        self.k = n_pick

    def forward(self, x):
        topval = x.topk(self.k, dim=1)[0][:, -1]
        topval = topval.expand(x.shape[1], x.shape[0]).permute(1, 0)
        comp = (x >= topval).to(x)
        return comp


class LinearDecoder(nn.Module):
    def __init__(self, dim):
        super(LinearDecoder, self).__init__()
        self.layer1 = nn.Linear(
            in_features=dim, out_features=dim)
        self.kwta = KWTAMask()

    def forward(self, selection, inp):
        kwta_out = self.kwta(selection).clone().detach()
        selected = kwta_out * inp
        out = self.layer1(selected)
        return selected, out


