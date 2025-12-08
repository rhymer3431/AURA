import torch
import torch.nn as nn
import torch.nn.functional as F


class REACTLoss(nn.Module):
    def __init__(
        self,
        pred_freq,
        lambda_cb=1.0,
        gamma=2.0,
        bg_weight=0.2,      # background down-weight
        temperature=1.0     # logit scaling
    ):
        super().__init__()

        self.lambda_cb = lambda_cb
        self.gamma = gamma
        self.bg_weight = bg_weight
        self.temperature = temperature

        # 1) frequency bias
        freq_bias = torch.log(torch.tensor(pred_freq, dtype=torch.float32) + 1e-6)
        freq_bias = (freq_bias - freq_bias.mean()) / (freq_bias.std() + 1e-6)
        self.register_buffer("freq_bias", freq_bias)

        # 2) class-balanced weight = 1/sqrt(freq)
        cb = 1.0 / torch.sqrt(torch.tensor(pred_freq, dtype=torch.float32) + 1e-6)
        cb = cb / cb.mean()     # normalize
        self.register_buffer("cb_weight", cb)

    def forward(self, logits, labels):
        """
        logits: (M, C)
        labels: (M,)
        """
        # temperature scaling
        logits = logits / self.temperature

        # 1) apply frequency bias
        biased_logits = logits + self.freq_bias.unsqueeze(0)

        # 2) softmax + CE (per-sample)
        ce = F.cross_entropy(biased_logits, labels, reduction="none")

        # 3) focal loss term
        with torch.no_grad():
            pt = torch.softmax(biased_logits, dim=-1)[torch.arange(len(labels)), labels]

        focal = (1 - pt) ** self.gamma

        # 4) class-balanced weight
        cb_w = self.cb_weight[labels]

        # 5) background down-weight (label=0)
        bg_mask = (labels == 0).float()
        fg_mask = 1 - bg_mask
        bg_scale = bg_mask * self.bg_weight + fg_mask * 1.0

        # 6) combine all components
        loss = focal * ce * cb_w * bg_scale

        return loss.mean()
