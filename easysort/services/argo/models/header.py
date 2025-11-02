from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d, Linear, BatchNorm2d

def dw_sep(in_ch, out_ch, k=5, s=1, d=1):
    # depthwise conv -> pointwise conv
    return (
        Conv2d(in_ch, in_ch, kernel_size=k, stride=s, padding=(k//2)*d, dilation=d, groups=in_ch, bias=False),
        BatchNorm2d(in_ch),
        Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        BatchNorm2d(out_ch),
    )

class TinyMaskPresenceNet:
    def __init__(self, in_ch=1):
        # A bit of receptive field via k=5 and one dilated block
        self.dw1 = dw_sep(in_ch, 16, k=5, s=2, d=1)   # (H/2, W/2)
        self.dw2 = dw_sep(16, 32, k=5, s=2, d=1)      # (H/4, W/4)
        self.dw3 = dw_sep(32, 64, k=3, s=2, d=2)      # (H/8, W/8) dilated -> bigger context
        self.head = Linear(64*2, 1)  # concat(avg, max) -> 128 -> 1

        # Optional: bias the logit to prior (use in training loop)
        self.init_logit_bias = 0.0

    def _block(self, x, blk):
        dw, bn1, pw, bn2 = blk
        x = dw(x).relu()
        x = bn1(x)
        x = pw(x).relu()
        x = bn2(x)
        return x

    def __call__(self, x: Tensor) -> Tensor:
        # Accepts (B, C, H, W) or (B, H, W)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self._block(x, self.dw1)
        x = self._block(x, self.dw2)
        x = self._block(x, self.dw3)

        # Global average and max pool (input-size agnostic)
        H, W = x.shape[-2], x.shape[-1]
        xav = x.avg_pool2d(kernel_size=(H, W)).reshape((x.shape[0], x.shape[1]))  # (B,64)
        xmax = x.max_pool2d(kernel_size=(H, W)).reshape((x.shape[0], x.shape[1]))  # (B,64)
        x = Tensor.cat(xav, xmax, dim=1)  # (B,128)

        logits = self.head(x)  # (B,1)
        return logits

    @property
    def params(self) -> list[Tensor]:
        ps: list[Tensor] = []
        for blk in (self.dw1, self.dw2, self.dw3):
            dw, bn1, pw, bn2 = blk
            for m in (dw, bn1, pw, bn2):
                if hasattr(m, 'weight') and m.weight is not None:
                    ps.append(m.weight)
                if hasattr(m, 'bias') and getattr(m, 'bias', None) is not None:
                    ps.append(m.bias)
        if hasattr(self.head, 'weight') and self.head.weight is not None:
            ps.append(self.head.weight)
        if hasattr(self.head, 'bias') and self.head.bias is not None:
            ps.append(self.head.bias)
        return ps