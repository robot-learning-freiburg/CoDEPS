import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FlowHead(nn.Module):
    """This network predicts the object motion, i.e., scene flow without static areas.
    """

    def __init__(self,
                 num_ch_enc,
                 upsample_mode: str = 'bilinear',
                 use_skips: bool = True,
                 auto_mask: bool = True):
        super().__init__()

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = [16, 32, 64, 128, 256]

        self.upsample_mode = upsample_mode
        self.use_skips = use_skips
        self.auto_mask = auto_mask

        self.upconvs_0, self.upconvs_1 = nn.ModuleDict(), nn.ModuleDict()

        for i in range(4, -1, -1):
            # Upconv 0 (convolutions over previous layer)
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.upconvs_0[str(i)] = nn.Sequential(
                nn.Conv2d(num_ch_in, num_ch_out, 3, padding_mode='reflect', padding=1),
                nn.ELU(inplace=True))

            # Upconv 1 (convolutions over concatenated features: skip connections)
            if i > 0:
                num_ch_in = self.num_ch_dec[i]
                if self.use_skips:
                    num_ch_in += self.num_ch_enc[i - 1]
                num_ch_out = self.num_ch_dec[i]
                self.upconvs_1[str(i)] = nn.Sequential(
                    nn.Conv2d(num_ch_in, num_ch_out, 3, padding_mode='reflect', padding=1),
                    nn.ELU(inplace=True))

        # Final layer with 3 output channels for spatial displacement
        num_ch_in = self.num_ch_dec[0]
        num_ch_out = 3
        self.translation_layer = nn.Sequential(
            nn.Conv2d(num_ch_in, num_ch_out, 3, padding_mode='reflect', padding=1),
            nn.ELU(inplace=True))


    def forward(self, in_feats: Tensor) -> Tensor:
        x = in_feats[-1]
        for i in range(4, -1, -1):
            x = self.upconvs_0[str(i)](x)
            x = [F.interpolate(x, scale_factor=2, mode=self.upsample_mode, align_corners=True)]
            if i > 0:
                if self.use_skips:
                    x += [in_feats[i - 1]]
                x = torch.cat(x, dim=1)
                x = self.upconvs_1[str(i)](x)

        flow_map = .001 * self.translation_layer(x[0])

        if self.auto_mask:
            flow_map = self._mask(flow_map)

        return flow_map

    @staticmethod
    def _mask(x: Tensor) -> Tensor:
        """Copied from
        https://github.com/chamorajg/pytorch_depth_and_motion_planning/blob/c688338d9d8d8b7dad5722a5eeb0ed8b393a82a5/object_motion_net.py
        """
        sq_x = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
        mean_sq_x = torch.mean(sq_x, dim=(0, 2, 3))
        mask_x = (sq_x > mean_sq_x).type(x.dtype)
        x = x * mask_x
        return x
