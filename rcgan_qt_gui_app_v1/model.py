import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_out, 4, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        return h_cur, c_cur

    def init_hidden(self, batch_size, spatial_size, device):
        h, w = spatial_size
        return (
            torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
            torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
        )


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class RecurrentGenerator(nn.Module):
    def __init__(self, cond_channels=6, hidden_dim=512, out_channels=3):
        super().__init__()
        self.enc1 = ConvBlock(3 + cond_channels, 64, normalize=False)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, hidden_dim)

        self.convlstm = ConvLSTMCell(hidden_dim, hidden_dim)

        self.dec1 = DeconvBlock(hidden_dim + hidden_dim, 256)
        self.dec2 = DeconvBlock(256 + 256, 128)
        self.dec3 = DeconvBlock(128 + 128, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def encode_frame(self, frame, cond_map):
        x = torch.cat([frame, cond_map], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return e1, e2, e3, e4

    def forward(self, prev_img, curr_img, cond_map):
        _, _, _, prev_e4 = self.encode_frame(prev_img, cond_map)
        curr_e1, curr_e2, curr_e3, curr_e4 = self.encode_frame(curr_img, cond_map)

        b, c, h, w = prev_e4.shape
        h_t, c_t = self.convlstm.init_hidden(b, (h, w), prev_e4.device)

        h_t, c_t = self.convlstm(prev_e4, h_t, c_t)
        h_t, c_t = self.convlstm(curr_e4, h_t, c_t)

        d1 = self.dec1(torch.cat([h_t, curr_e4], dim=1))
        d2 = self.dec2(torch.cat([d1, curr_e3], dim=1))
        d3 = self.dec3(torch.cat([d2, curr_e2], dim=1))

        return self.final(torch.cat([d3, curr_e1], dim=1))
