import torch
from torch import nn

class PositionalEncoding(nn.Module):
    """Positional encoding.

    Defined in :numref:`sec_self-attention-and-positional-encoding`"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)*0.1
        self.P[:, :, 1::2] = torch.cos(X)*0.1

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class Addnorm(nn.Module):
    def __init__(self, normshape, dropout, *args, **kwargs):
        super(Addnorm, self).__init__(*args, **kwargs)
        self.norm = nn.LayerNorm([normshape])
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, y):
        return self.norm(x+self.dropout(y))

class transformerencoderlayer(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1, *args, **kwargs):
        super(transformerencoderlayer, self).__init__(*args, **kwargs)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.addnorm1 = Addnorm(hidden_size, dropout=dropout)
        self.ffn = nn.Sequential(nn.Linear(hidden_size, hidden_size*2), nn.ELU(2),
                                 nn.Linear(hidden_size*2, hidden_size))
        self.addnorm2 = Addnorm(hidden_size, dropout=dropout)
    def forward(self, x):
        x = self.addnorm1(self.attention(x, x, x)[0], x)
        x =  self.addnorm2(self.ffn(x)[0], x)
        return x
        

class Amplitude_frequency_increment_Model(nn.Module):
    def __init__(self, hidden_size, *args, **kwargs):
        super(Amplitude_frequency_increment_Model, self).__init__(*args, **kwargs)
        self.pos = PositionalEncoding(hidden_size, dropout=0.1)
        self.l1 = nn.Linear(6, hidden_size)
        self.en = transformerencoderlayer(hidden_size)
        self.l2 = nn.Sequential(nn.Linear(hidden_size, 6), nn.ELU(2),
                                 nn.Linear(6, 1))

    def forward(self, x):
        # (B, 1, 6, 12)
        x_real = x.real.squeeze(1).permute(0,2,1)  # (B, 7, 6)
        x_imag = x.imag.squeeze(1).permute(0,2,1)
        x = torch.concatenate([x_real, x_imag], dim=1)
        x = self.l2(self.en(self.pos(self.l1(x))))
        x_real, x_imag = x[:, :7, :], x[:, 7:, :]

        return torch.complex(x_real, x_imag).permute(0,2,1).unsqueeze(1)


class Residual(nn.Module):
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = nn.ReLU()(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return nn.ReLU()(Y)

class Post_processing_Model(nn.Module):
    def __init__(self, hidden_size, *args, **kwargs):
        super(Post_processing_Model, self).__init__(*args, **kwargs)
        self.net = nn.Sequential(nn.Conv2d(1, hidden_size, 1),
                                 Residual(hidden_size, hidden_size),
                                 nn.Conv2d(hidden_size, 1, 1))
    def forward(self, x):
        return self.net(x)

class Hierarchical_Model(nn.Module):
    def __init__(self, hidden_size, *args, **kwargs):
        super(Hierarchical_Model, self).__init__(*args, **kwargs)
        self.net1 = Amplitude_frequency_increment_Model(hidden_size)
        self.net2 = Post_processing_Model(64)
    def forward(self, x):
        try:
            assert x.dtype is torch.complex64
            x = self.net1(x)
        except:
            assert x.dtype is torch.float32
            x = self.net2(x.real)
        return x

def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)