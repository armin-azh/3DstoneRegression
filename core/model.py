from torch import nn
from torch import Tensor

from torchsummary import summary


class Model3DV1(nn.Module):
    def __init__(self, n_channels, n_feature):
        super(Model3DV1, self).__init__()

        self._model = nn.Sequential(
            nn.Conv3d(in_channels=n_channels, out_channels=n_feature, kernel_size=3),
            nn.MaxPool3d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=n_feature, out_channels=n_feature * 2, kernel_size=3),
            nn.MaxPool3d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=n_feature * 2, out_channels=n_feature * 4, kernel_size=3),
            nn.MaxPool3d(kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=65536, out_features=1),
            # nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)


if __name__ == '__main__':
    m = Model3DV1(n_channels=1, n_feature=32)
    summary(m, (1, 120, 120, 120))
