import torch
from torch import nn
from torchvision.transforms import v2

transform_data = v2.Compose([
    v2.Resize((64, 64)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # image is no longer flattened
        self.features = nn.Sequential(
            # block 1, input 64x64 pixels, 3 channels (RGB)
            # kernel size = window size. 3 -> 3x3 = 9 pixels
            # stride = step size in pixels, default is 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            # output: 64x64, 32 channels
            nn.ReLU(),
            # halves size to 32x32, 32 channels
            nn.MaxPool2d(kernel_size=2, stride=2),
            # block 2, input is 32x32 from MaxPool2d
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3, stride=1, padding=1),
            # output: 32x32, 64 channels
            nn.ReLU(),
            # halves size to 16x16, 64 channels
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # flattens 16x16 pixels * 64 channels to a vector of length 16384
        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            # input features, 128 hidden features
            nn.Linear(64*16*16,128),
            nn.ReLU(),
            # output layer, 2 classes (cat, dog)
            nn.Linear(128,2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)