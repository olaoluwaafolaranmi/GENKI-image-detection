import torch.nn as nn
import torch
from pytorch_model_summary import summary


class GENKIModel(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.dropout = nn.Dropout(0.2)

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout2d(0.25)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout2d(0.25)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout2d(0.25)
        )

        self.flat = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Linear(2048,128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fc2 = nn.Linear(128, 1)


    def forward(self, x):
        # input 64x64x3 out 32x32x32
        x = self.block1(x)
        # input 32x32x32 out 16x16x32
        x = self.block2(x)
        # input 16x16x32 out 8x8x32
        x = self.block3(x)
        # input 8x8x32 out 2048
        x = self.flat(x)
        # input 2048 out 128
        x = self.fc1(x)
        #input 128 out 1
        x = self.fc2(x)

        return x
        

def main():
    model = GENKIModel()
    print(summary(model, torch.rand((64,3,64,64)).float()))


if __name__ == '__main__':
    main()