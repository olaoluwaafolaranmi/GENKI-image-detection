import torch.nn as nn


class GENKIModel(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.dropout = nn.Dropout(0.2)

        self.conv1 = nn.Conv2d(3,32, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv3 = nn.Conv2d(32, 16, kernel_size=(3,3), stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2))    

        self.flat = nn.Flatten()

        self.fc4 = nn.Linear(1024,128)
        self.act4 = nn.ReLU()

        self.fc5 = nn.Linear(128, 1)
        self.act5 = nn.Sigmoid()


    def forward(self, x):

        #input 64x64x3  out 32x32x32
        x= self.act1(self.conv1(x))
        x= self.dropout(self.pool1(x))  
        #input 32x32x32  out 16x16x32
        x= self.act2(self.conv2(x))
        x= self.dropout(self.pool2(x))
        #input 16x16x32  out 8x8x32
        x= self.act3(self.conv3(x))
        x= self.pool3(x)
        #input 8x8x32  out 2048
        x= self.flat(x)
        #input 2048  out 128
        x = self.act4(self.fc4(x))
        #input 128  out 1
        x = self.act5(self.fc5(x))

        return x
        


        
        #consider adding drop out layers

        