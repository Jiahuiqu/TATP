from transformers import ViTModel
import torch
import torch.nn as nn



class embedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(embedding, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear = nn.Linear(256, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x= self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.linear(x)
        return x.reshape(x.size(0), -1,self.out_channels)

class classfication_head(nn.Module):
    #use MLP
    def __init__(self, in_channels, out_channels):
        super(classfication_head, self).__init__()
        self.fc0 = nn.Linear(in_channels, 256)
        self.fc = nn.Linear(256, out_channels)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.fc0(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

class vit_model(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(vit_model, self).__init__()
        self.embedding = embedding(in_channels, 768)
        # self.vit = ViTModel.from_pretrained('../large_vit')
        self.vit = ViTModel.from_pretrained('large_vit')

        self.encoder = self.vit.encoder

        self.classfication_head = classfication_head(768, out_channels)
    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)[0]
        x = self.classfication_head(x[:,0,:])
        return x

if __name__ == '__main__':
    x = torch.randn(4, 144, 11, 11)
    model = vit_model(144, 15)
    print(model(x).shape)

