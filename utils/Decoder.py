import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, input_dim, output_channels, output_height, output_width):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.output_channels = output_channels
        self.output_height = output_height
        self.output_width = output_width

        # 定义解码器层
        self.fc = nn.Linear(input_dim, 128 * (output_height // 8) * (output_width // 8))
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # 将输入张量从 (1, 768) 转换为 (1, 128, H//8, W//8)
        x = self.fc(x)
        x = x.view(-1, 128, self.output_height // 8, self.output_width // 8)

        # 通过反卷积层逐步放大图像
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.tanh(self.deconv3(x))  # 使用 tanh 激活函数将输出限制在 [-1, 1] 范围内

        return x

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoded_tensors = [
        torch.randn(1, 768).to(device),
        torch.randn(1, 768).to(device),
        torch.randn(1, 768).to(device)
    ]

    # 定义解码器
    decoder1 = Decoder(input_dim=768, output_channels=3, output_height=30, output_width=30).to(device)
    decoder2 = Decoder(input_dim=768, output_channels=194, output_height=30, output_width=30).to(device)
    decoder3 = Decoder(input_dim=768, output_channels=194, output_height=30, output_width=30).to(device)

    # 解码张量
    decoded_tensors = []
    decoded_tensors.append(decoder1(encoded_tensors[0]))
    decoded_tensors.append(decoder2(encoded_tensors[1]))
    decoded_tensors.append(decoder3(encoded_tensors[2]))

    # 打印解码后的张量形状
    for i, decoded_tensor in enumerate(decoded_tensors):
        print(f"Decoded tensor {i + 1} shape: {decoded_tensor.shape}")