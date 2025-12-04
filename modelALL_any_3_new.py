import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
from torch import optim
from mamba.UnifyEncoder_any_new import MultiModalModel
from utils import cliptoken
from utils.Align import lalign_kl, lalign_kl2, lalign_maxmin
from utils.textnet import Text_Net_Adapter
from tune.vit_tune import Vit_encoder


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def onecat(specific, text_feat):
    specific_ = []
    for i,spe in enumerate(specific):
        text_feat = text_feat
        spe = torch.cat((spe, text_feat[i].unsqueeze(0).expand(spe.size(0), 1, -1)), dim=1)
        specific_.append(spe)
    return specific_

class modelAll(nn.Module):
    def __init__(self,device,in_channels_list,pache_size=11,num_classes=15):
        super(modelAll, self).__init__()
        self.device = device
        self.in_channels_list = in_channels_list
        self.encoder = MultiModalModel(in_channels_list=self.in_channels_list, pache_size=pache_size, device= self.device)
        self.text_net = Text_Net_Adapter(adapter_hidden_dim=64)
        self.vit = ViTModel.from_pretrained('large_vit')
        self.vit_encoder = Vit_encoder(self.vit,self.device, adapter_hidden_dim=64, use_adapter=True, len_modility=len(in_channels_list))
        self.FC = nn.ModuleList()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.vit.config.hidden_size))
        nn.init.normal_(self.cls_token, std=0.02)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

        for i in range(len(in_channels_list)):
            self.FC.append(nn.Linear(512, 768))

    def l1_loss(self, in_tensors, decoded_tensors):
        losses = []
        for input_tensor, decoded_tensor in zip(in_tensors, decoded_tensors):
            loss = F.l1_loss(decoded_tensor, input_tensor)
            losses.append(loss)
        return sum(losses) / len(losses)


    def forward(self, hsi, lidar, sar, text_c, text_m):
        B, C, H, W = hsi.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, E]
        fusion, specific_prompt, decoders, specific_align = self.encoder(hsi, lidar,sar)
        fusion = torch.cat((cls_tokens, fusion), dim=1)
        text_c = cliptoken.tokenize(text_c).to(self.device)
        text_feat_c = self.text_net(text_c)
        text_m = cliptoken.tokenize(text_m).to(self.device)
        text_feat_m = self.text_net(text_m)
        loss_kl = lalign_kl2(specific_align, specific_prompt)
        loss_laign = lalign_maxmin(specific_prompt, text_feat_c)
        l1 = self.l1_loss([hsi, lidar, sar], decoders)
        KV_cat = onecat(specific_prompt, text_feat_m)
        for i in range(len(self.in_channels_list)):
            KV_cat[i] = self.FC[i](KV_cat[i])
        KV_cat = tuple(KV_cat)
        out = self.vit_encoder(fusion,KV_cat)
        # out = self.vit_encoder(fusion,tuple(specific_prompt))
        out = self.vit.layernorm(out)
        out = torch.squeeze(self.classifier(out[:,0,:]))
        return out, loss_laign + 0.5 * loss_kl, l1

