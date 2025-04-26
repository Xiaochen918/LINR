import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math


def make_coord(shape, ranges=None, flatten=True):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    
    return ret

class INR_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))

        return x.view(*shape, -1)

class INRModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.L = 4
        self.hidden_list = [256, 256, 256]
        self.local_ensemble = True
        self.feat_unfold = True
        self.cell_decode = True
        imnet_in_dim = dim

        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 2 + 4 * self.L  
        if self.cell_decode:
            imnet_in_dim += 2

        self.imnet = INR_MLP(imnet_in_dim, dim, self.hidden_list)

    def forward(self, inp):
        B, h, w = inp.shape[0], inp.shape[2], inp.shape[3]
        coord = make_coord((h, w)).to(inp.device)
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        cell = cell.unsqueeze(0).repeat(B, 1, 1)
        coord = coord.unsqueeze(0).repeat(B, 1, 1)
        points_enc = self.positional_encoding(coord, L=self.L)
        coord = torch.cat([coord, points_enc], dim=-1)

        return self.query_rgb(inp, coord, cell)

    def query_rgb(self, inp, coord, cell=None):
        feat = inp
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(inp.device) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                bs, q, h, w = feat.shape
                q_feat = feat.view(bs, q, -1).permute(0, 2, 1)

                bs, q, h, w = feat_coord.shape
                q_coord = feat_coord.view(bs, q, -1).permute(0, 2, 1)

                points_enc = self.positional_encoding(q_coord, L=self.L)
                q_coord = torch.cat([q_coord, points_enc], dim=-1)  

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]
            areas[0] = areas[3]
            areas[3] = t
            t = areas[1]
            areas[1] = areas[2]
            areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        bs, q, h, w = feat.shape
        ret = ret.view(bs, h, w, -1).permute(0, 3, 1, 2)

        return ret

    def positional_encoding(self, input, L): 
        shape = input.shape
        freq = 2 ** torch.arange(L, dtype=torch.float32).to(input.device) * np.pi  
        spectrum = input[..., None] * freq  
        sin, cos = spectrum.sin(), spectrum.cos()  
        input_enc = torch.stack([sin, cos], dim=-2)  
        input_enc = input_enc.view(*shape[:-1], -1)

        return input_enc

class TokenGeneration(nn.Module):
    def __init__(self, dim, topk_win_num):
        super().__init__()
        
        self.INR_s = INRModule(dim)
        self.max_pool = nn.AdaptiveMaxPool2d((1,1))
        self.topk_win_num = topk_win_num
        self.winSize = 2

    def forward(self, z, x):
        # template
        B, N_t, C = z.shape
        h_t = int(math.sqrt(N_t))
        z_center = (z.permute(0,2,1).reshape(B,C,h_t,h_t))
        x_t_avg = self.max_pool(z_center)
        
        # search region
        B, N_s, C = x.shape
        h_s = int(math.sqrt(N_s))
        win_Size_all = int(self.winSize*self.winSize)
        win_Num_H = h_s//self.winSize

        x_conv = x.permute(0,2,1).reshape(1,B*C,h_s,h_s)
        x_conv = F.normalize(x_conv, dim=1)
        x_t_avg = F.normalize(x_t_avg, dim=1)
        sim_x_s = F.conv2d(x_conv, x_t_avg, groups=B).reshape(B,N_s)
        sim_x_s = sim_x_s.reshape(B,win_Num_H,self.winSize,win_Num_H,self.winSize).permute(0,1,3,2,4)
        sim_x_s = (sim_x_s.reshape(B,-1,win_Size_all)).mean(dim=-1)
        index_x_s_T = torch.topk(sim_x_s,k=self.topk_win_num,dim=-1)[1] # [B,win_topk]
        index_x_s_T = index_x_s_T.unsqueeze(dim=-1).unsqueeze(dim=-1).expand(-1,-1,win_Size_all,C)

        x_ext = x.reshape(B,win_Num_H,self.winSize,win_Num_H,self.winSize,C)
        x_ext = x_ext.permute(0,1,3,2,4,5).reshape(B,-1,win_Size_all,C)
        x_ext = torch.gather(x_ext,dim=1,index=index_x_s_T)
        x_ext = x_ext.permute(0,1,3,2).reshape(B*self.topk_win_num,C,self.winSize,self.winSize)
        x_ext = F.interpolate(x_ext, scale_factor=2, mode='bilinear', align_corners=False)
        x_ext = (self.INR_s(x_ext)+x_ext).reshape(B,self.topk_win_num,C,-1)
        x_ext = x_ext.permute(0,1,3,2).reshape(B,-1,C)

        return x_ext