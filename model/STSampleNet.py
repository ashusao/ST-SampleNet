import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary
from collections import OrderedDict

import pickle

from model import SubsetOperator

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=True)


class ConvBnGelu(nn.Module):

    def __init__(self, in_channels, out_channels, k_size):
        super(ConvBnGelu, self).__init__()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k_size,
                     stride=1, padding='same', bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = F.gelu(self.bn(x))
        return x


class ResUnit(nn.Module):
    def __init__(self, n_filter, k_size):
        super(ResUnit, self).__init__()
        self.conv1_bn_gelu = ConvBnGelu(n_filter, n_filter, k_size)
        self.conv2_bn_gelu = ConvBnGelu(n_filter, n_filter, k_size)

    def forward(self, x):
        residual = x

        out = self.conv1_bn_gelu(x)
        out = self.conv2_bn_gelu(out)

        out += residual  # residual connection

        return out


class ResNet(nn.Module):
    def __init__(self, ResUnit, n_filter, k_size, n_block=1):
        super(ResNet, self).__init__()
        self.stacked_resunits = self.make_stack_resunits(ResUnit, n_filter, k_size, n_block)

    def make_stack_resunits(self, ResUnit, n_filter, k_size, n_block):
        layers = []

        for i in range(n_block):
            layers.append(ResUnit(n_filter, k_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stacked_resunits(x)

        return x


class ScaledDotProductAttention(nn.Module):

    def __init__(self, q_size):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = q_size ** 0.5

    def forward(self, q, k, v):
        # q, k, v = (b_size, seq_len, feat_size)
        qk_t = q.bmm(k.transpose(-1, -2))
        score = F.softmax(qk_t / self.scale, dim=-1)
        v = score.bmm(v)
        return v, score


class AttentionHead(nn.Module):

    def __init__(self, dim_feat, dim_head):
        super(AttentionHead, self).__init__()
        self.linear_q = nn.Linear(dim_feat, dim_head)
        self.linear_k = nn.Linear(dim_feat, dim_head)
        self.linear_v = nn.Linear(dim_feat, dim_head)
        self.attention = ScaledDotProductAttention(dim_head)

    def forward(self, q, k, v):
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)
        v, score = self.attention(q, k, v)

        return v, score


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, dim_feat):
        super(MultiHeadAttention, self).__init__()

        self.dim_head = dim_feat // n_head
        self.dim_feat = dim_feat
        self.n_head = n_head

        self.heads = nn.ModuleList(
            [AttentionHead(dim_feat=dim_feat, dim_head=self.dim_head) for _ in range(self.n_head)]
        )
        self.linear = nn.Linear(self.n_head * self.dim_head, self.dim_feat)

    def forward(self, q, k, v):
        x, score = [], []
        for h in self.heads:
            x_, sc_ = h(q, k, v)
            x.append(x_)
            score.append(sc_)

        x = torch.cat(x, dim=-1)
        score = torch.stack(score, dim=1)
        return self.linear(x), score


class EncoderLayer(nn.Module):

    def __init__(self, dim_feat=32, n_head=4, dim_ffn=64, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.dim_feat = dim_feat
        self.n_head = n_head
        self.dim_feed_fwd = dim_ffn

        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)

        self.multi_head_attention = MultiHeadAttention(n_head=n_head, dim_feat=dim_feat)
        self.layer_norm = nn.LayerNorm(dim_feat)
        self.ffn = self.point_wise_feed_forward(dim_in=dim_feat, dim_ffn=dim_ffn)

    def point_wise_feed_forward(self, dim_in=32, dim_ffn=64):
        return nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(dim_in, dim_ffn)),
            ('gelu1', nn.GELU()),
            ('fc2', nn.Linear(dim_ffn, dim_in))
        ]))

    def forward(self, X):
        tmp = X
        X, score = self.multi_head_attention(q=X, k=X, v=X) # self attention
        X = self.layer_norm(tmp + self.dropout_1(X))

        tmp = X
        X = self.ffn(X)
        X = self.layer_norm(tmp + self.dropout_2(X))

        return X, score


class Encoder(nn.Module):

    def __init__(self, embed_dim=32, n_head=4, dim_ffn=64, n_layer=1, dropout=0.1):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(dim_feat=embed_dim, n_head=n_head, dim_ffn=dim_ffn, dropout=dropout)
            for _ in range(n_layer)
        ])

    def forward(self, X):
        score = []
        for layer in self.encoder_layers:
            X, sc = layer(X)
            score.append(sc)
        score = torch.stack(score, dim=1)
        return X, score


class POIFeatureExtractor(nn.Module):

    def __init__(self, dim_in=1, dim_out=32, n_block=1):
        super(POIFeatureExtractor, self).__init__()

        self.in_dim = dim_in
        self.out_dim = dim_out

        self.cnn_extractor = nn.Sequential(OrderedDict([
            ('conv1', ConvBnGelu(in_channels=dim_in, out_channels=dim_out // 4, k_size=3)),
            ('ResNet', ResNet(ResUnit, n_filter=dim_out // 4, k_size=3, n_block=n_block)),
            ('conv1x1', ConvBnGelu(in_channels=dim_out // 4, out_channels=dim_out, k_size=1))
        ]))

    def forward(self, x):
        x = self.cnn_extractor(x)
        return x


class LocalFeatureExtractor(nn.Module):

    def __init__(self, dim_in=1, dim_out=32, n_block=2):
        super(LocalFeatureExtractor, self).__init__()

        self.in_dim = dim_in
        self.out_dim = dim_out

        self.cnn_extractor = nn.Sequential(OrderedDict([
            ('conv1', ConvBnGelu(in_channels=dim_in, out_channels=dim_out // 4, k_size=3)),
            ('ResNet', ResNet(ResUnit, n_filter=dim_out // 4, k_size=3, n_block=n_block)),
            ('conv1x1', ConvBnGelu(in_channels=dim_out // 4, out_channels=dim_out, k_size=1))
        ]))


    def forward(self, x):
        x = self.cnn_extractor(x)
        return x


class TemporalPositionEmbedding(nn.Module):

    def __init__(self, seq_len, dim_embed=512, device=None):
        super(TemporalPositionEmbedding, self).__init__()
        self.seq_len = seq_len
        self.device = device
        self.position_embed = nn.Embedding(seq_len, dim_embed)

    def forward(self, x):  # (b, seq_len, embed_dim)
        #print(x.shape)
        pos = torch.arange(self.seq_len, device=self.device)
        pos = pos.unsqueeze(0).expand(x.size(0), -1)                # (b, seq_len)
        #print(pos.shape, x.shape, self.position_embed(pos).shape)
        x = x + self.position_embed(pos)                            # (b, seq_len, embed_dim)

        return x


class PositionEmbedding(nn.Module):

    def __init__(self, seq_len, dim_embed=512, device=None):
        super(PositionEmbedding, self).__init__()
        self.seq_len = seq_len
        self.device = device
        self.position_embed = nn.Embedding(seq_len, dim_embed)

    def forward(self, x):  # (b, seq_len, dim_embed)
        #print(x.shape)
        pos = torch.arange(self.seq_len, device=self.device)
        pos = pos.unsqueeze(0).expand(x.size(0), -1)                # (b, seq_len)
        #print(pos.shape, x.shape, self.position_embed(pos).shape)
        x = x + self.position_embed(pos)                            # (b, seq_len, dim_embed)

        return x


class SpatialPositionEmbedding(nn.Module):

    def __init__(self, embed_dim=512, hirerachy_ratio=(0.5, 0.3, 0.2), city='hannover', device=None, dir=''):
        super(SpatialPositionEmbedding, self).__init__()
        self.device = device
        self.city = city
        self.dir = dir
        self.grid_geohash_dict, self.vocab_size = self.create_vocabulary()

        if round(sum(hirerachy_ratio)) != 1:
            raise ValueError("Proportions must sum to 1")

        len_embed = [int(embed_dim * proportion) for proportion in hirerachy_ratio]
        len_embed[-1] += embed_dim - sum(len_embed)

        self.embed_levels = nn.ModuleList([
            nn.Embedding(size, length) for size, length in zip(self.vocab_size, len_embed)
        ])

    def create_vocabulary(self):

        f_name = self.dir + self.city +'_geohash.pkl'
        with open(f_name, 'rb') as f:
            grid_geohash = pickle.load(f)

        # create vocabulary of geohashes
        geohash_to_number = [dict() for _ in range(4)]
        number_to_geohash = [dict() for _ in range(4)]

        sets = [set() for _ in range(4)]
        len_sets = []

        for k, v in grid_geohash.items():
            for i, val in enumerate(v):
                sets[i].add(val)

        for h, s in enumerate(sets):
            s = sorted(s)
            for i, val in enumerate(s):
                number_to_geohash[h][i] = val
                geohash_to_number[h][val] = i
            len_sets.append(len(s))

        grid_geohash_to_number = {}

        # convert list of geo-hashes to list of corresponding ids
        for k, v in grid_geohash.items():
            grid_geohash_to_number[k] = []
            for i, val in enumerate(v):
                grid_geohash_to_number[k].append(geohash_to_number[i][val])

        return grid_geohash_to_number, len_sets

    def forward(self, x):
        grid_pos = torch.arange(x.size(1), device=self.device)
        grid_pos = torch.stack([torch.tensor(self.grid_geohash_dict[key.item()], device=self.device) for key in grid_pos])
        grid_pos = grid_pos.unsqueeze(0).expand(x.size(0), x.size(1), -1)

        embed_levels = [embed(grid_pos[:, :, i]) for i, embed in enumerate(self.embed_levels)]
        embed = torch.cat(embed_levels, dim=-1)
        return x + embed


class RegionSampler(nn.Module):

    def __init__(self, k=100, tau=1.0, dim_in=128, dim_out=374):
        super().__init__()

        self.k = k
        self.tau = tau
        self.n = dim_out
        self.sampler = SubsetOperator(k=k, tau=self.tau, hard=False)

        self.in_fc = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, dim_in),
            nn.GELU()
        )
        self.out_fc = nn.Sequential(
            nn.Linear(dim_in, dim_in // 4),
            nn.GELU(),
            nn.Linear(dim_in // 4, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.in_fc(x)
        _, _, R, F = x.size()
        x_local = x[:, :, :, :(F // 2)]
        x_global = x[:, :, :, (F // 2):].mean(dim=-2).unsqueeze(-2)
        x = torch.cat((x_local, x_global.expand(-1, -1, R, -1)), dim=-1)
        x = self.out_fc(x)

        # take keep probablity
        x = x[:, :, :, 1].reshape(x.size(0), x.size(1), -1)

        if self.training:
            prob = self.sampler(x)
            _, top_idx = torch.topk(prob, k=self.k, largest=True)
        else:
            _, top_idx = torch.topk(x, k=self.k, largest=True)

        return top_idx


class STSampleNet(nn.Module):

    def __init__(self, len_conf=(4, 3, 2), n_c=1, n_poi=10, embed_dim=32, map_w=22, map_h=17, dim_ts_feat=8,
                 n_head_spatial=4, n_head_temporal=4, n_layer_spatial=4, n_layer_temporal=4, dropout=0.1,
                 hirerachy_ratio=(0.3, 0.3, 0.2, 0.2), region_keep_rate=0.8,  tau=1.0, city='hannover',
                 teacher=False, device=None, dir=''):
        super(STSampleNet, self).__init__()

        self.len_conf = len_conf
        self.n_time = sum(len_conf)
        self.map_w = map_w
        self.map_h = map_h
        self.n_poi = n_poi
        self.dir = dir
        self.n_layer_spatial = n_layer_spatial
        self.n_layer_temporal = n_layer_temporal
        self.ts_feat_dim = dim_ts_feat
        self.spatial_n_head = n_head_spatial
        self.temporal_n_head = n_head_temporal
        self.dim_feat_prop = hirerachy_ratio
        self.city = city
        self.teacher = teacher
        self.device = device

        self.k_region = int(region_keep_rate * self.map_h * self.map_w)

        self.n_c = n_c
        self.feat_dim = embed_dim

        self.dropout = nn.Dropout(p=dropout)

        # feature encoder modules

        self.local_feature_encoder = LocalFeatureExtractor(dim_in=n_c, dim_out=embed_dim, n_block=3)
        #self.local_feature_encoder = nn.Linear(in_features=n_c, out_features=embed_dim)
        self.poi_feature_encoder = POIFeatureExtractor(dim_in=self.n_poi, dim_out=embed_dim, n_block=3)
        self.ts_encoder = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=dim_ts_feat, out_features=embed_dim // 2)),
            ('gelu1', nn.GELU()),
            ('fc2', nn.Linear(in_features=embed_dim // 2, out_features=embed_dim)),
            ('gelu2', nn.GELU()),
        ]))

        if not teacher:
            self.region_sampler = RegionSampler(k=self.k_region, tau=tau, dim_in=embed_dim, dim_out=self.map_w * self.map_h)

        self.cls_token_region = nn.Parameter(torch.zeros(1, self.n_time, 1, embed_dim))
        self.spatial_pos_embed = SpatialPositionEmbedding(embed_dim=embed_dim, hirerachy_ratio=hirerachy_ratio, city=self.city,
                                                          device=device, dir=self.dir)
        #self.spatial_pos_embed = PositionEmbedding(seq_len=map_h * map_w, dim_embed=embed_dim, device=device)
        self.spatial_encoder = Encoder(embed_dim=embed_dim, n_head=n_head_spatial, dim_ffn=3 * embed_dim,
                                       n_layer=n_layer_spatial, dropout=dropout)

        self.cls_token_time = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.temporal_pos_embed = TemporalPositionEmbedding(seq_len=self.n_time, dim_embed=embed_dim, device=device)
        self.temporal_encoder = Encoder(embed_dim=embed_dim, n_head=n_head_temporal, dim_ffn=3 * embed_dim,
                                        n_layer=n_layer_temporal, dropout=dropout)

        self.predict = nn.Linear(in_features=embed_dim, out_features=self.n_c * self.map_w * self.map_h)

    def feature_encoder(self, X, ts, X_poi):

        X = torch.stack([self.local_feature_encoder(X[:, t, :, :, :]) for t in range(X.size(1))], dim=1)
        X_poi = self.poi_feature_encoder(X_poi)
        ts = self.ts_encoder(ts)

        return X, ts, X_poi

    def forward(self, X, ts, X_poi):

        X, ts, X_poi = self.feature_encoder(X, ts, X_poi)

        X = torch.permute(X, (0, 1, 3, 4, 2))
        X_poi_ = torch.permute(X_poi, (0, 2, 3, 1)).unsqueeze(1).expand(-1, sum(self.len_conf), -1, -1, -1)
        ts_ = ts.unsqueeze(2).unsqueeze(3).expand(-1, -1, self.map_h, self.map_w, -1)

        X = X.view(-1, X.size(1), self.map_w * self.map_h, self.feat_dim)  # (b, c, h*w, f)
        X_poi_ = X_poi_.view(-1, X_poi_.size(1), self.map_w * self.map_h, self.feat_dim)  # (b, c, h*w, f)
        ts_ = ts_.view(-1, ts_.size(1), self.map_w * self.map_h, self.feat_dim)  # (b, c, h*w, f)

        # Apply spatial position encoding in each timestamp
        X = torch.stack([self.dropout(self.spatial_pos_embed(X[:, t, :, :])) for t in range(X.size(1))], dim=1)

        Z = X + X_poi_ + ts_

        cls_token_region = self.cls_token_region.expand(Z.shape[0], -1, -1, -1)

        keep_idx = torch.tensor(0.0, device=self.device)
        if not self.teacher:
            Z_poi_time = X_poi_ + ts_
            keep_idx = self.region_sampler(Z_poi_time)

            keep_idx = keep_idx.unsqueeze(-1).expand(-1, -1, -1, Z.size(-1))
            Z_keep = Z.gather(dim=-2, index=keep_idx)
            Z = torch.cat((cls_token_region, Z_keep), dim=-2)
        else:
            Z = torch.cat((cls_token_region, Z), dim=-2)

        Z_tmp, score_spatial = [], []
        for t in range(Z.size(1)):
            Z_, score = self.spatial_encoder(Z[:, t, :, :])
            Z_tmp.append(Z_)
            score_spatial.append(score)

        Z = torch.stack(Z_tmp, dim=1)
        score_spatial = torch.stack(score_spatial, dim=1)

        # extract cls_token
        Z_t = Z[:, :, 0, :]

        Z_t_ts = Z_t + ts
        Z_t_ts = self.temporal_pos_embed(Z_t_ts)

        cls_token_temp = self.cls_token_time.expand(Z_t_ts.shape[0], -1, -1)
        Z_t_ts = torch.cat((cls_token_temp, Z_t_ts), dim=-2)

        Z_t_ts, score_temporal = self.temporal_encoder(Z_t_ts)
        out = self.predict(Z_t_ts[:, 0, :])

        pred = F.tanh(out)
        pred = pred.view(-1, self.n_c, self.map_h, self.map_w)

        return pred, Z_t, score_spatial, score_temporal, keep_idx

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STSampleNet(len_conf=(4, 3, 2), n_c=3, embed_dim=128, map_w=20, map_h=20,
                        dim_ts_feat=10, n_head_spatial=3, n_head_temporal=3,
                        n_layer_spatial=2, n_layer_temporal=2, hirerachy_ratio=(0.3, 0.3, 0.2, 0.2),
                        region_keep_rate=0.8, tau=1.0, city='hannover', teacher=False, device=device)
    #print(model)
    model.to(device)

    summary(model, [(9, 3, 20, 20),  (9, 10), (10, 20, 20)], device=device)