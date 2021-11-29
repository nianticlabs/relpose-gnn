"""
implementation of PoseNet and MapNet networks
ToDo(Ozgur): where is the original implementation from?
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.autograd import Variable
from torch_geometric.nn import knn_graph

from niantic.modules.att import AttentionBlock
from niantic.modules.my_gnn_layer import simpleConv, simpleConvEdge, simpleConvEdge_upt

# from path_config import PATH_PROJECT, PATH_PYTHON

# Note: moved to python/__init__.py
# os.environ['TORCH_HOME'] = os.path.join('../../..', 'data', 'models')


# def trace_hook(m, g_in, g_out):
#  for idx,g in enumerate(g_in):
#    g = g.cpu().data.numpy()
#    if np.isnan(g).any():
#      set_trace()
#  return None

def filter_hook(m, g_in, g_out):
    g_filtered = []
    for g in g_in:
        g = g.clone()
        g[g != g] = 0
        g_filtered.append(g)
    return tuple(g_filtered)


class PoseNet(nn.Module):
    def __init__(self, feature_extractor, droprate=0.5, pretrained=True,
                 feat_dim=2048, filter_nans=False):
        super(PoseNet, self).__init__()
        self.droprate = droprate

        # replace the last FC layer in feature extractor
        self.feature_extractor = feature_extractor
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

        self.fc_xyz = nn.Linear(feat_dim, 3)
        self.fc_wpqr = nn.Linear(feat_dim, 3)
        if filter_nans:
            self.fc_wpqr.register_backward_hook(hook=filter_hook)

        # initialize
        if pretrained:
            init_modules = [self.feature_extractor.fc, self.fc_xyz, self.fc_wpqr]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.relu(x)
        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)

        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        return torch.cat((xyz, wpqr), 1)


class MapNet(nn.Module):
    """
    Implements the MapNet model (green block in Fig. 2 of paper)
    """

    def __init__(self, mapnet):
        """
        :param mapnet: the MapNet (two CNN blocks inside the green block in Fig. 2
        of paper). Not to be confused with MapNet, the model!
        """
        super(MapNet, self).__init__()
        self.mapnet = mapnet

    def forward(self, x):
        """
        :param x: image blob (N x T x C x H x W)
        :return: pose outputs
         (N x T x 6)
        """
        s = x.size()
        x = x.view(x.size(0), 3, 256, -1).contiguous()
        # x = x.view(-1, *s[2:])
        poses = self.mapnet(x)
        poses = poses.view(s[0], s[1], -1)
        return poses


class PoseNetX(nn.Module):
    """PoseNet with GNN top of it"""

    def __init__(self, feature_extractor, droprate=0.5, pretrained=True,
                 feat_dim=2048, filter_nans=False, input_img_height=256, use_gnn=False):
        super(PoseNetX, self).__init__()
        self.droprate = droprate
        self.input_img_height = input_img_height
        self.use_gnn = use_gnn

        # replace the last FC layer in feature extractor
        self.feature_extractor = feature_extractor
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

        if self.use_gnn:
            self.gnn1 = simpleConv(feat_dim, feat_dim)
            self.gnn2 = simpleConv(feat_dim, feat_dim)
            # self.gnn3 = simpleConv(feat_dim, feat_dim)
            # self.gnn4 = simpleConv(feat_dim, feat_dim)

        # self.bn = nn.BatchNorm1d(feat_dim)

        self.fc_xyz = nn.Linear(feat_dim, 3)
        self.fc_wpqr = nn.Linear(feat_dim, 3)
        if filter_nans:
            self.fc_wpqr.register_backward_hook(hook=filter_hook)

        # initialize
        if pretrained:
            init_modules = [self.feature_extractor.fc, self.fc_xyz, self.fc_wpqr],
            # self.gnn1, self.gnn2, self.gnn3, self.gnn4]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = x.view(x.size(0), 3, self.input_img_height, -1).contiguous()
        x = self.feature_extractor(x)
        x = F.relu(x)

        if self.use_gnn:
            x = self.gnn1(x, edge_index)
            x = F.relu(x)
            # x = self.gnn1(x, edge_index)
            # x = F.relu(x)
            # x = self.gnn3(x, edge_index)
            # x = F.relu(x)
            # x = self.gnn4(x, edge_index)
            # x = F.relu(x)

        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)

        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        return torch.cat((xyz, wpqr), 1)


class PoseNetX2(nn.Module):
    """PosentX + VO loss"""

    def __init__(self, feature_extractor, droprate=0.5, pretrained=True,
                 feat_dim=2048, filter_nans=False, input_img_height=256, use_gnn=False):
        super(PoseNetX2, self).__init__()
        self.droprate = droprate
        self.input_img_height = input_img_height
        self.use_gnn = use_gnn

        # replace the last FC layer in feature extractor
        self.feature_extractor = feature_extractor
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

        # if self.use_gnn:
        self.gnn1 = simpleConv(feat_dim, feat_dim)
        self.gnn2 = simpleConv(feat_dim, feat_dim)

        self.fc_xyz = nn.Linear(feat_dim, 3)
        self.fc_wpqr = nn.Linear(feat_dim, 3)

        self.fc_xyz_R = nn.Linear(feat_dim * 2, 3)
        self.fc_wpqr_R = nn.Linear(feat_dim * 2, 3)

        if filter_nans:
            self.fc_wpqr.register_backward_hook(hook=filter_hook)

        # initialize
        if pretrained:
            init_modules = [self.feature_extractor.fc, self.fc_xyz, self.fc_wpqr, self.gnn1,
                            self.gnn2]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def compute_edge_features(self, e, edge_index):
        num_edges = edge_index.size(1)
        edge_feat = Variable(torch.zeros((num_edges, e.size(1) * 2)),
                             requires_grad=True).cuda()

        for i in range(num_edges):
            nodes = edge_index[:, i]
            node_1 = torch.min(nodes)
            node_2 = torch.max(nodes)
            edge_feat[i] = torch.cat((e[node_1], e[node_2]), 0)

        return edge_feat

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = x.view(x.size(0), 3, self.input_img_height, -1).contiguous()
        x = self.feature_extractor(x)
        x = F.relu(x)

        if self.use_gnn:
            x = self.gnn1(x, edge_index)
            x = F.relu(x)
            # x = self.gnn2(x, edge_index)
            # x = F.relu(x)

        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)

        # Compute edge features
        edge_feat = self.compute_edge_features(x, edge_index)

        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)

        xyz_R = self.fc_xyz_R(edge_feat)
        wpqr_R = self.fc_wpqr_R(edge_feat)

        return torch.cat((xyz, wpqr), 1), torch.cat((xyz_R, wpqr_R), 1)


class PoseNetX3(nn.Module):
    """PosentX + edge features + VO loss"""

    def __init__(self, feature_extractor, feature_extractor_2, droprate=0.5, pretrained=True,
                 feat_dim=2048, edge_feat_dim=2048, filter_nans=False, input_img_height=256,
                 use_gnn=False):
        super(PoseNetX3, self).__init__()
        self.droprate = droprate
        self.input_img_height = input_img_height
        self.use_gnn = use_gnn

        # replace the last FC layer in feature extractor
        self.feature_extractor = feature_extractor
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

        # replace the last FC layer in feature extractor
        self.edge_feature_extractor = feature_extractor_2
        self.edge_feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.edge_feature_extractor.fc.in_features
        self.edge_feature_extractor.fc = nn.Linear(fe_out_planes, edge_feat_dim)
        self.edge_feature_extractor.conv1 = nn.Conv2d(in_channels=6, out_channels=64,
                                                      kernel_size=(7, 7), stride=(2, 2),
                                                      padding=(3, 3), bias=False)

        # if self.use_gnn:
        self.gnn1 = simpleConvEdge(feat_dim, edge_feat_dim, feat_dim)
        self.gnn2 = simpleConvEdge(feat_dim, edge_feat_dim, feat_dim)
        # self.gnn3 = simpleConvEdge(feat_dim, edge_feat_dim, feat_dim)
        # self.gnn4 = simpleConvEdge(feat_dim, edge_feat_dim, feat_dim)

        # self.bn1 = nn.BatchNorm1d(feat_dim)
        # self.bn2 = nn.BatchNorm1d(feat_dim)
        # self.bn_e1 = nn.BatchNorm1d(edge_feat_dim)
        # self.bn_e2 = nn.BatchNorm1d(edge_feat_dim)

        self.fc_xyz = nn.Linear(feat_dim, 3)
        self.fc_wpqr = nn.Linear(feat_dim, 3)

        self.fc_xyz_R = nn.Linear(edge_feat_dim, 3)
        self.fc_wpqr_R = nn.Linear(edge_feat_dim, 3)

        if filter_nans:
            self.fc_wpqr.register_backward_hook(hook=filter_hook)

        # initialize
        if pretrained:
            init_modules = [self.feature_extractor.fc,
                            self.edge_feature_extractor.conv1, self.edge_feature_extractor.fc,
                            self.fc_xyz, self.fc_wpqr, self.fc_xyz_R, self.fc_wpqr_R,
                            self.gnn1, self.gnn2]  # , self.gnn3] #, self.gnn4]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def compute_edge_features(self, e, edge_index):
        num_edges = edge_index.size(1)
        if len(e.shape) == 4:
            edge_feat = Variable(torch.zeros((num_edges, e.size(1) * 2, e.size(2), e.size(3))),
                                 requires_grad=True).cuda()
        else:
            edge_feat = Variable(torch.zeros((num_edges, e.size(1) * 2)),
                                 requires_grad=True).cuda()

        for i in range(num_edges):
            nodes = edge_index[:, i]
            node_1 = torch.min(nodes)
            node_2 = torch.max(nodes)
            edge_feat[i] = torch.cat((e[node_1], e[node_2]), 0)

        return edge_feat

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = x.view(x.size(0), 3, self.input_img_height, -1).contiguous()

        edge_feat = self.compute_edge_features(x, edge_index)
        x = self.feature_extractor(x)
        x = F.relu(x)

        # Compute edge features
        edge_feat = self.edge_feature_extractor(edge_feat)
        edge_feat = F.relu(edge_feat)

        if self.use_gnn:
            x, edge_feat = self.gnn1(x, edge_index, edge_feat)
            x = F.relu(x)
            # x = self.bn1(x)
            edge_feat = F.relu(edge_feat)
            # edge_feat = self.bn_e1(edge_feat)

            # x = F.dropout(x, p=self.droprate)
            # edge_feat = F.dropout(edge_feat, p=self.droprate)

            x, edge_feat = self.gnn2(x, edge_index, edge_feat)
            x = F.relu(x)
            # x = self.bn2(x)
            edge_feat = F.relu(edge_feat)
            # edge_feat = self.bn_e2(edge_feat)

            # x, edge_feat = self.gnn3(x, edge_index, edge_feat)
            # x = F.relu(x)
            # edge_feat = F.relu(edge_feat)

            # x, edge_feat = self.gnn4(x, edge_index, edge_feat)
            # x = F.relu(x)
            # edge_feat = F.relu(edge_feat)

        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)

        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)

        xyz_R = self.fc_xyz_R(edge_feat)
        wpqr_R = self.fc_wpqr_R(edge_feat)

        return torch.cat((xyz, wpqr), 1), torch.cat((xyz_R, wpqr_R), 1)


class PoseNetX_LIGHT(nn.Module):
    """PosentX + edge features + VO loss"""

    def __init__(self, feature_extractor, droprate=0.5, pretrained=True,
                 feat_dim=2048, edge_feat_dim=2048, filter_nans=False, input_img_height=256,
                 use_gnn=False):
        super(PoseNetX_LIGHT, self).__init__()
        self.droprate = droprate
        self.input_img_height = input_img_height
        self.use_gnn = use_gnn

        # replace the last FC layer in feature extractor
        self.feature_extractor = feature_extractor
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

        self.proj_edge = nn.Linear(feat_dim * 2, edge_feat_dim)

        if self.use_gnn:
            self.gnn1 = simpleConvEdge(feat_dim, edge_feat_dim, feat_dim)
            self.gnn2 = simpleConvEdge(feat_dim, edge_feat_dim, feat_dim)
            # self.gnn3 = simpleConvEdge(feat_dim, edge_feat_dim, feat_dim)
            # self.gnn4 = simpleConvEdge(feat_dim, edge_feat_dim, feat_dim)

        self.fc_xyz = nn.Linear(feat_dim, 3)
        self.fc_wpqr = nn.Linear(feat_dim, 3)

        self.fc_xyz_R = nn.Linear(edge_feat_dim, 3)
        self.fc_wpqr_R = nn.Linear(edge_feat_dim, 3)

        if filter_nans:
            self.fc_wpqr.register_backward_hook(hook=filter_hook)

        # initialize
        if pretrained:
            init_modules = [self.feature_extractor.fc, self.proj_edge,
                            self.fc_xyz, self.fc_wpqr, self.fc_xyz_R, self.fc_wpqr_R,
                            self.gnn1, self.gnn2]  # , self.gnn3, self.gnn4]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def compute_edge_features(self, e, edge_index):
        num_edges = edge_index.size(1)
        if len(e.shape) == 4:
            edge_feat = Variable(torch.zeros((num_edges, e.size(1) * 2, e.size(2), e.size(3))),
                                 requires_grad=True).cuda()
        else:
            edge_feat = Variable(torch.zeros((num_edges, e.size(1) * 2)),
                                 requires_grad=True).cuda()

        for i in range(num_edges):
            nodes = edge_index[:, i]
            node_1 = torch.min(nodes)
            node_2 = torch.max(nodes)
            edge_feat[i] = torch.cat((e[node_1], e[node_2]), 0)

        return edge_feat

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = x.view(x.size(0), 3, self.input_img_height, -1).contiguous()

        x = self.feature_extractor(x)

        # Compute edge features
        edge_feat = self.compute_edge_features(x, edge_index)
        edge_feat = self.proj_edge(edge_feat)

        x = F.relu(x)
        edge_feat = F.relu(edge_feat)

        if self.use_gnn:
            x, edge_feat = self.gnn1(x, edge_index, edge_feat)
            x = F.relu(x)
            edge_feat = F.relu(edge_feat)

            # x = F.dropout(x, p=self.droprate)
            # edge_feat = F.dropout(edge_feat, p=self.droprate)

            x, edge_feat = self.gnn2(x, edge_index, edge_feat)
            x = F.relu(x)
            edge_feat = F.relu(edge_feat)

            # x, edge_feat = self.gnn3(x, edge_index, edge_feat)
            # x = F.relu(x)
            # edge_feat = F.relu(edge_feat)
            #
            # x, edge_feat = self.gnn4(x, edge_index, edge_feat)
            # x = F.relu(x)
            # edge_feat = F.relu(edge_feat)

        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)
            edge_feat = F.dropout(edge_feat, p=self.droprate)

        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)

        xyz_R = self.fc_xyz_R(edge_feat)
        wpqr_R = self.fc_wpqr_R(edge_feat)

        return torch.cat((xyz, wpqr), 1), torch.cat((xyz_R, wpqr_R), 1)


class PoseNetXOX(nn.Module):
    """PosentX + edge features + VO loss - no absolute pose modeling at all"""

    def __init__(self, feature_extractor_2, droprate=0.5, pretrained=True,
                 feat_dim=1024, edge_feat_dim=1024, filter_nans=False, input_img_height=256,
                 use_gnn=False):
        super(PoseNetXOX, self).__init__()
        self.droprate = droprate
        self.input_img_height = input_img_height
        self.use_gnn = use_gnn
        self.feat_dim = feat_dim

        # replace the last FC layer in feature extractor
        # self.feature_extractor = feature_extractor
        # self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        # fe_out_planes = self.feature_extractor.fc.in_features
        # self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

        # replace the last FC layer in feature extractor
        self.edge_feature_extractor = feature_extractor_2
        self.edge_feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.edge_feature_extractor.fc.in_features
        self.edge_feature_extractor.fc = nn.Linear(fe_out_planes, edge_feat_dim)
        self.edge_feature_extractor.conv1 = nn.Conv2d(in_channels=6, out_channels=64,
                                                      kernel_size=(7, 7), stride=(2, 2),
                                                      padding=(3, 3), bias=False)

        # if self.use_gnn:
        self.gnn1 = simpleConvEdge(feat_dim, edge_feat_dim, feat_dim)
        self.gnn2 = simpleConvEdge(feat_dim, edge_feat_dim, feat_dim)
        # self.gnn3 = simpleConvEdge(feat_dim, edge_feat_dim, feat_dim)
        # self.gnn4 = simpleConvEdge(feat_dim, edge_feat_dim, feat_dim)

        # self.fc_xyz  = nn.Linear(feat_dim, 3)
        # self.fc_wpqr = nn.Linear(feat_dim, 3)

        self.fc_xyz_R = nn.Linear(edge_feat_dim, 3)
        self.fc_wpqr_R = nn.Linear(edge_feat_dim, 3)

        # if filter_nans:
        # self.fc_wpqr.register_backward_hook(hook=filter_hook)

        # initialize
        if pretrained:
            init_modules = [self.edge_feature_extractor.conv1, self.edge_feature_extractor.fc,
                            self.fc_xyz_R, self.fc_wpqr_R,
                            self.gnn1, self.gnn2]  # , self.gnn3] #, self.gnn4]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def compute_edge_features(self, e, edge_index):
        num_edges = edge_index.size(1)
        if len(e.shape) == 4:
            edge_feat = Variable(torch.zeros((num_edges, e.size(1) * 2, e.size(2), e.size(3))),
                                 requires_grad=True).cuda()
        else:
            edge_feat = Variable(torch.zeros((num_edges, e.size(1) * 2)),
                                 requires_grad=True).cuda()

        for i in range(num_edges):
            nodes = edge_index[:, i]
            node_1 = torch.min(nodes)
            node_2 = torch.max(nodes)
            edge_feat[i] = torch.cat((e[node_1], e[node_2]), 0)

        return edge_feat

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = x.view(x.size(0), 3, self.input_img_height, -1).contiguous()

        edge_feat = self.compute_edge_features(x, edge_index)
        # x = self.feature_extractor(x)
        # x = F.relu(x)

        x = Variable(torch.zeros(x.size(0), self.feat_dim), requires_grad=False).cuda()

        # Compute edge features
        edge_feat = self.edge_feature_extractor(edge_feat)
        edge_feat = F.relu(edge_feat)

        if self.use_gnn:
            x, edge_feat = self.gnn1(x, edge_index, edge_feat)
            x = F.relu(x)
            edge_feat = F.relu(edge_feat)

            # x = F.dropout(x, p=self.droprate)
            # edge_feat = F.dropout(edge_feat, p=self.droprate)

            x, edge_feat = self.gnn2(x, edge_index, edge_feat)
            # x = F.relu(x)
            edge_feat = F.relu(edge_feat)

            # x, edge_feat = self.gnn3(x, edge_index, edge_feat)
            # x = F.relu(x)
            # edge_feat = F.relu(edge_feat)

            # x, edge_feat = self.gnn4(x, edge_index, edge_feat)
            # x = F.relu(x)
            # edge_feat = F.relu(edge_feat)

        if self.droprate > 0:
            # x = F.dropout(x, p=self.droprate)
            edge_feat = F.dropout(edge_feat, p=self.droprate)

        # xyz  = self.fc_xyz(x)
        # wpqr = self.fc_wpqr(x)

        xyz_R = self.fc_xyz_R(edge_feat)
        wpqr_R = self.fc_wpqr_R(edge_feat)

        return torch.cat((xyz_R[:x.size(0), ...], wpqr_R[:x.size(0), ...]), 1), torch.cat(
            (xyz_R, wpqr_R), 1)


class PoseNetX_LIGHT_KNN(nn.Module):
    """PosentX + edge features + VO loss"""

    def __init__(self, feature_extractor, droprate=0.5, pretrained=True,
                 feat_dim=2048, edge_feat_dim=2048, node_dim=2048,
                 filter_nans=False, input_img_height=256, use_gnn=False, use_attention=False,
                 knn=4, use_AP=True):
        super(PoseNetX_LIGHT_KNN, self).__init__()
        self.droprate = droprate
        self.input_img_height = input_img_height
        self.use_gnn = use_gnn
        self.use_attention = use_attention
        self.knn = knn
        self.use_AP = use_AP

        # replace the last FC layer in feature extractor
        self.feature_extractor = feature_extractor
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

        # self.proj_node = nn.Linear(feat_dim, node_dim)
        self.proj_edge = nn.Linear(feat_dim * 2, edge_feat_dim)

        if self.use_gnn:
            self.gnn1 = simpleConvEdge_upt(node_dim, edge_feat_dim, node_dim)
            self.gnn2 = simpleConvEdge_upt(node_dim, edge_feat_dim, node_dim)
            # self.gnn3 = simpleConvEdge_upt(feat_dim, edge_feat_dim, feat_dim)
            # self.gnn4 = simpleConvEdge(feat_dim, edge_feat_dim, feat_dim)
        else:
            self.mlp = nn.Sequential(nn.Linear(edge_feat_dim, edge_feat_dim),
                                     nn.ReLU(),
                                     nn.Linear(edge_feat_dim, edge_feat_dim))

        if self.use_attention:
            self.att = AttentionBlock(feat_dim)

        if self.use_AP:
            self.fc_xyz = nn.Linear(node_dim, 3)
            self.fc_wpqr = nn.Linear(node_dim, 3)
        else:
            self.fc_xyz = nn.Linear(node_dim * 2, 3)
            self.fc_wpqr = nn.Linear(node_dim * 2, 3)

        self.fc_xyz_R = nn.Linear(node_dim, 3)
        self.fc_wpqr_R = nn.Linear(node_dim, 3)

        if filter_nans:
            self.fc_wpqr.register_backward_hook(hook=filter_hook)

        # initialize
        if pretrained:
            if self.use_gnn:
                init_modules = [self.feature_extractor.fc, self.proj_edge,
                                self.fc_xyz, self.fc_wpqr, self.fc_xyz_R, self.fc_wpqr_R,
                                self.gnn1, self.gnn2]  # , self.gnn3] #, self.gnn4]
            else:
                init_modules = [self.feature_extractor.fc, self.proj_edge,
                                self.fc_xyz, self.fc_wpqr, self.fc_xyz_R, self.fc_wpqr_R,
                                self.mlp]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def compute_edge_features(self, e, edge_index):
        num_edges = edge_index.size(1)
        if len(e.shape) == 4:
            edge_feat = Variable(torch.zeros((num_edges, e.size(1) * 2, e.size(2), e.size(3))),
                                 requires_grad=True).cuda()
        else:
            edge_feat = Variable(torch.zeros((num_edges, e.size(1) * 2)),
                                 requires_grad=True).cuda()

        for i in range(num_edges):
            nodes = edge_index[:, i]
            node_1 = torch.min(nodes)
            node_2 = torch.max(nodes)
            edge_feat[i] = torch.cat((e[node_1], e[node_2]), 0)

        return edge_feat

    def compute_RP(self, p, edge_index):
        num_edges = edge_index.size(1)
        RP = Variable(torch.zeros((num_edges, p.size(1))), requires_grad=True).cuda()

        for i in range(num_edges):
            nodes = edge_index[:, i]
            node_s = nodes[0]
            node_t = nodes[1]
            RP[i] = p[node_s] - p[node_t]

        return RP

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = x.view(x.size(0), 3, self.input_img_height, -1).contiguous()

        x = self.feature_extractor(x)

        # Attention
        if self.use_attention:
            x = self.att(x.view(x.size(0), -1))

        # #KNN graph
        if self.knn > 0:
            edge_index = knn_graph(x, k=self.knn, batch=data.batch, loop=False)

        # Compute edge features
        edge_feat = self.compute_edge_features(x, edge_index)
        edge_feat = self.proj_edge(edge_feat)
        edge_feat = F.relu(edge_feat)

        # x = self.proj_node(x)
        # x = F.relu(x)

        if self.use_gnn:
            x, edge_feat = self.gnn1(x, edge_index, edge_feat)
            x = F.relu(x)
            edge_feat = F.relu(edge_feat)

            # x = F.dropout(x, p=self.droprate)
            # edge_feat = F.dropout(edge_feat, p=self.droprate)

            x, edge_feat = self.gnn2(x, edge_index, edge_feat)
            x = F.relu(x)
            edge_feat = F.relu(edge_feat)

        else:
            edge_feat = self.mlp(edge_feat)

        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)
            edge_feat = F.dropout(edge_feat, p=self.droprate)

        if self.use_AP:
            xyz = self.fc_xyz(x)
            wpqr = self.fc_wpqr(x)
        else:
            x2 = self.compute_edge_features(x, edge_index)
            xyz = self.fc_xyz(x2)
            wpqr = self.fc_wpqr(x2)

        xyz_R = self.fc_xyz_R(edge_feat)
        wpqr_R = self.fc_wpqr_R(edge_feat)

        return torch.cat((xyz, wpqr), 1), torch.cat((xyz_R, wpqr_R), 1), edge_index


class PoseNetX_R4(nn.Module):
    """PosentX + edge features + VO loss"""

    def __init__(self, feature_extractor, droprate=0.5, pretrained=True,
                 feat_dim=1024, edge_feat_dim=1024, node_dim=1024,
                 filter_nans=False, input_img_height=256, use_gnn=False, use_attention=False,
                 knn=4, use_AP=True):
        super(PoseNetX_R4, self).__init__()
        self.droprate = droprate
        self.input_img_height = input_img_height
        self.use_gnn = use_gnn
        self.use_attention = use_attention
        self.knn = knn
        self.use_AP = use_AP

        # replace the last FC layer in feature extractor
        self.feature_extractor = feature_extractor
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

        # self.proj_node = nn.Linear(feat_dim, node_dim)
        self.proj_edge = nn.Linear(feat_dim * 2, edge_feat_dim)

        if self.use_gnn:
            self.gnn1 = simpleConvEdge_upt(node_dim, edge_feat_dim, node_dim)
            self.gnn2 = simpleConvEdge_upt(node_dim, edge_feat_dim, node_dim)
            # self.gnn3 = simpleConvEdge_upt(feat_dim, edge_feat_dim, feat_dim)
            # self.gnn4 = simpleConvEdge(feat_dim, edge_feat_dim, feat_dim)
        else:
            self.mlp = nn.Sequential(nn.Linear(edge_feat_dim, edge_feat_dim),
                                     nn.ReLU(),
                                     nn.Linear(edge_feat_dim, edge_feat_dim))

        if self.use_attention:
            self.att = AttentionBlock(feat_dim)

        if self.use_AP:
            self.fc_xyz = nn.Linear(node_dim, 3)
            self.fc_wpqr = nn.Linear(node_dim, 3)
        else:
            self.fc_xyz = nn.Linear(node_dim * 2, 3)
            self.fc_wpqr = nn.Linear(node_dim * 2, 3)

        self.fc_xyz_R = nn.Linear(node_dim, 3)
        self.fc_wpqr_R = nn.Linear(node_dim, 3)

        if filter_nans:
            self.fc_wpqr.register_backward_hook(hook=filter_hook)

        # initialize
        if pretrained:
            if self.use_gnn:
                init_modules = [self.feature_extractor.fc, self.proj_edge,
                                self.fc_xyz, self.fc_wpqr, self.fc_xyz_R, self.fc_wpqr_R,
                                self.gnn1, self.gnn2]  # , self.gnn3] #, self.gnn4]
            else:
                init_modules = [self.feature_extractor.fc, self.proj_edge,
                                self.fc_xyz, self.fc_wpqr, self.fc_xyz_R, self.fc_wpqr_R,
                                self.mlp]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def compute_edge_features(self, e, edge_index):
        num_edges = edge_index.size(1)
        if len(e.shape) == 4:
            edge_feat = Variable(torch.zeros((num_edges, e.size(1) * 2, e.size(2), e.size(3))),
                                 requires_grad=True).cuda()
        else:
            edge_feat = Variable(torch.zeros((num_edges, e.size(1) * 2)),
                                 requires_grad=True).cuda()

        for i in range(num_edges):
            nodes = edge_index[:, i]
            node_1 = torch.min(nodes)
            node_2 = torch.max(nodes)
            edge_feat[i] = torch.cat((e[node_1], e[node_2]), 0)

        return edge_feat

    def compute_RP(self, p, edge_index):
        num_edges = edge_index.size(1)
        RP = Variable(torch.zeros((num_edges, p.size(1))), requires_grad=True).cuda()

        for i in range(num_edges):
            nodes = edge_index[:, i]
            node_s = nodes[0]
            node_t = nodes[1]
            RP[i] = p[node_s] - p[node_t]

        return RP

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = x.view(x.size(0), 3, self.input_img_height, -1).contiguous()

        x = self.feature_extractor(x)

        # Attention
        if self.use_attention:
            x = self.att(x.view(x.size(0), -1))

        # #KNN graph
        # if self.knn > 0:
        #   edge_index = knn_graph(x, k=self.knn, batch=data.batch, loop=False)
        # else:
        #   edge_index = knn_graph(x, k=8, batch=data.batch, loop=False)

        # Compute edge features
        edge_feat = self.compute_edge_features(x, edge_index)
        edge_feat = self.proj_edge(edge_feat)
        edge_feat = F.relu(edge_feat)

        # x = self.proj_node(x)
        # x = F.relu(x)

        if self.use_gnn:
            x, edge_feat = self.gnn1(x, edge_index, edge_feat)
            x = F.relu(x)
            edge_feat = F.relu(edge_feat)

            x = F.dropout(x, p=self.droprate)
            edge_feat = F.dropout(edge_feat, p=self.droprate)

            x, edge_feat = self.gnn2(x, edge_index, edge_feat)
            x = F.relu(x)
            edge_feat = F.relu(edge_feat)

        else:
            edge_feat = self.mlp(edge_feat)

        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)
            edge_feat = F.dropout(edge_feat, p=self.droprate)

        if self.use_AP:
            xyz = self.fc_xyz(x)
            wpqr = self.fc_wpqr(x)
        else:
            x2 = self.compute_edge_features(x, edge_index)
            xyz = self.fc_xyz(x2)
            wpqr = self.fc_wpqr(x2)

        xyz_R = self.fc_xyz_R(edge_feat)
        wpqr_R = self.fc_wpqr_R(edge_feat)

        return torch.cat((xyz, wpqr), 1), torch.cat((xyz_R, wpqr_R), 1), edge_index


class PoseNetX_R2(nn.Module):
    """PosentX + edge features + VO loss"""

    def __init__(self, feature_extractor, droprate=0.5, pretrained=True,
                 feat_dim=1024, edge_feat_dim=1024, node_dim=1024,
                 filter_nans=False, input_img_height=256, use_gnn=False, use_attention=False,
                 knn=-1, use_AP=True,
                 gnn_recursion=2,
                 device: int = 0,
                 L: int = 1):
        super(PoseNetX_R2, self).__init__()
        self.droprate = droprate
        self.input_img_height = input_img_height
        self.use_gnn = use_gnn
        self.n_layers = L
        self.use_attention = use_attention
        self.knn = knn
        self.use_AP = use_AP
        self.gnn_recursion = gnn_recursion
        self.device = device

        # replace the last FC layer in feature extractor
        self.feature_extractor = feature_extractor
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

        # self.proj_node = nn.Linear(feat_dim, node_dim)
        self.proj_edge = nn.Linear(feat_dim * 2, edge_feat_dim)

        if self.use_gnn:
            for layer in range(L):
                setattr(self, f'gnn{layer + 1}',
                        simpleConvEdge_upt(node_dim, edge_feat_dim, node_dim))
                # self.gnn1 = simpleConvEdge_upt(node_dim, edge_feat_dim, node_dim)
            # if 1 < self.n_layers:
            #     self.gnn2 = simpleConvEdge_upt(node_dim, edge_feat_dim, node_dim)
            # if 2 < self.n_layers:
            #     raise NotImplementedError(f'More than 2 layers not implemented')
        # else:
        #   self.mlp = nn.Sequential(nn.Linear(edge_feat_dim  , edge_feat_dim),
        #                  nn.ReLU(),
        #                  nn.Linear(edge_feat_dim, edge_feat_dim))

        if self.use_attention:
            self.att = AttentionBlock(feat_dim)

        if self.use_AP:
            self.fc_xyz = nn.Linear(node_dim, 3)
            self.fc_wpqr = nn.Linear(node_dim, 3)
        else:
            self.fc_xyz = nn.Linear(node_dim * 2, 3)
            self.fc_wpqr = nn.Linear(node_dim * 2, 3)

        self.fc_xyz_R = nn.Linear(node_dim, 3)
        self.fc_wpqr_R = nn.Linear(node_dim, 3)

        if filter_nans:
            self.fc_wpqr.register_backward_hook(hook=filter_hook)

        # initialize
        if pretrained:
            if self.use_gnn:
                init_modules = [self.feature_extractor.fc, self.proj_edge,
                                self.fc_xyz, self.fc_wpqr, self.fc_xyz_R, self.fc_wpqr_R,
                                self.gnn1]  # , self.gnn2]#, self.gnn3] #, self.gnn4]
            else:
                init_modules = [self.feature_extractor.fc, self.proj_edge,
                                self.fc_xyz, self.fc_wpqr, self.fc_xyz_R, self.fc_wpqr_R,
                                self.mlp]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def compute_edge_features(self, e, edge_index):
        # num_edges = edge_index.size(1)
        # if len(e.shape) == 4:
        #     edge_feat = Variable(torch.zeros((num_edges, e.size(1) * 2, e.size(2), e.size(3))),
        #                          requires_grad=True).to(self.device)
        # else:
        #     edge_feat = Variable(torch.zeros((num_edges, e.size(1) * 2)),
        #                          requires_grad=True).to(self.device)
        #
        # for i in range(num_edges):
        #     nodes = edge_index[:, i]
        #     node_1 = torch.min(nodes)
        #     node_2 = torch.max(nodes)
        #     edge_feat[i] = torch.cat((e[node_1], e[node_2]), 0)
        #
        edge_feat = torch.cat(
            (e[torch.min(edge_index, 0)[0], ...],
             e[torch.max(edge_index, 0)[0], ...]),
            dim=1)
        # torch.testing.assert_allclose(edge_feat, edge_feat2)
        return edge_feat

    def compute_RP(self, p, edge_index):
        num_edges = edge_index.size(1)
        RP = Variable(torch.zeros((num_edges, p.size(1))), requires_grad=True).to(self.device)

        for i in range(num_edges):
            nodes = edge_index[:, i]
            node_s = nodes[0]
            node_t = nodes[1]
            RP[i] = p[node_s] - p[node_t]

        return RP

    def forward(self, data, k=None):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = x.view(x.size(0), 3, self.input_img_height, -1).contiguous()

        x = self.feature_extractor(x)

        # Attention
        if self.use_attention:
            x = self.att(x.view(x.size(0), -1))

        if k is not None:
            edge_index_knn = knn_graph(x, k=k, batch=data.batch, loop=False)

        # #KNN graph
        if self.knn > 0:
            edge_index = knn_graph(x, k=self.knn, batch=data.batch, loop=False)
        elif k is not None:
            edge_index = knn_graph(x, k=k, batch=data.batch, loop=False)

        # Compute edge features
        edge_feat = self.compute_edge_features(x, edge_index)
        edge_feat = self.proj_edge(edge_feat)
        edge_feat = F.relu(edge_feat)

        # x = self.proj_node(x)
        # x = F.relu(x)

        if self.use_gnn:
            for r in range(self.gnn_recursion):
                if r == 0:
                    x, edge_feat = self.gnn1(x, edge_index, edge_feat)
                    x = F.relu(x)
                    edge_feat = F.relu(edge_feat)
                else:
                    x, edge_feat = self.gnn1(x, edge_index, edge_feat)
                    x = F.relu(x)
                    edge_feat = F.relu(edge_feat)
        else:
            edge_feat = self.mlp(edge_feat)

        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)
            edge_feat = F.dropout(edge_feat, p=self.droprate)

        if self.use_AP:
            xyz = self.fc_xyz(x)
            wpqr = self.fc_wpqr(x)
        else:
            x2 = self.compute_edge_features(x, edge_index)
            xyz = self.fc_xyz(x2)
            wpqr = self.fc_wpqr(x2)

        xyz_R = self.fc_xyz_R(edge_feat)
        wpqr_R = self.fc_wpqr_R(edge_feat)

        if k is not None:
            return torch.cat((xyz, wpqr), 1), torch.cat((xyz_R, wpqr_R), 1), edge_index_knn
        else:
            return torch.cat((xyz, wpqr), 1), torch.cat((xyz_R, wpqr_R), 1), edge_index


class PoseNetX_R3(nn.Module):
    """PosentX + edge features + VO loss"""

    def __init__(self, feature_extractor, droprate=0.5, pretrained=True,
                 feat_dim=1024, edge_feat_dim=1024, node_dim=1024,
                 filter_nans=False, input_img_height=256, use_gnn=False, use_attention=False,
                 knn=4, use_AP=True,
                 gnn_recursion=2):
        super(PoseNetX_R3, self).__init__()
        self.droprate = droprate
        self.input_img_height = input_img_height
        self.use_gnn = use_gnn
        self.use_attention = use_attention
        self.knn = knn
        self.use_AP = use_AP
        self.gnn_recursion = gnn_recursion

        # replace the last FC layer in feature extractor
        self.feature_extractor = feature_extractor
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

        # self.proj_node = nn.Linear(feat_dim, node_dim)
        self.proj_edge = nn.Linear(feat_dim * 2, edge_feat_dim)

        if self.use_gnn:
            self.gnn1 = simpleConvEdge_upt(node_dim, edge_feat_dim, node_dim)
            self.gnn2 = simpleConvEdge_upt(node_dim, edge_feat_dim, node_dim)
        else:
            self.mlp = nn.Sequential(nn.Linear(edge_feat_dim, edge_feat_dim),
                                     nn.ReLU(),
                                     nn.Linear(edge_feat_dim, edge_feat_dim))

        if self.use_attention:
            self.att = AttentionBlock(feat_dim)

        if self.use_AP:
            self.fc_xyz = nn.Linear(node_dim, 3)
            self.fc_wpqr = nn.Linear(node_dim, 3)
        else:
            self.fc_xyz = nn.Linear(node_dim * 2, 3)
            self.fc_wpqr = nn.Linear(node_dim * 2, 3)

        self.fc_xyz_R = nn.Linear(node_dim, 3)
        self.fc_wpqr_R = nn.Linear(node_dim, 3)

        if filter_nans:
            self.fc_wpqr.register_backward_hook(hook=filter_hook)

        # initialize
        if pretrained:
            if self.use_gnn:
                init_modules = [self.feature_extractor.fc, self.proj_edge,
                                self.fc_xyz, self.fc_wpqr, self.fc_xyz_R, self.fc_wpqr_R,
                                self.gnn1, self.gnn2]  # , self.gnn3] #, self.gnn4]
            else:
                init_modules = [self.feature_extractor.fc, self.proj_edge,
                                self.fc_xyz, self.fc_wpqr, self.fc_xyz_R, self.fc_wpqr_R,
                                self.mlp]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def compute_edge_features(self, e, edge_index):
        num_edges = edge_index.size(1)
        if len(e.shape) == 4:
            edge_feat = Variable(torch.zeros((num_edges, e.size(1) * 2, e.size(2), e.size(3))),
                                 requires_grad=True).cuda()
        else:
            edge_feat = Variable(torch.zeros((num_edges, e.size(1) * 2)),
                                 requires_grad=True).cuda()

        for i in range(num_edges):
            nodes = edge_index[:, i]
            node_1 = torch.min(nodes)
            node_2 = torch.max(nodes)
            edge_feat[i] = torch.cat((e[node_1], e[node_2]), 0)

        return edge_feat

    def compute_RP(self, p, edge_index):
        num_edges = edge_index.size(1)
        RP = Variable(torch.zeros((num_edges, p.size(1))), requires_grad=True).cuda()

        for i in range(num_edges):
            nodes = edge_index[:, i]
            node_s = nodes[0]
            node_t = nodes[1]
            RP[i] = p[node_s] - p[node_t]

        return RP

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = x.view(x.size(0), 3, self.input_img_height, -1).contiguous()

        x = self.feature_extractor(x)

        # Attention
        if self.use_attention:
            x = self.att(x.view(x.size(0), -1))

        # #KNN graph
        # if self.knn > 0:
        #   edge_index = knn_graph(x, k=self.knn, batch=data.batch, loop=False)
        # else:
        #   edge_index = knn_graph(x, k=8, batch=data.batch, loop=False)

        # Compute edge features
        edge_feat = self.compute_edge_features(x, edge_index)
        edge_feat = self.proj_edge(edge_feat)
        edge_feat = F.relu(edge_feat)

        # x = self.proj_node(x)
        # x = F.relu(x)

        if self.use_gnn:
            for r in range(self.gnn_recursion):
                if r == 0:
                    x, edge_feat = self.gnn1(x, edge_index, edge_feat)
                    x = F.relu(x)
                    edge_feat = F.relu(edge_feat)
                else:
                    x, edge_feat = self.gnn1(x, edge_index, edge_feat)
                    x = F.relu(x)
                    edge_feat = F.relu(edge_feat)

            for r in range(self.gnn_recursion):
                if r == 0:
                    x, edge_feat = self.gnn2(x, edge_index, edge_feat)
                    x = F.relu(x)
                    edge_feat = F.relu(edge_feat)
                else:
                    x, edge_feat = self.gnn2(x, edge_index, edge_feat)
                    x = F.relu(x)
                    edge_feat = F.relu(edge_feat)

        else:
            edge_feat = self.mlp(edge_feat)

        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)
            edge_feat = F.dropout(edge_feat, p=self.droprate)

        if self.use_AP:
            xyz = self.fc_xyz(x)
            wpqr = self.fc_wpqr(x)
        else:
            x2 = self.compute_edge_features(x, edge_index)
            xyz = self.fc_xyz(x2)
            wpqr = self.fc_wpqr(x2)

        xyz_R = self.fc_xyz_R(edge_feat)
        wpqr_R = self.fc_wpqr_R(edge_feat)

        return torch.cat((xyz, wpqr), 1), torch.cat((xyz_R, wpqr_R), 1), edge_index


class PoseNet_nolog(nn.Module):
    """PosentX + edge features + VO loss"""

    def __init__(self, feature_extractor, droprate=0.5, pretrained=True,
                 feat_dim=2048, edge_feat_dim=2048, node_dim=2048,
                 filter_nans=False, input_img_height=256, use_gnn=False, use_attention=False,
                 knn=4, use_AP=True):
        super(PoseNet_nolog, self).__init__()
        self.droprate = droprate
        self.input_img_height = input_img_height
        self.use_gnn = use_gnn
        self.use_attention = use_attention
        self.knn = knn
        self.use_AP = use_AP

        # replace the last FC layer in feature extractor
        self.feature_extractor = feature_extractor
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

        self.fc_xyz = nn.Linear(node_dim, 3)
        self.fc_wpqr = nn.Linear(node_dim, 4)

        if filter_nans:
            self.fc_wpqr.register_backward_hook(hook=filter_hook)

        # initialize
        if pretrained:

            init_modules = [self.feature_extractor.fc,
                            self.fc_xyz, self.fc_wpqr]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = x.view(x.size(0), 3, self.input_img_height, -1).contiguous()

        x = self.feature_extractor(x)

        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)

        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)

        return torch.cat((xyz, wpqr), 1), 0, 0
