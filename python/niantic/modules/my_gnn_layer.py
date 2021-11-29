import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU
from torch.nn.functional import cosine_similarity
from torch_cluster import knn_graph
from torch_geometric.nn.conv import MessagePassing

from niantic.modules.att import AttentionBlock


def batch_cosine_similarity(x, y, pooling=None):
    b = x.size(0)
    c = x.size(1)

    if pooling is None:
        outx = x.view(b, c, -1)
        outy = y.view(b, c, -1)
    elif pooling == 'max':
        outx = F.adaptive_max_pool2d(x, 1)
        outy = F.adaptive_max_pool2d(y, 1)

        outx = outx.view(b, c, -1)
        outy = outy.view(b, c, -1)

    elif pooling == 'avg':
        outx = F.adaptive_avg_pool2d(x, 1)
        outy = F.adaptive_avg_pool2d(y, 1)

        outx = outx.view(b, c, -1)
        outy = outy.view(b, c, -1)

    corr = cosine_similarity(outx, outy, dim=2).view(b, c)
    # corr = Variable.sqrt(Variable.sum((outx - outy) ** 2, dim=2))
    # print("corr: ", corr.size())
    # print(corr)
    corr = torch.sigmoid(corr)
    # corr = torch.softmax(corr, dim=1)
    # print(corr)
    corr = corr.view(b, c, 1, 1)
    # return x
    # corr * x + y * (1 - corr)
    return corr


class myGNN(MessagePassing):
    def __init__(self, in_channels, out_channels, H, W, aggr="add", attention=False, pooling=None,
                 k=-1, first_GNN_layer=False, **kwargs):
        super(myGNN, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.H = H
        self.W = W
        self.attention = attention
        self.k = k
        self.pooling = pooling

        if first_GNN_layer:
            self.in_channels_edge = in_channels * 4
        else:
            self.in_channels_edge = in_channels * 3

        self.conv_message = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 3, out_channels=out_channels, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_edge = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels_edge, out_channels=out_channels, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_updating = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveMaxPool2d(1)

        # Define edge model
        self.edge_model = EdgeModel(self.conv_edge, self.in_channels, self.H, self.W,
                                    self.in_channels * 2)

        init_modules = [self.conv_message, self.conv_updating, self.conv_edge]

        self.init_parameters(init_modules=init_modules)

    def init_parameters(self, init_modules=None):
        if init_modules is None:
            return
        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x, edge_index, edge_attr, batch=None):

        # Edge Update
        if self.edge_model is not None:
            row, col = edge_index
            edge_attr = self.edge_model(x[row], x[col], edge_attr)

        # Node Update
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        if self.k > 0:
            if self.pooling is None:
                x_pooling = x
            elif self.pooling == 'max':
                x_full = x.view(x.size(0), self.in_channels, self.H, self.W).contiguous()
                x_pooling = F.adaptive_max_pool2d(x_full, 1).view(x.size(0), -1)
            elif self.pooling == 'avg':
                x_full = x.view(x.size(0), self.in_channels, self.H, self.W).contiguous()
                x_pooling = F.adaptive_avg_pool2d(x_full, 1).view(x.size(0), -1)
            edge_index = knn_graph(x_pooling, self.k, batch=batch, loop=False, flow=self.flow,
                                   cosine=True)
        return out, edge_index, edge_attr

    def message(self, x_i, x_j, edge_attr):
        """
        :param x_i: E x in_channels
        :param x_j: E x in_channels
        :return: message
        """
        # *******update edge features here possible
        # self.__edgemat__ = edgemat

        x_i_full = x_i.view(x_i.size(0), self.in_channels, self.H,
                            self.W).contiguous()  # recover the full size
        x_j_full = x_j.view(x_j.size(0), self.in_channels, self.H,
                            self.W).contiguous()  # recover the full size
        edge_attr = edge_attr.view(edge_attr.size(0), -1, self.H,
                                   self.W).contiguous()  # recover the full size

        message = self.conv_message(torch.cat([x_i_full, x_j_full, edge_attr], dim=1).contiguous())

        if self.attention:
            atten = batch_cosine_similarity(x_i_full, x_j_full, pooling=self.pooling)
            message = message * atten

        return message.view(message.size(0), -1)

    def update(self, aggr_out, x):
        # two 3x3 layers for feature fusion
        x = x.view(x.size(0), self.in_channels, self.H,
                   self.W).contiguous()  # recover the full size
        aggr_out = aggr_out.view(aggr_out.size(0), self.in_channels, self.H,
                                 self.W).contiguous()  # recover the full size

        out = self.conv_updating(torch.cat([x, aggr_out], dim=1).contiguous())
        # iout = self.conv2(out)
        return out.view(out.size(0), -1)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class EdgeModel(nn.Module):
    """
    Class used to perform the edge update during Neural message passing
    """

    def __init__(self, edge_CNN, in_channels, H, W, in_channels_edge):
        super(EdgeModel, self).__init__()
        self.in_channels, self.H, self.W = in_channels, H, W
        self.in_channels_edge = in_channels_edge
        self.edge_CNN = edge_CNN

    def forward(self, source, target, edge_attr):
        source_full = source.view(source.size(0), self.in_channels, self.H,
                                  self.W).contiguous()  # recover the full size
        target_full = target.view(target.size(0), self.in_channels, self.H,
                                  self.W).contiguous()  # recover the full size
        edge_attr = edge_attr.view(edge_attr.size(0), -1, self.H,
                                   self.W).contiguous()  # recover the full size

        out = torch.cat([source_full, target_full, edge_attr], dim=1).contiguous()
        out = self.edge_CNN(out)

        return out.view(out.size(0), -1)


class simpleEdgeModelAtt(nn.Module):
    """
    Class used to perform the edge update during Neural message passing
    """

    def __init__(self, in_channels, edge_channels, out_channels):
        super(simpleEdgeModelAtt, self).__init__()
        self.in_channels = in_channels
        self.edge_mlp = Seq(Linear(2 * in_channels + edge_channels, out_channels),
                            ReLU(),
                            Linear(out_channels, out_channels))

        self.att = AttentionBlock(in_channels)

    def forward(self, source, target, edge_attr):
        out = torch.cat([source, target, edge_attr], dim=1).contiguous()
        out = self.edge_mlp(out)
        out = self.att(out)
        return out


class simpleEdgeModel(nn.Module):
    """
    Class used to perform the edge update during Neural message passing
    """

    def __init__(self, in_channels, edge_channels, out_channels):
        super(simpleEdgeModel, self).__init__()
        self.in_channels = in_channels
        self.edge_mlp = Seq(Linear(2 * in_channels + edge_channels, out_channels),
                            ReLU(),
                            Linear(out_channels, out_channels))

    def forward(self, source, target, edge_attr):
        out = torch.cat([source, target, edge_attr], dim=1).contiguous()
        out = self.edge_mlp(out)
        return out


class simpleConvEdge(MessagePassing):
    def __init__(self, in_channels, edge_channels, out_channels, use_attention=True):
        super(simpleConvEdge, self).__init__(aggr='mean')  # "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels + edge_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

        self.edge_model = simpleEdgeModel(in_channels, edge_channels, edge_channels)

        if use_attention:
            self.att = AttentionBlock(in_channels)

    def forward(self, x, edge_index, edge_attr):
        # Edge Update
        if self.edge_model is not None:
            row, col = edge_index
            edge_attr = self.edge_model(x[row], x[col], edge_attr)

        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        out = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

        return out, edge_attr

    def message(self, x_i, x_j, edge_attr):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)  # tmp has shape [E, 2 * in_channels]
        tmp = self.mlp(tmp)
        tmp = self.att(tmp)

        return tmp


class simpleConvEdge_upt(MessagePassing):
    def __init__(self, in_channels, edge_channels, out_channels, use_attention=True):
        super(simpleConvEdge_upt, self).__init__(aggr='mean')  # "Max" aggregation.
        self.mlp = Seq(Linear(in_channels + edge_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

        self.mlp_updating = Seq(Linear(2 * in_channels, out_channels),
                                ReLU(),
                                Linear(out_channels, out_channels))

        self.edge_model = simpleEdgeModel(in_channels, edge_channels, edge_channels)

        if use_attention:
            self.att = AttentionBlock(in_channels)

    def forward(self, x, edge_index, edge_attr):
        # Edge Update
        if self.edge_model is not None:
            row, col = edge_index
            edge_attr = self.edge_model(x[row], x[col], edge_attr)

        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        out = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)
        return out, edge_attr

    def message(self, x_i, x_j, edge_attr):
        msg = self.mlp(torch.cat([x_j, edge_attr], dim=1))
        msg = self.att(msg)
        return msg

    def update(self, aggr_out, x):
        out = self.mlp_updating(torch.cat([x, aggr_out], dim=1))
        return out


class simpleConvEdge_upt_att(MessagePassing):
    def __init__(self, in_channels, edge_channels, out_channels, use_attention=True):
        super(simpleConvEdge_upt_att, self).__init__(aggr='mean')  # "Max" aggregation.
        self.mlp = Seq(Linear(in_channels + edge_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

        self.mlp_updating = Seq(Linear(2 * in_channels, out_channels),
                                ReLU(),
                                Linear(out_channels, out_channels))

        self.edge_model = simpleEdgeModel(in_channels, edge_channels, edge_channels)

        self.mlp_att = Seq(Linear(edge_channels, out_channels),
                           ReLU(),
                           Linear(out_channels, out_channels),
                           nn.Sigmoid())

    def forward(self, x, edge_index, edge_attr):
        # Edge Update
        if self.edge_model is not None:
            row, col = edge_index
            edge_attr = self.edge_model(x[row], x[col], edge_attr)

        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        out = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)
        return out, edge_attr

    def message(self, x_i, x_j, edge_attr):
        msg = self.mlp(torch.cat([x_i, x_j, edge_attr], dim=1))
        att = self.mlp_att(edge_attr)
        msg = msg * att

        return msg

    def update(self, aggr_out, x):
        out = self.mlp_updating(torch.cat([x, aggr_out], dim=1))
        return out


class simpleConvEdge2(MessagePassing):
    def __init__(self, in_channels, edge_channels, out_channels):
        super(simpleConvEdge2, self).__init__(aggr='mean')  # "Max" aggregation.
        self.mlp1 = Seq(Linear(2 * in_channels + edge_channels, out_channels),
                        ReLU(),
                        Linear(out_channels, out_channels))

        self.mlp2 = Seq(Linear(2 * in_channels + edge_channels, out_channels),
                        ReLU(),
                        Linear(out_channels, out_channels))

        self.edge_model = simpleEdgeModel(in_channels, edge_channels, edge_channels)

    def forward(self, x, edge_index, edge_attr):
        # Edge Update
        if self.edge_model is not None:
            row, col = edge_index
            edge_attr = self.edge_model(x[row], x[col], edge_attr)

        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        out = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

        return out, edge_attr

    def message(self, x_i, x_j, edge_attr):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)  # tmp has shape [E, 2 * in_channels]

        p1 = self.mlp1(tmp)
        # p1 = x_i[:, :3] + self.mlp1(tmp)
        p2 = self.mlp2(tmp)
        # p2 = x_i[:, 3:] + self.mlp2(tmp)

        return torch.cat([p1, p2], dim=1)


class simpleConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(simpleConv, self).__init__(aggr='mean')  # "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)


class EdgeConvRot(MessagePassing):
    def __init__(self, in_channels, edge_channels, out_channels):
        super(EdgeConvRot, self).__init__(aggr='mean',
                                          flow="target_to_source")  # "Max" aggregation.
        self.mlp0 = Seq(Linear(edge_channels, out_channels),
                        ReLU(),
                        Linear(out_channels, out_channels))

        self.mlp = Seq(Linear(2 * in_channels + edge_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        if x_i.size(1) > 5:
            W = torch.cat([torch.cat([x_i, x_j], dim=1), edge_attr],
                          dim=1)  # tmp has shape [E, 2 * in_channels]
            W = self.mlp(W)
        else:
            W = edge_attr  # torch.cat([torch.cat([x_i, x_j], dim=1), edge_attr], dim=1)  # tmp has shape [E, 2 * in_channels]
            W = self.mlp0(W)
        return W

    def propagate(self, edge_index, size, x, edge_attr):
        row, col = edge_index
        x_i = x[row]
        x_j = x[col]
        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        edge_out = self.message(x_i, x_j, edge_attr)
        out = scatter_(self.aggr, edge_out, edge_index[i], dim_size=size[i])
        return out, edge_out
