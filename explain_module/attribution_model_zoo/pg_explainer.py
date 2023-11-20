import copy
import math
import numpy as np

import torch

import warnings
from torch_geometric.nn import MessagePassing
from explain_module.utils.edge_mask_net import EdgeMaskNet
from explain_module.attribution_model_zoo.basic_explainer import Explainer

EPS = 1e-5
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PGExplainer(Explainer):
    coeffs = {
        'edge_size': 0.001,
        'edge_ent': 0.01,
    }

    def __init__(self, gnn_model,
                 n_in_channels=14,
                 e_in_channels=3,
                 hid=50, n_layers=2,
                 n_label=2
                 ):
        super(PGExplainer, self).__init__(gnn_model)

        self.edge_mask = []

        for i in range(n_label):
            self.edge_mask.append(EdgeMaskNet(
                n_in_channels,
                e_in_channels,
                hid=hid,
                n_layers=n_layers).to(device))

    def __set_masks__(self, mask, model):

        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = mask

    def __clear_masks__(self, model):
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None

    def __reparameterize__(self, log_alpha, beta=0.1, training=True):
        if training:
            random_noise = torch.rand(log_alpha.size()).to(device)
            gate_inputs = torch.log2(random_noise) - torch.log2(1.0 - random_noise)
            gate_inputs = (2*gate_inputs + log_alpha) / beta + EPS
            gate_inputs = gate_inputs.sigmoid()
            # gate_inputs=(log_alpha/beta+EPS).sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs

    def __loss__(self, log_logits, mask, pred_label):

        # loss = criterion(log_logits, pred_label)
        idx = [i for i in range(len(pred_label))]
        loss = -log_logits.softmax(dim=1)[idx, pred_label.view(-1)].sum()
        loss = loss + self.coeffs['edge_size'] * mask.sum()
        ent = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()
        return loss

    # batch version
    def pack_subgraph(self, graph, imp, top_ratio=0.2):

        if abs(top_ratio - 1.0) < EPS:
            return graph, imp

        exp_subgraph = copy.deepcopy(graph)
        top_idx = torch.LongTensor([])
        graph_map = graph.batch[graph.edge_index[0, :]]

        # extract ego graph
        for i in range(graph.num_graphs):
            edge_indicator = torch.where(graph_map == i)[0].detach().cpu()
            Gi_n_edge = len(edge_indicator)
            topk = max(math.ceil(top_ratio * Gi_n_edge), 1)

            Gi_pos_edge_idx = np.argsort(-imp[edge_indicator])[:topk]
            top_idx = torch.cat([top_idx, edge_indicator[Gi_pos_edge_idx]])

        exp_subgraph.edge_attr = graph.edge_attr[top_idx]
        exp_subgraph.edge_index = graph.edge_index[:, top_idx]
        exp_subgraph.x, exp_subgraph.edge_index, exp_subgraph.batch = \
            self.__relabel__(exp_subgraph, exp_subgraph.edge_index)

        return exp_subgraph, imp[top_idx]

    def get_mask(self, graph):
        # batch version
        graph_map = graph.batch[graph.edge_index[0, :]]
        mask = torch.FloatTensor([]).to(device)
        for i in range(len(graph.y)):
            edge_indicator = (graph_map == i).bool()
            G_i_mask = self.edge_mask[graph.y[i]](
                graph.x,
                graph.edge_index[:, edge_indicator],
                graph.edge_attr[edge_indicator, :]
            ).view(-1)
            mask = torch.cat([mask, G_i_mask])
        return mask

    def get_pos_edge(self, graph, mask, ratio):

        num_edge = [0]
        num_node = [0]
        sep_edge_idx = []
        graph_map = graph.batch[graph.edge_index[0, :]]
        pos_idx = torch.LongTensor([])
        mask = mask.detach().cpu()
        for i in range(graph.num_graphs):
            edge_indicator = torch.where(graph_map == i)[0].detach().cpu()
            Gi_n_edge = len(edge_indicator)
            topk = max(math.ceil(ratio * Gi_n_edge), 1)

            Gi_pos_edge_idx = np.argsort(-mask[edge_indicator])[:topk]

            pos_idx = torch.cat([pos_idx, edge_indicator[Gi_pos_edge_idx]])
            num_edge.append(num_edge[i] + Gi_n_edge)
            num_node.append(
                num_node[i] + (graph.batch == i).sum().long()
            )
            sep_edge_idx.append(Gi_pos_edge_idx)

        return pos_idx, num_edge, num_node, sep_edge_idx

    def explain_graph(self, graph, model=None,
                      temp=0.1, ratio=0.1,
                      draw_graph=0, vis_ratio=0.2,
                      train_mode=False, supplement=False
                      ):
        if model == None:
            model = self.model

        ori_mask = self.get_mask(graph)
        re_mask = self.__reparameterize__(ori_mask, training=True, beta=temp)

        imp = re_mask.detach().cpu().numpy()
        sum_imp=round(sum(imp) / len(imp), 3)
        # print(sum_imp)
        self.last_result = (graph, imp)

        if train_mode:
            # ----------------------------------------------------
            # (Reddit5k) batch version: get positive edge index(G_s) for ego graph
            pos_idx, num_edge, num_node, sep_edge_idx = self.get_pos_edge(graph, re_mask, ratio)
            pos_edge_mask = re_mask[pos_idx]
            #fjf eg.remask:[1,3,2,5,2,3] batch:[1,1,1,2,2,2] posmask:0.3[3,5]or0.8[3,2,5,3]

            # fjf获取子图精度
            pos_edge_index = graph.edge_index[:, pos_idx]
            pos_edge_attr = graph.edge_attr[pos_idx, :]
            G1_x, G1_pos_edge_index, G1_batch = self.__relabel__(graph, pos_edge_index)
            log_logits = self.model(
                x=G1_x,
                edge_index=G1_pos_edge_index,
                edge_attr=pos_edge_attr,
                batch=G1_batch
            )
            G1_acc = (graph.y == log_logits.argmax(dim=1)).detach().cpu().float().view(-1, 1)
            G_acc = G1_acc  # no supplement

            self.__set_masks__(pos_edge_mask, self.model)
            log_logits = self.model(
                x=G1_x,
                edge_index=G1_pos_edge_index,
                edge_attr=pos_edge_attr,
                batch=G1_batch
            )
            loss = self.__loss__(log_logits, re_mask, graph.y)

            self.__clear_masks__(self.model)
            return loss, G_acc.mean(dim=0).view(-1, 1), G1_acc, \
                   float(len(pos_idx)) / graph.edge_index.size(1), \
                   float(G1_pos_edge_index.size(1)) / graph.edge_index.size(1),sum_imp


        if draw_graph:
            self.visualize(graph, imp, 'PGExplainer', vis_ratio=vis_ratio)

        return imp


