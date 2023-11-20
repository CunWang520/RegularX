from math import sqrt
import torch
from torch_geometric.nn import MessagePassing

EPS = 1e-10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
GNNExplainer for interpreting graph prediction results from GNNs.

Citation: 
Based on torch_geometric.nn.models.GNNExplainer
which generates explainations in node prediction tasks.

Original Paper:
Ying et al. GNNExplainer: Generating Explanations for Graph Neural Networks.
"""

class MetaGNNGExplainer(torch.nn.Module):

    coeffs = {
        'edge_size': 0.05,
        'edge_ent': 0.5,
    }

    def __init__(self, r, model, epochs, lr=0.01, log=True):
        super(MetaGNNGExplainer, self).__init__()
        self.model = model
        self.model.to(device)
        self.epochs = epochs
        self.lr = lr
        self.log = log
        self.r=r

    def __set_masks__(self, x, edge_index, init="normal"):

        N = x.size(0)
        E = edge_index.size(1)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.edge_mask = None

    def __loss__(self, log_logits, pred_label):

        loss = -log_logits[0, pred_label]
        m = self.edge_mask.sigmoid()
        # eg对于平均node=30的MUTAG m几乎都是0.5了 意义何在
        #或者说 先randn 再sigmod的意义在：
        # ①初始化集中在0.5左右 大的往上 小的往下 可能理想的结果是最终变成1 0形式
        # ②sigmod后处于中间状态（靠近0.5）mask的个体移动更快
        loss = loss + self.r*self.coeffs['edge_size'] * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)  # 限制软掩码
        loss = loss + self.r*self.coeffs['edge_ent'] * ent.mean()
        return loss

    def __loss2__(self, log_logits, pred_label,epoch):

        loss = -log_logits[0, pred_label]
        m = self.edge_mask.sigmoid()
        #eg对于平均node=30的MUTAG m几乎都是0.5了 意义何在
        #或者说 先randn 再sigmod的意义在：
        # ①初始化集中在0.5左右 大的往上 小的往下 可能理想的结果是最终变成1 0形式
        # ②sigmod后处于中间状态（靠近0.5）mask的个体移动更快 TODO
        # loss = loss + self.coeffs['edge_size'] * m.sum()  # fjf mask的L1  #替代
        if epoch>170:
            loss = loss + 1.1*self.coeffs['edge_size'] * m.sum()
        elif epoch>140:
            loss = loss + 1.0*self.coeffs['edge_size'] * m.sum()
        elif epoch>90:
            loss = loss + 0.9*self.coeffs['edge_size'] * m.sum()
        elif epoch>60:
            loss = loss + 0.8*self.coeffs['edge_size'] * m.sum()
        elif epoch>30:
            loss = loss + 0.7*self.coeffs['edge_size'] * m.sum()
        else:
            loss = loss + 0.6*self.coeffs['edge_size'] * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)  #限制软掩码
        loss = loss + self.coeffs['edge_ent'] * ent.mean()
        return loss

    def __loss_delate_L1__(self, log_logits, pred_label):
        loss = -log_logits[0, pred_label]
        m = self.edge_mask.sigmoid()
        #eg对于平均node=30的MUTAG m几乎都是0.5了 意义何在
        #或者说 先randn 再sigmod的意义在：
        # ①初始化集中在0.5左右 大的往上 小的往下 可能理想的结果是最终变成1 0形式
        # ②sigmod后处于中间状态（靠近0.5）mask的个体移动更快 TODO
        # loss = loss + self.coeffs['edge_size'] * m.sum()  #fjf mask的L1
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)  #fjf 限制软掩码
        loss = loss + self.coeffs['edge_ent'] * ent.mean()
        return loss

    def explain_graph(self, graph, **kwargs):

        self.__clear_masks__()
        num_edges = graph.edge_index.size(1)

        # Get the initial prediction.
        with torch.no_grad():
            log_logits = self.model(x=graph.x,
                                    edge_index=graph.edge_index,
                                    edge_attr=graph.edge_attr,
                                    batch=graph.batch)
            pred_label = log_logits.argmax(dim=-1)

        self.__set_masks__(graph.x, graph.edge_index)
        self.to(graph.x.device)

        optimizer = torch.optim.Adam([self.edge_mask], lr=self.lr)

        for epoch in range(1, self.epochs + 1):

            optimizer.zero_grad()
            log_logits = self.model(x=graph.x,
                                    edge_index=graph.edge_index,
                                    edge_attr=graph.edge_attr,
                                    batch=graph.batch)
            # loss = self.__loss_delate_L1__(log_logits, pred_label)  #替代测试
            loss = self.__loss__(log_logits, pred_label)

            loss.backward()
            optimizer.step()

        edge_mask = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()

        return edge_mask

    def __repr__(self):
        return f'{self.__class__.__name__}()'
