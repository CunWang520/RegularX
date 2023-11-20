import os
import copy
import math
import numpy as np
import networkx as nx
from pathlib import Path
import matplotlib.cm as cm
import sys

import torch
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os.path as osp

from PIL import Image
from visual_genome import local as vgl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-6
ifd_pert = 0.1
n_class_dict = {'MutagNet': 2, 'Tox21Net': 2, 'Reddit5kNet': 5, 'VGNet': 5, 'BA2MotifNet': 2, 'BA3MotifNet': 3}
vis_dict = {
    'MutagNet': {'node_size': 400, 'linewidths': 1, 'font_size': 10, 'width': 3},
    'Tox21Net': {'node_size': 400, 'linewidths': 1, 'font_size': 10, 'width': 3},
    'BA3MotifNet': {'node_size': 300, 'linewidths': 1, 'font_size': 10, 'width': 3},
    'defult': {'node_size': 200, 'linewidths': 1, 'font_size': 10, 'width': 2}
}
chem_graph_label_dict = {'MutagNet': {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S',
                                      8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'},
                         'Tox21Net': {0: 'O', 1: 'C', 2: 'N', 3: 'F', 4: 'Cl', 5: 'S', 6: 'Br', 7: 'Si',
                                      8: 'Na', 9: 'I', 10: 'Hg', 11: 'B', 12: 'K', 13: 'P', 14: 'Au',
                                      15: 'Cr', 16: 'Sn', 17: 'Ca', 18: 'Cd', 19: 'Zn', 20: 'V', 21: 'As',
                                      22: 'Li', 23: 'Cu', 24: 'Co', 25: 'Ag', 26: 'Se', 27: 'Pt', 28: 'Al',
                                      29: 'Bi', 30: 'Sb', 31: 'Ba', 32: 'Fe', 33: 'H', 34: 'Ti', 35: 'Tl',
                                      36: 'Sr', 37: 'In', 38: 'Dy', 39: 'Ni', 40: 'Be', 41: 'Mg', 42: 'Nd',
                                      43: 'Pd', 44: 'Mn', 45: 'Zr', 46: 'Pb', 47: 'Yb', 48: 'Mo', 49: 'Ge',
                                      50: 'Ru', 51: 'Eu', 52: 'Sc'}
                         }
rec_color = ['cyan', 'mediumblue', 'deeppink', 'darkorange', 'gold', 'chartreuse', 'lightcoral', 'darkviolet', 'teal',
             'lightgrey', ]


class Explainer(object):

    def __init__(self, gnn_model_path):
        self.model = torch.load(gnn_model_path).to(device)
        self.model.eval()
        self.model_name = self.model.__class__.__name__
        self.name = self.__class__.__name__

        self.path = gnn_model_path
        self.last_result = None
        self.vis_dict = None

    def explain_graph(self, graph, **kwargs):
        """
        Main part for different graph attribution methods
        :param graph: target graph instance to be explained
        :param kwargs:
        :return: edge_imp, i.e., attributions for edges, which are derived from the attribution methods.
        """
        raise NotImplementedError

    def get_cxplain_scores(self, graph):
        # initialize the ranking list with cxplain.
        y = graph.y
        orig_pred = self.model(graph.x,
                               graph.edge_index,
                               graph.edge_attr,
                               graph.batch)[0, y]

        scores = []
        for e_id in range(graph.num_edges):
            edge_mask = torch.ones(graph.num_edges, dtype=torch.bool)
            edge_mask[e_id] = False
            masked_edge_index = graph.edge_index[:, edge_mask]
            masked_edge_attr = graph.edge_attr[edge_mask]

            masked_pred = self.model(graph.x,
                                     masked_edge_index,
                                     masked_edge_attr,
                                     graph.batch)[0, y]

            scores.append(orig_pred - masked_pred)
            # scores.append(orig_pred - masked_pred)
        scores = torch.tensor(scores)
        return scores.cpu().detach().numpy()

    @staticmethod
    def get_rank(lst, r=1):

        topk_idx = list(np.argsort(-lst))
        top_pred = np.zeros_like(lst)
        n = len(lst)
        k = int(r * n)
        for i in range(k):
            top_pred[topk_idx[i]] = n - i
        return top_pred

    @staticmethod
    def norm_imp(imp):
        # _min = np.min(imp)
        # _max = np.max(imp) + 1e-16
        # imp = (imp - _min)/(_max - _min)
        # return imp
        imp[imp < 0] = 0
        imp += 1e-16
        return imp / imp.sum()

    def __relabel__(self, g, edge_index):

        sub_nodes = torch.unique(edge_index)
        x = g.x[sub_nodes]
        batch = g.batch[sub_nodes]
        row, col = edge_index

        # remapping the nodes in the explanatory subgraph to new ids.
        node_idx = row.new_full((g.num_nodes,), -1)
        node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
        edge_index = node_idx[edge_index]
        return x, edge_index, batch

    def pack_explanatory_subgraph(self, top_ratio=0.2):

        graph, imp = self.last_result
        top_idx = torch.LongTensor([])
        graph_map = graph.batch[graph.edge_index[0, :]]
        exp_subgraph = copy.deepcopy(graph)
        exp_subgraph.y = graph.y

        for i in range(graph.num_graphs):
            edge_indicator = torch.where(graph_map == i)[0].detach().cpu()
            Gi_n_edge = len(edge_indicator)
            topk = max(math.floor(top_ratio * Gi_n_edge), 1)

            Gi_pos_edge_idx = np.argsort(-imp[edge_indicator])[:topk]
            top_idx = torch.cat([top_idx, edge_indicator[Gi_pos_edge_idx]])
        # retrieval properties of the explanatory subgraph
        # .... the edge_attr.
        exp_subgraph.edge_attr = graph.edge_attr[top_idx]
        # .... the edge_index.
        exp_subgraph.edge_index = graph.edge_index[:, top_idx]
        # .... the nodes.
        # exp_subgraph.x = graph.x
        exp_subgraph.x, exp_subgraph.edge_index, exp_subgraph.batch = \
            self.__relabel__(exp_subgraph, exp_subgraph.edge_index)

        return exp_subgraph

    def evaluate_recall(self, topk=10):
        graph, imp = self.last_result
        E = graph.num_edges
        if isinstance(graph.ground_truth_mask, list):
            graph.ground_truth_mask = graph.ground_truth_mask[0]
        index = np.argsort(-imp)[:topk]
        values = graph.ground_truth_mask[index]
        # ).detach().cpu().numpy()
        return float(
            values.sum()) / graph.ground_truth_mask.sum()  # (values.sum().float() / graph.ground_truth_mask.sum()).detach().cpu().numpy()

    def evaluate_precision(self, topk=10):
        graph, imp = self.last_result
        E = graph.num_edges
        if isinstance(graph.ground_truth_mask, list):
            graph.ground_truth_mask = graph.ground_truth_mask[0]
        index = np.argsort(-imp)[:topk]
        values = graph.ground_truth_mask[index]
        return float(values.sum()) / topk  # ((values.sum().float()) / topk).detach().cpu().numpy()

    def evaluate_acc(self, top_ratio_list):
        assert self.last_result is not None
        acc = np.array([[]])
        for idx, top_ratio in enumerate(top_ratio_list):
            if abs(top_ratio - 1.0) < EPS:
                acc = np.concatenate([acc, [[1.0]]], axis=1)
                continue

            exp_subgraph = self.pack_explanatory_subgraph(top_ratio)
            self.model(exp_subgraph.x,
                       exp_subgraph.edge_index,
                       exp_subgraph.edge_attr,
                       exp_subgraph.batch
                       )
            res = (exp_subgraph.y == self.model.readout.argmax(dim=1)).detach().cpu().float().view(-1, 1).numpy()
            acc = np.concatenate([acc, res], axis=1)
        return acc

    def evaluate_acc_net(self, dict_flat, size_dict, indices_dict, top_graph_ratio, top_net_ratio_list=[0.1 * i for i in range(10, -1, -1)]):

        assert self.last_result is not None
        acc = np.array([[]])
        if abs(top_graph_ratio-1.0)<EPS:
            graph, imp = self.last_result
            exp_subgraph=graph
        else:
            exp_subgraph = self.pack_explanatory_subgraph(top_graph_ratio)
        for idx, top_ratio_net in enumerate(top_net_ratio_list):
            para_new_dict = {}
            for key in dict_flat:
                para_new = dict_flat[key]
                if ('weight' in key) and ('batch_norms' not in key):
                    k=int((1.-top_ratio_net)*(para_new.size()[0]))
                    para_new[indices_dict[key][0:k]] = 0.
                para_new_dict[key]=para_new.view(size_dict[key])
            self.model.load_state_dict(para_new_dict)
            self.model(exp_subgraph.x,
                       exp_subgraph.edge_index,
                       exp_subgraph.edge_attr,
                       exp_subgraph.batch
                       )
            res = (exp_subgraph.y == self.model.readout.argmax(dim=1)).detach().cpu().float().view(-1, 1).numpy()
            acc = np.concatenate([acc, res], axis=1)
        # print(acc)
        return acc

    def evaluate_contrastivity(self):

        assert self.last_result is not None
        # for ba3, we extract the top 10% edge importance distribution for evaluation
        graph, imp = self.last_result
        idx = np.argsort(-imp)[int(0.1 * graph.num_edges):]
        _imp = copy.copy(imp)
        _imp[idx] = 0
        counter_graph = copy.deepcopy(graph)
        counter_classes = [i for i in range(n_class_dict[self.model_name])]
        counter_classes.pop(graph.y)
        counter_accumulate = 0
        for c in counter_classes:
            counter_graph.y = torch.LongTensor([c]).cuda()
            if self.name == "Screener" and \
                    isinstance(graph.name[0], str) and \
                    "reddit" in graph.name[0]:
                counter_imp, _ = self.explain_graph(counter_graph, large_scale=True)
            elif self.name == "Screener":
                counter_imp, _ = self.explain_graph(counter_graph)
            else:
                counter_imp = self.explain_graph(counter_graph)
            counter_imp = self.norm_imp(counter_imp)
            idx = np.argsort(-counter_imp)[int(0.1 * graph.num_edges):]
            counter_imp[idx] = 0
            # counter_imp += 1e-8
            # counter_imp = counter_imp / counter_imp.sum()
            tmp = scipy.stats.spearmanr(counter_imp, _imp)[0]

            if np.isnan(tmp):
                tmp = 1
            counter_accumulate += abs(tmp)
        self.last_result = graph, imp  # may be unnecessary

        return counter_accumulate / len(counter_classes)

    def evaluate_infidelity(self, N=5, p0=0.25):
        assert self.last_result is not None
        graph, imp = self.last_result

        imp = torch.FloatTensor(imp + 1e-8).cuda()
        imp = imp / imp.sum()
        ps = p0 * torch.ones_like(imp)

        self.model(graph.x,
                   graph.edge_index,
                   graph.edge_attr,
                   graph.batch
                   )
        ori_pred = self.model.readout[0, graph.y]
        lst = []
        for _ in range(N):
            p0 = torch.bernoulli(ps)
            edge_mask = (1.0 - p0).bool()
            self.model(graph.x,
                       graph.edge_index[:, edge_mask],
                       graph.edge_attr[edge_mask],
                       graph.batch
                       )
            pert_pred = self.model.readout[0, graph.y]
            infd = pow(sum(p0 * imp) - (ori_pred - pert_pred), 2).cpu().detach().numpy()
            lst.append(infd)
        lst = np.array(lst)
        return lst.mean()

    def visualize(self, graph, edge_imp, method, vis_ratio=0.2):
        topk = max(int(vis_ratio * graph.num_edges), 1)  # 10
        if self.model_name == "VGNet":
            topk = 3
            idx = np.argsort(-edge_imp)[:topk]
            top_edges = graph.edge_index[:, idx]
            all = graph.edge_index
            # nodes_idx = torch.unique(top_edges)

            scene_graph = vgl.get_scene_graph(image_id=int(graph.name),
                                              images='visual_genome/raw',
                                              image_data_dir='visual_genome/raw/by-id/',
                                              synset_file='visual_genome/raw/synsets.json')
            # top_edges = np.array([[26, 0, 8],
            #                      [7, 11,0]],dtype=np.int)
            # you can also use api to get the scence graph if the network is stable,
            # in this case, all the .json data is unneccessary
            # scene_graph = api.get_scene_graph_of_image(id=int(graph.id))
            # print(scene_graph.relationships)
            # for idx, e in enumerate(all.T):
            #    print(idx, '  ', scene_graph.objects[e[0]], '---', scene_graph.objects[e[Reddit5k]])
            # print(idx)
            r = 0.95  # transparency
            import os.path as osp
            img = Image.open("data/VG/raw/%d-%d.jpg" % (graph.name, graph.y))
            data = list(img.getdata())
            ndata = list(
                [(int((255 - p[0]) * r + p[0]), int((255 - p[1]) * r + p[1]), int((255 - p[2]) * r + p[2])) for p in
                 data])
            mode = img.mode
            width, height = img.size
            edges = list(top_edges.T)
            for i, (u, v) in enumerate(edges[::-1]):
                r = 1.0 - 1.0 / len(edges) * (i + 1)
                # r=0
                obj1 = scene_graph.objects[u]
                obj2 = scene_graph.objects[v]
                for obj in [obj1, obj2]:
                    for x in range(obj.x, obj.width + obj.x):
                        for y in range(obj.y, obj.y + obj.height):
                            ndata[y * width + x] = (int((255 - data[y * width + x][0]) * r + data[y * width + x][0]),
                                                    int((255 - data[y * width + x][1]) * r + data[y * width + x][1]),
                                                    int((255 - data[y * width + x][2]) * r + data[y * width + x][2]))

            img = Image.new(mode, (width, height))
            img.putdata(ndata)

            plt.imshow(img)
            ax = plt.gca()
            for i, (u, v) in enumerate(edges):
                obj1 = scene_graph.objects[u]
                obj2 = scene_graph.objects[v]
                ax.annotate("", xy=(obj2.x, obj2.y), xytext=(obj1.x, obj1.y),
                            arrowprops=dict(width=topk - i, color='wheat', headwidth=5))
                for obj in [obj1, obj2]:
                    ax.text(obj.x, obj.y - 8, str(obj), style='italic',
                            fontsize=13,
                            bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 3,
                                  'edgecolor': rec_color[i % len(rec_color)]}
                            )
                    ax.add_patch(Rectangle((obj.x, obj.y),
                                           obj.width,
                                           obj.height,
                                           fill=False,
                                           edgecolor=rec_color[i % len(rec_color)],
                                           linewidth=1.5))
            plt.tick_params(labelbottom='off', labelleft='off')
            plt.axis('off')

        else:
            idx = np.argsort(-edge_imp)[:topk]
            print(idx)
            G = nx.DiGraph()
            G.add_nodes_from(range(graph.num_nodes))
            G.add_edges_from(list(graph.edge_index.cpu().numpy().T))
            if self.vis_dict is None:
                self.vis_dict = vis_dict[self.model_name] if self.model_name in vis_dict.keys() else vis_dict['defult']

            pos = nx.kamada_kawai_layout(G) if graph.pos is None else graph.pos[0]
            edge_pos_mask = np.zeros(graph.num_edges, dtype=np.bool_)
            # edge_pos_mask[np.array(graph.ground_truth_mask[0], dtype=np.bool_)] = True
            edge_pos_mask[idx] = True
            vmax = sum(edge_pos_mask)
            node_pos_mask = np.zeros(graph.num_nodes, dtype=np.bool_)
            node_neg_mask = np.zeros(graph.num_nodes, dtype=np.bool_)
            node_pos_idx = np.unique(graph.edge_index[:, edge_pos_mask].cpu().numpy()).tolist()
            node_neg_idx = list(set([i for i in range(graph.num_nodes)]) - set(node_pos_idx))
            node_pos_mask[node_pos_idx] = True
            node_neg_mask[node_neg_idx] = True
            nx.draw_networkx_nodes(G, pos={i: pos[i] for i in node_pos_idx},
                                   nodelist=node_pos_idx,
                                   node_size=self.vis_dict['node_size'],
                                   node_color=graph.z[0][node_pos_idx],
                                   alpha=1, cmap='winter',
                                   linewidths=self.vis_dict['linewidths'],
                                   edgecolors='red',
                                   vmin=-max(graph.z[0]), vmax=max(graph.z[0])
                                   )
            nx.draw_networkx_nodes(G, pos={i: pos[i] for i in node_neg_idx},
                                   nodelist=node_neg_idx,
                                   node_size=self.vis_dict['node_size'],
                                   node_color=graph.z[0][node_neg_idx],
                                   alpha=0.2, cmap='winter',
                                   linewidths=self.vis_dict['linewidths'],
                                   edgecolors='whitesmoke',
                                   vmin=-max(graph.z[0]), vmax=max(graph.z[0])
                                   )
            if 'BA' in self.model_name:

                nx.draw_networkx_edges(G, pos=pos,
                                       edgelist=list(graph.edge_index.cpu().numpy().T),
                                       edge_color='grey',
                                       width=self.vis_dict['width'],
                                       arrows=False
                                       )
            else:
                labels = graph.z[0]
                label_dict = chem_graph_label_dict[self.model_name]
                nx.draw_networkx_labels(G, pos=pos,
                                        labels={i: label_dict[labels[i]] for i in range(graph.num_nodes)},
                                        font_size=self.vis_dict['font_size'],
                                        font_weight='bold', font_color='k'
                                        )
                nx.draw_networkx_edges(G, pos=pos,
                                       edgelist=list(graph.edge_index.cpu().numpy().T),
                                       edge_color='whitesmoke',
                                       width=self.vis_dict['width'],
                                       arrows=False
                                       )

            nx.draw_networkx_edges(G, pos=pos,
                                   edgelist=list(graph.edge_index[:, edge_pos_mask].cpu().numpy().T),
                                   edge_color=self.get_rank(edge_imp[edge_pos_mask]),
                                   # np.ones(len(edge_imp[edge_pos_mask])),
                                   width=self.vis_dict['width'],
                                   edge_cmap=cm.get_cmap('bwr'),
                                   edge_vmin=-vmax, edge_vmax=vmax,
                                   arrows=False
                                   )
            ax = plt.gca()
            ax.set_facecolor('aliceblue')

        if method in ["Screener", "RandomCaster"]:
            folder = Path(r'image/%s/%s/r-%.2f' % (self.model_name, method, self.ratio))
        else:
            folder = Path(r'image/%s/%s' % (self.model_name, method))
        if not os.path.exists(folder):
            os.makedirs(folder)
        if isinstance(graph.name[0], str):
            plt.savefig(folder / Path(r'%d-%s.png' % (int(graph.y), str(graph.name[0]))), dpi=500,
                        bbox_inches='tight')
        else:
            plt.savefig(folder / Path(r'%d-%d.png' % (graph.y, int(graph.name[0]))), dpi=500, bbox_inches='tight')

        plt.cla()
