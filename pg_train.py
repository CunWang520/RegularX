from tqdm import tqdm

from torch_geometric.data import DataLoader

from explain_module.utils import *
from explain_module.utils.parser import *
from explain_module.gnn_model_zoo.tox21_gnn import Tox21Net
from explain_module.gnn_model_zoo.ba3motif_gnn import BA3MotifNet
from explain_module.gnn_model_zoo.mutag_gnn import MutagNet
from explain_module.gnn_model_zoo.reddit5k_gnn import Reddit5kNet
from explain_module.gnn_model_zoo.vg_gnn import VGNet
from explain_module.data_loader_zoo.tox21_dataloader import Tox21
from explain_module.data_loader_zoo.mutag_dataloader import Mutagenicity
from explain_module.data_loader_zoo.reddit5k_dataloader import Reddit5k
from explain_module.data_loader_zoo.ba3motif_dataloader import BA3Motif
from explain_module.data_loader_zoo.vg_dataloader import Visual_Genome

from explain_module.attribution_model_zoo.pg_explainer import PGExplainer
import os.path as osp
import warnings
import time
import argparse

warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datasets = ["mutag", "reddit5k", "vg", "ba3","tox21"]
datasets_y_num=[2,5,2,3,2]


def parse_args():
    t = time.strftime('%b %d %Y %H:%M:%S', time.localtime(time.time()))
    parser = argparse.ArgumentParser(description="Train the PGExplainer.")
    parser.add_argument('--dataset_num', type=int, default=3,
                        help='Choose dataset(s).')
    parser.add_argument('--ACC_ratio_lst', type=list, default=[0.1],
                        help='.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of loops to train the mask.')

    return parser.parse_args()


def print_to_file(path, fileName=None):
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    if not fileName:
        fileName = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
    sys.stdout = Logger(fileName + '.log', path=path)

    print(fileName.center(50, '*'))


if __name__ == "__main__":

    temp = 0.1
    ro = 1
    set_seed(2021)
    args = parse_args()
    dataset_name = datasets[args.dataset_num]
    path = 'trained/gnns/%s_net.pt' % dataset_name
    # print_to_file('', fileName='log/%s_pg_tem0d1_r0d%d' % (dataset_name, ro))
    if dataset_name == "mutag":
        folder = 'data/MUTAG'
        train_dataset = Mutagenicity(folder, mode='training')
        test_dataset = Mutagenicity(folder, mode='testing')
    elif dataset_name == "vg":
        folder = 'data/VG'
        train_dataset = Visual_Genome(folder, mode='training')
        test_dataset = Visual_Genome(folder, mode='testing')
    elif dataset_name == "reddit5k":
        folder = 'data/Reddit5k'
        train_dataset = Reddit5k(folder, mode='training')
        test_dataset = Reddit5k(folder, mode='testing')
    elif dataset_name == "ba3":
        folder = 'data/BA3'
        train_dataset = BA3Motif(folder, mode='training')
        test_dataset = BA3Motif(folder, mode='testing')
    elif dataset_name == "tox21":
        folder = 'data/Tox21'
        train_dataset = Tox21(folder, mode='training')
        test_dataset = Tox21(folder, mode='testing')
    else:
        raise ValueError

    dataset_mask = []
    for label, dataset in zip(['train', 'test'], [train_dataset, test_dataset]):

        flitered_path = folder + "/filtered_idx_%s.pt" % label
        if osp.exists(flitered_path):
            graph_mask = torch.load(flitered_path)
        else:
            loader = DataLoader(dataset,
                                batch_size=1,
                                shuffle=False
                                )
            # filter graphs with right prediction
            model = torch.load(path).to(device)
            graph_mask = torch.zeros(len(loader.dataset), dtype=torch.bool)
            idx = 0
            for g in tqdm(iter(loader), total=len(loader)):

                g.to(device)
                model(g.x, g.edge_index, g.edge_attr, g.batch)
                if g.y == model.readout.argmax(dim=1):
                    graph_mask[idx] = True
                idx += 1

            torch.save(graph_mask, flitered_path)
            dataset_mask.append(graph_mask)

        print("number of graphs(%s): %4d" % (label, graph_mask.nonzero().size(0)))
        exec("%s_loader = DataLoader(dataset[graph_mask], batch_size=64, shuffle=False, drop_last=False)" % label)

    explainer = PGExplainer(path,
                            n_in_channels=torch.flatten(dataset[0].x, 1, -1).size(1),
                            e_in_channels=dataset[0].edge_attr.size(1),
                            n_label=datasets_y_num[args.dataset_num])
    parameters = list()
    for k, edge_mask in enumerate(explainer.edge_mask):
        edge_mask.train()

        parameters += list(explainer.edge_mask[k].parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-3)

    adjust = True
    top_ratio = 0.1 * ro
    if adjust:
        ratio = 1.0
    else:
        ratio = top_ratio
    cnt = 0
    loss_all = 0
    last_acc = 1.0
    top_ratio_list = args.ACC_ratio_lst
    sum_imp_all_list=[]
    for epoch in range(args.epoch):
        sum_imp_all=0
        _r = []
        r = []
        G1_acc_loger = torch.FloatTensor([])
        G_acc_loger = torch.FloatTensor([])
        for g in train_loader:
            g.to(device)
            optimizer.zero_grad()
            loss, G_acc, G1_acc, pos_r, neg_r,sum_imp = explainer.explain_graph(g,
                                                                        ratio=ratio,
                                                                        train_mode=True,
                                                                        temp=temp)
            sum_imp_all+=sum_imp

            loss.backward()
            optimizer.step()

            loss_all += loss
            _r.append(pos_r)
            r.append(neg_r)
            G1_acc_loger = torch.cat([G1_acc_loger, G1_acc])
            G_acc_loger = torch.cat([G_acc_loger, G_acc])
        print(sum_imp_all)
        sum_imp_all_list.append(sum_imp_all)
        test_G1_acc_loger = torch.FloatTensor([])
        test_G_acc_loger = torch.FloatTensor([])
        for g in test_loader:
            g.to(device)
            loss, G_acc, G1_acc, pos_r, neg_r,none = explainer.explain_graph(g, train_mode=True, ratio=ratio)
            test_G1_acc_loger = torch.cat([test_G1_acc_loger, G1_acc])
            test_G_acc_loger = torch.cat([test_G_acc_loger, G_acc])

        train_loss = loss_all / len(train_loader.dataset)
        print("Epoch: %d, Train Loss: %.3f, R1: %.3f, ACC: %.3f(Train) %.3f(Test)" % (
            epoch + 1,  train_loss,np.mean(_r),G1_acc_loger.mean(axis=0), test_G1_acc_loger.mean(axis=0)))

        if adjust:
            if abs(ratio - top_ratio) < 0.005:
                cnt += 1
            if cnt ==5:
                break
            cur = test_G_acc_loger.mean()
            if cur/last_acc > 0.98:
                ratio = max(top_ratio, 0.7 * ratio)
            else:
                ratio = min(1., 1.25 * ratio)
            last_acc = cur
    torch.save(explainer, 'trained/pg/%s.pt' % dataset_name)
print('ACC std: ', np.array(test_G1_acc_loger).std(axis=0))
ave_acc = np.array(test_G1_acc_loger).mean(axis=1)
print('ACC-ROC: %.4f   std: %4f' % (test_G1_acc_loger.mean(), test_G1_acc_loger.std()))

print(sum_imp_all_list)