from tqdm import tqdm
import scipy.stats
import os.path as osp
import copy

import torch_geometric.transforms as T

from explain_module.utils import *
from explain_module.utils.parser import *
from explain_module.gnn_model_zoo.tox21_gnn import Tox21Net
from explain_module.gnn_model_zoo.ba3motif_gnn import BA3MotifNet
from explain_module.gnn_model_zoo.ba2motif_gnn import BA2MotifNet
from explain_module.gnn_model_zoo.mutag_gnn import MutagNet
from explain_module.gnn_model_zoo.reddit5k_gnn import Reddit5kNet
from explain_module.gnn_model_zoo.vg_gnn import VGNet
from explain_module.gnn_model_zoo.mnist_gnn import MNISTNet
from explain_module.data_loader_zoo.tox21_dataloader import Tox21
from explain_module.data_loader_zoo.mutag_dataloader import Mutagenicity
from explain_module.data_loader_zoo.reddit5k_dataloader import Reddit5k
from explain_module.data_loader_zoo.ba3motif_dataloader import BA3Motif
from explain_module.data_loader_zoo.ba2motif_dataloader import BA2Motif
from explain_module.data_loader_zoo.vg_dataloader import Visual_Genome
# from explain_module.data_loader_zoo.mnist_dataloader import Mnist
from torch_geometric.datasets import MNISTSuperpixels

# from explain_module.attribution_model_zoo.random_caster import RandomCaster
from explain_module.attribution_model_zoo.cxplainer import CXplainer
from explain_module.attribution_model_zoo.sa_explainer import SAExplainer
from explain_module.attribution_model_zoo.ig_explainer import IGExplainer
from explain_module.attribution_model_zoo.deeplift import DeepLIFTExplainer
from explain_module.attribution_model_zoo.gradcam import GradCam
from explain_module.attribution_model_zoo.gnnexplainer import GNNExplainer
from explain_module.attribution_model_zoo.pgm_explainer import PGMExplainer
# from explain_module.attribution_model_zoo.screener import Screener
from explain_module.attribution_model_zoo.pg_explainer import PGExplainer
# from explain_module.attribution_model_zoo.cge_gnn import CGE_GNNExplainer
import warnings
from torch_geometric.data import DataLoader

warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datasets = ["mutag", "reddit5k", "vg", "ba3","tox21",'mnist','ba2']
Explainers = {
    # 0: RandomCaster,
    1: SAExplainer,
    2: IGExplainer,
    3: DeepLIFTExplainer,
    4: GradCam,
    5: GNNExplainer,
    6: CXplainer,
    7: PGExplainer,
    8: PGMExplainer,
    # 9: Screener,
    # # 10:LT_GNNExplainer,
    # 11:CGE_GNNExplainer
}

if __name__ == "__main__":

    args = parse_args()

    set_seed(args.random_seed)
    dataset_num = eval(args.dataset_num)
    idx_explainers = eval(args.idx_explainers)
    top_ratio_list = args.ACC_ratio_lst  #[0.1 * (i+1) for i in range(10)]
    for idx_dataset, num in enumerate(dataset_num):
        dataset_name = datasets[num]
        if len(args.log_name):
            print_to_file("log/%s" % dataset_name, fileName="%s" % args.log_name)
        path = 'trained/gnns/%s_net.pt' % dataset_name
        explainers = []
        for i in idx_explainers:
            # PGExplainer
            if i == 7:
                pg = torch.load("trained/pg/%s.pt" % dataset_name)
                explainers.append(pg)
            else:
                explainers.append(Explainers[i](path))

        # args_print(args)

        if dataset_name == "mutag":
            folder = 'data/MUTAG'
            dataset_train = Mutagenicity(folder, mode='training')
            dataset_test = Mutagenicity(folder, mode='testing')
        elif dataset_name == "vg":
            folder = 'data/VG'
            dataset_train = Visual_Genome(folder, mode='training')
            dataset_test = Visual_Genome(folder, mode='testing')
        elif dataset_name == "reddit5k":
            folder = 'data/Reddit5k'
            dataset_train = Reddit5k(folder, mode='training')
            dataset_test = Reddit5k(folder, mode='testing')
        elif dataset_name == "ba3":
            folder = 'data/BA3'
            dataset_train = BA3Motif(folder, mode='training')
            dataset_test = BA3Motif(folder, mode='testing')
        elif dataset_name == "ba2":
            folder = 'data/BA2'
            dataset_train = BA2Motif(folder, mode='training')
            dataset_test = BA2Motif(folder, mode='testing')
        elif dataset_name == "tox21":
            folder = 'data/Tox21'
            dataset_train = Tox21(folder, mode='training')
            dataset_test = Tox21(folder, mode='testing')
        elif dataset_name == "mnist":
            folder = 'data/MNIST'
            transform = T.Cartesian(cat=False, max_value=9)
            dataset_train = MNISTSuperpixels("data/MNIST", True, transform=transform)
            dataset_test = MNISTSuperpixels("data/MNIST", False, transform=transform)
            # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            # test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        else:
            raise ValueError
        dataset_mask = []
        flitered_path = folder + "/filtered_idx_test.pt"
        if osp.exists(flitered_path):
            graph_mask = torch.load(flitered_path)
        else:
            loader = DataLoader(dataset_test,
                                batch_size=1,
                                shuffle=False
                                )
            model = torch.load(path).to(device)
            graph_mask = torch.zeros(len(loader.dataset), dtype=torch.bool)
            idx = 0
            for g in tqdm(iter(loader), total=len(loader)):

                g.to(device)
                if dataset_name == "mnist":
                    model(g)
                else:
                    model(g.x, g.edge_index, g.edge_attr, g.batch)
                if g.y == model.readout.argmax(dim=1):
                    graph_mask[idx] = True
                idx += 1

            torch.save(graph_mask, flitered_path)
            dataset_mask.append(graph_mask)

        n_filtered = graph_mask.nonzero().size(0)
        print("number of graphs(Testing): %4d" % n_filtered)
        _testdataset = dataset_test[graph_mask].shuffle()
        loader = DataLoader(_testdataset[:min(n_filtered, args.num_test)], batch_size=1, shuffle=False, drop_last=False)



        ICEs_all = []
        n_exps = len(explainers)
        n_ratios = len(top_ratio_list)
        for i, explainer in enumerate(explainers):
            print(explainer.name,dataset_name,'----------------------')
            acc_loger = []
            # sc_loger = []
            # contra_loger = []
            # precision_loger18 = []
            # precision_loger14=[]
            # precision_loger10 = []
            # precision_loger6 = []
            # recall_loger18 = []
            # recall_loger14 = []
            # recall_loger10 = []
            # recall_loger6 = []


            t1 = time.time()

            acc_r=[]
            for r in [1 * i for i in range(0,25)]:
                print(dataset_name, 'under regularization =   ( 0.05  *', r,')')
                print('acc under sparsity (0.9,0.8,...,0):')
                for g in tqdm(iter(loader), total=len(loader)):
                    g.to(device)
                    if explainer.name == 'CGE_GNNExplainer':
                        imp1 = explainer.explain_graph(g, draw_graph=args.draw_graph, dataset=dataset_name,  # key
                                                       vis_ratio=args.vis_ratio)
                    elif explainer.name == 'GNNExplainer':
                        imp1 = explainer.explain_graph(r,g, draw_graph=args.draw_graph,  # key
                                                       vis_ratio=args.vis_ratio)

                    else:
                        imp1 = explainer.explain_graph(g, draw_graph=args.draw_graph,  # key
                                                       vis_ratio=args.vis_ratio)
                    # if args.eval_SC:
                    #     tmp = scipy.stats.spearmanr(imp1, imp2)[0]
                    #     if np.isnan(tmp):
                    #         tmp = 1
                    #     sc_loger.append(abs(tmp))
                    if args.eval_ACC:
                        acc_loger.append(explainer.evaluate_acc(top_ratio_list))
                    # if args.eval_CTS:
                    #     contra_loger.append(explainer.evaluate_contrastivity())
                    # if 'ba' in dataset_name:
                    #     precision_loger18.append(explainer.evaluate_precision(topk=18))
                    #     recall_loger18.append(explainer.evaluate_recall(topk=18))
                    #     precision_loger14.append(explainer.evaluate_precision(topk=14))
                    #     recall_loger14.append(explainer.evaluate_recall(topk=14))
                    #     precision_loger10.append(explainer.evaluate_precision(topk=args.topk))
                    #     recall_loger10.append(explainer.evaluate_recall(topk=args.topk))
                    #     precision_loger6.append(explainer.evaluate_precision(topk=6))
                    #     recall_loger6.append(explainer.evaluate_recall(topk=6))
                #     if args.eval_net_onecut:   #fjf
                #         para_dict_old = copy.deepcopy(explainer.model.state_dict())
                #         for i in range(10,0,-1):
                #             para_dict = {}
                #             para_size_dict = {}
                #             para_indices_dict = {}
                #             for key in explainer.model.state_dict():
                #                 para_size_dict[key] = explainer.model.state_dict()[key].size()
                #                 para_dict[key] =explainer.model.state_dict()[key].view(-1)
                #                 sorted, indices = torch.sort(torch.abs(para_dict[key]))
                #                 para_indices_dict[key] = indices
                #             eval('one_cut%d.append(explainer.evaluate_acc_net(para_dict,para_size_dict,para_indices_dict,0.1*i))'%i)
                #             explainer.model.load_state_dict(para_dict_old)
                t2 = time.time()

                # if len(ICEs_all):
                #     print('output ICE boxplot')
                #     ICEs_all = np.array(ICEs_all)
                #     colors = ['pink', 'lightblue', 'lightgreen', "orchid"]
                #     labels = ['Mutagenicity', 'REDDIT-MULTI-5K', 'Visual Genome', 'BA-3']
                #
                #     plt.figure(figsize=(5.2, 7))
                #     fig = plt.gcf()
                #     bplot = plt.boxplot(ICEs_all, patch_artist=True, widths=0.8)
                #     for patch, color in zip(bplot['boxes'], colors):
                #         patch.set_facecolor(colors[num])
                #
                #     plt.xlabel(labels[num], fontsize=15, weight='bold')
                #     ax = plt.gca()
                #     plt.yticks(fontsize=13, weight='bold')
                #     plt.xticks([])
                #     plt.savefig("picture/boxplot_%s.png" % dataset_name, dpi=500)
                #     plt.show()

                # print('total : [%.2fs]' % (t2 - t1))
                # print('per graph: [%.3fs]' % ((t2 - t1) / len(loader)))
                # if 'ba' in dataset_name:
                #     if dataset_name=='ba2':
                #         print('Precision@ 18:   ', np.array(precision_loger18).mean(axis=0))
                #         # print('Recall@ 18:   ', recall_loger18)
                #         print('Precision@ 14:   ', np.array(precision_loger14).mean(axis=0))
                #         # print('Recall@ 14:   ', recall_loger14)
                #         print('Precision@', args.topk, ': ', np.array(precision_loger10).mean(axis=0))
                #         # print('Recall@', args.topk, ': ', recall_loger10)
                #         print('Precision@ 6:   ', np.array(precision_loger6).mean(axis=0))
                #         # print('Recall@ 6:   ', recall_loger6)
                #     else:
                #         print('Precision@ 18:   ', np.array(precision_loger18).mean(axis=0))
                #         print('Recall@ 18:   ', np.array(recall_loger18).mean(axis=0))
                #         print('Precision@ 14:   ', np.array(precision_loger14).mean(axis=0))
                #         print('Recall@ 14:   ', np.array(recall_loger14).mean(axis=0))
                #         print('Precision@', args.topk, ': ', np.array(precision_loger10).mean(axis=0))
                #         print('Recall@', args.topk, ': ', np.array(recall_loger10).mean(axis=0))
                #         print('Precision@ 6:   ', np.array(precision_loger6).mean(axis=0))
                #         print('Recall@ 6:   ', np.array(recall_loger6).mean(axis=0))
                if args.eval_ACC:
                    print('ACC: ', np.array(acc_loger).mean(axis=0))  # fjf 每statio取平均
                    acc_r.append(np.array(acc_loger).mean(axis=0)[0])
                    # print('ACC std: ', np.array(acc_loger).std(axis=0))  #fjf 每statio取平均
                    ave_acc = np.array(acc_loger).mean(axis=1)
                    print('ACC-ROC: %.4f   std: %4f' % (ave_acc.mean(), ave_acc.std()))  # fjf 每个个体不同statio取平均
                # if args.eval_SC:
                #     print('SC: ', np.array(sc_loger).mean())
                # if args.eval_CTS:
                #     print('CTS: ', np.array(contra_loger).mean())
                # if args.eval_net_onecut:
                #     for i in range(10):
                #         print('one_cut_ACC_graph%f:  '%((i+1)/10), eval('np.array(one_cut%d).mean(axis=0)'%(i+1)))

            print('==========', dataset_name, '========')
            print(acc_r)