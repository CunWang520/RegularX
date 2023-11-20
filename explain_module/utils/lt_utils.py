from math import sqrt
import torch
from torch_geometric.nn import MessagePassing
import copy
import math
import numpy as np
import torch.nn as nn
from abc import ABC
import random
import explain_module.utils
import sys


def pack_explanatory_subgraph(graph, imp, top_ratio):
	top_idx = torch.LongTensor([])
	graph_map = graph.batch[graph.edge_index[0, :]]
	exp_subgraph = copy.deepcopy(graph)
	exp_subgraph.y = graph.y

	for i in range(graph.num_graphs):
		edge_indicator = torch.where(graph_map == i)[0].detach().cpu()
		Gi_n_edge = len(edge_indicator)
		topk = max(math.floor(top_ratio * Gi_n_edge), 1)

		Gi_pos_edge_idx = np.argsort(-imp[edge_indicator].cpu())[:topk]
		top_idx = torch.cat([top_idx, edge_indicator[Gi_pos_edge_idx]])
	# retrieval properties of the explanatory subgraph
	# .... the edge_attr.
	exp_subgraph.edge_attr = graph.edge_attr[top_idx]
	# .... the edge_index.
	exp_subgraph.edge_index = graph.edge_index[:, top_idx]
	# .... the nodes.
	# exp_subgraph.x = graph.x
	exp_subgraph.x, exp_subgraph.edge_index, exp_subgraph.batch = \
		__relabel__(exp_subgraph, exp_subgraph.edge_index)

	return exp_subgraph


def __relabel__(g, edge_index):

	sub_nodes = torch.unique(edge_index)
	x = g.x[sub_nodes]
	batch = g.batch[sub_nodes]
	row, col = edge_index

	# remapping the nodes in the explanatory subgraph to new ids.
	node_idx = row.new_full((g.num_nodes,), -1)
	node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
	edge_index = node_idx[edge_index]
	return x, edge_index, batch


def combine_mask(mask, top_ratio_every_time):
	lt_num = len(mask)
	for i in range(lt_num - 1, 0, -1):
		top_ratio = top_ratio_every_time[i - 1]
		Gi_edge_idx = len(mask[i - 1])
		topk = max(math.floor(top_ratio * Gi_edge_idx), 1)
		Gi_pos_edge_idx = np.argsort(-np.abs(mask[i - 1].cpu()))[:topk]
		diff_every_time = max(mask[i - 1]) - min(mask[i]) + 1e-4
		for k, index in enumerate(np.unique(Gi_pos_edge_idx)):
			mask[i - 1][int(index)] = mask[i][k] + diff_every_time
	return mask[0]





class AddTrainableMask(ABC):
	_tensor_name: str

	def __init__(self):
		pass

	def __call__(self, module, inputs):
		setattr(module, self._tensor_name, self.apply_mask(module))

	def apply_mask(self, module):
		mask_train = getattr(module, self._tensor_name + "_mask_train")
		mask_fixed = getattr(module, self._tensor_name + "_mask_fixed")
		orig_weight = getattr(module, self._tensor_name + "_orig_weight")
		pruned_weight = mask_train * mask_fixed * orig_weight

		return pruned_weight

	@classmethod
	def apply(cls, module, name, mask_train, mask_fixed, *args, **kwargs):
		method = cls(*args, **kwargs)
		method._tensor_name = name
		orig = getattr(module, name)

		module.register_parameter(name + "_mask_train", mask_train.to(dtype=orig.dtype))
		module.register_parameter(name + "_mask_fixed", mask_fixed.to(dtype=orig.dtype))
		module.register_parameter(name + "_orig_weight", orig)
		del module._parameters[name]

		setattr(module, name, method.apply_mask(module))
		module.register_forward_pre_hook(method)

		return method


def add_mask(model, dataset='ba3'):

	if dataset == 'ba3':
		for i in range(2):
			mask1_train = nn.Parameter(torch.ones_like(model.convs[i].lin1.weight))
			mask1_fixed = nn.Parameter(torch.ones_like(model.convs[i].lin1.weight), requires_grad=False)
			AddTrainableMask.apply(model.convs[i].lin1, 'weight', mask1_train, mask1_fixed)
			mask1_train = nn.Parameter(torch.ones_like(model.convs[i].lin2.weight))
			mask1_fixed = nn.Parameter(torch.ones_like(model.convs[i].lin2.weight), requires_grad=False)
			AddTrainableMask.apply(model.convs[i].lin2, 'weight', mask1_train, mask1_fixed)
			mask1_train = nn.Parameter(torch.ones_like(model.convs[i].lin3.weight))
			mask1_fixed = nn.Parameter(torch.ones_like(model.convs[i].lin3.weight), requires_grad=False)
			AddTrainableMask.apply(model.convs[i].lin3, 'weight', mask1_train, mask1_fixed)
		mask1_train = nn.Parameter(torch.ones_like(model.lin1.weight))
		mask1_fixed = nn.Parameter(torch.ones_like(model.lin1.weight), requires_grad=False)
		AddTrainableMask.apply(model.lin1, 'weight', mask1_train, mask1_fixed)
		mask1_train = nn.Parameter(torch.ones_like(model.lin2.weight))
		mask1_fixed = nn.Parameter(torch.ones_like(model.lin2.weight), requires_grad=False)
		AddTrainableMask.apply(model.lin2, 'weight', mask1_train, mask1_fixed)
		mask1_train = nn.Parameter(torch.ones_like(model.node_emb.weight))
		mask1_fixed = nn.Parameter(torch.ones_like(model.node_emb.weight), requires_grad=False)
		AddTrainableMask.apply(model.node_emb, 'weight', mask1_train, mask1_fixed)

	elif dataset == 'ba2':
		for i in range(3):
			mask1_train = nn.Parameter(torch.ones_like(model.convs[i].lin1.weight))
			mask1_fixed = nn.Parameter(torch.ones_like(model.convs[i].lin1.weight), requires_grad=False)
			AddTrainableMask.apply(model.convs[i].lin1, 'weight', mask1_train, mask1_fixed)
			mask1_train = nn.Parameter(torch.ones_like(model.convs[i].lin2.weight))
			mask1_fixed = nn.Parameter(torch.ones_like(model.convs[i].lin2.weight), requires_grad=False)
			AddTrainableMask.apply(model.convs[i].lin2, 'weight', mask1_train, mask1_fixed)
			mask1_train = nn.Parameter(torch.ones_like(model.convs[i].lin3.weight))
			mask1_fixed = nn.Parameter(torch.ones_like(model.convs[i].lin3.weight), requires_grad=False)
			AddTrainableMask.apply(model.convs[i].lin3, 'weight', mask1_train, mask1_fixed)
		mask1_train = nn.Parameter(torch.ones_like(model.lin1.weight))
		mask1_fixed = nn.Parameter(torch.ones_like(model.lin1.weight), requires_grad=False)
		AddTrainableMask.apply(model.lin1, 'weight', mask1_train, mask1_fixed)
		mask1_train = nn.Parameter(torch.ones_like(model.lin2.weight))
		mask1_fixed = nn.Parameter(torch.ones_like(model.lin2.weight), requires_grad=False)
		AddTrainableMask.apply(model.lin2, 'weight', mask1_train, mask1_fixed)
		mask1_train = nn.Parameter(torch.ones_like(model.node_emb.weight))
		mask1_fixed = nn.Parameter(torch.ones_like(model.node_emb.weight), requires_grad=False)
		AddTrainableMask.apply(model.node_emb, 'weight', mask1_train, mask1_fixed)
	elif dataset == 'mutag':
		for i in range(2):
			for j in range(0,3,2):
				mask1_train = nn.Parameter(torch.ones_like(model.convs[i].nn[j].weight))
				mask1_fixed = nn.Parameter(torch.ones_like(model.convs[i].nn[j].weight), requires_grad=False)
				AddTrainableMask.apply(model.convs[i].nn[j], 'weight', mask1_train, mask1_fixed)
		mask1_train = nn.Parameter(torch.ones_like(model.lin1.weight))
		mask1_fixed = nn.Parameter(torch.ones_like(model.lin1.weight), requires_grad=False)
		AddTrainableMask.apply(model.lin1, 'weight', mask1_train, mask1_fixed)
		mask1_train = nn.Parameter(torch.ones_like(model.lin2.weight))
		mask1_fixed = nn.Parameter(torch.ones_like(model.lin2.weight), requires_grad=False)
		AddTrainableMask.apply(model.lin2, 'weight', mask1_train, mask1_fixed)
		mask1_train = nn.Parameter(torch.ones_like(model.node_emb.weight))
		mask1_fixed = nn.Parameter(torch.ones_like(model.node_emb.weight), requires_grad=False)
		AddTrainableMask.apply(model.node_emb, 'weight', mask1_train, mask1_fixed)

	elif dataset == 'reddit5k':
		for i in range(3):
			mask1_train = nn.Parameter(torch.ones_like(model.conv_block[i].lin_l.weight))
			mask1_fixed = nn.Parameter(torch.ones_like(model.conv_block[i].lin_l.weight), requires_grad=False)
			AddTrainableMask.apply(model.conv_block[i].lin_l, 'weight', mask1_train, mask1_fixed)
			mask1_train = nn.Parameter(torch.ones_like(model.conv_block[i].lin_r.weight))
			mask1_fixed = nn.Parameter(torch.ones_like(model.conv_block[i].lin_r.weight), requires_grad=False)
			AddTrainableMask.apply(model.conv_block[i].lin_r, 'weight', mask1_train, mask1_fixed)
		mask1_train = nn.Parameter(torch.ones_like(model.lin1.weight))
		mask1_fixed = nn.Parameter(torch.ones_like(model.lin1.weight), requires_grad=False)
		AddTrainableMask.apply(model.lin1, 'weight', mask1_train, mask1_fixed)
		mask1_train = nn.Parameter(torch.ones_like(model.lin2.weight))
		mask1_fixed = nn.Parameter(torch.ones_like(model.lin2.weight), requires_grad=False)
		AddTrainableMask.apply(model.lin2, 'weight', mask1_train, mask1_fixed)

	else:
		raise ValueError('need new addmask')


def soft_mask_init(model, seed, dataset='ba3'):
	# setup_seed(seed)
	c = 1e-5
	if dataset == 'ba3':
		for i in ('convs[0].lin1', 'convs[0].lin2', 'convs[0].lin3',
				  'convs[1].lin1', 'convs[1].lin2', 'convs[1].lin3',
				  'lin1','lin2','node_emb'):
			exec('model.%s.weight_mask_train.requires_grad = False'% i)
			exec('rand1 = (2 * torch.rand(model.%s.weight_mask_train.shape) - 1) * c'% i)
			exec('rand1 = rand1.to(model.%s.weight_mask_train.device)'% i)
			exec('rand1 = rand1 * model.%s.weight_mask_train'% i)
			exec('model.%s.weight_mask_train.add_(rand1)'% i)
			exec('model.%s.weight_mask_train.requires_grad = True'% i)
	elif dataset == 'ba2':
		for i in ('convs[0].lin1', 'convs[0].lin2', 'convs[0].lin3',
				  'convs[1].lin1', 'convs[1].lin2', 'convs[1].lin3',
				  'convs[2].lin1', 'convs[2].lin2', 'convs[2].lin3',
				  'lin1','lin2','node_emb'):
			exec('model.%s.weight_mask_train.requires_grad = False'% i)
			exec('rand1 = (2 * torch.rand(model.%s.weight_mask_train.shape) - 1) * c'% i)
			exec('rand1 = rand1.to(model.%s.weight_mask_train.device)'% i)
			exec('rand1 = rand1 * model.%s.weight_mask_train'% i)
			exec('model.%s.weight_mask_train.add_(rand1)'% i)
			exec('model.%s.weight_mask_train.requires_grad = True'% i)
	elif dataset == 'mutag':
		for i in ('convs[0].nn[0]', 'convs[0].nn[2]', 'convs[1].nn[0]',
				  'convs[1].nn[2]', 'lin1','lin2','node_emb'):
			exec('model.%s.weight_mask_train.requires_grad = False' % i)
			exec('rand1 = (2 * torch.rand(model.%s.weight_mask_train.shape) - 1) * c' % i)
			exec('rand1 = rand1.to(model.%s.weight_mask_train.device)' % i)
			exec('rand1 = rand1 * model.%s.weight_mask_train' % i)
			exec('model.%s.weight_mask_train.add_(rand1)' % i)
			exec('model.%s.weight_mask_train.requires_grad = True' % i)
	elif dataset == 'reddit5k':
		for i in ('conv_block[0].lin_l', 'conv_block[0].lin_r', 'conv_block[1].lin_l',
				  'conv_block[1].lin_r', 'conv_block[2].lin_l', 'conv_block[2].lin_r',
				  'lin1','lin2'):
			exec('model.%s.weight_mask_train.requires_grad = False' % i)
			exec('rand1 = (2 * torch.rand(model.%s.weight_mask_train.shape) - 1) * c' % i)
			exec('rand1 = rand1.to(model.%s.weight_mask_train.device)' % i)
			exec('rand1 = rand1 * model.%s.weight_mask_train' % i)
			exec('model.%s.weight_mask_train.add_(rand1)' % i)
			exec('model.%s.weight_mask_train.requires_grad = True' % i)
	else:
		raise ValueError('need new addmask')


def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	np.random.seed(seed)
	random.seed(seed)


def binary_weight(model, wei_percent, dataset='ba3'):
	wei_mask = get_mask_distribution(model,dataset=dataset)
	wei_total = wei_mask.shape[0]
	wei_y, wei_i = torch.sort(wei_mask.abs())
	### get threshold
	wei_thre_index = int(wei_total * wei_percent)
	wei_thre = wei_y[wei_thre_index]
	mask_dict = {}
	if dataset=='ba3':
		for i in ('convs.0.lin1.weight_mask_train', 'convs.0.lin2.weight_mask_train',
			  	'convs.0.lin3.weight_mask_train', 'convs.1.lin1.weight_mask_train',
			  	'convs.1.lin2.weight_mask_train', 'convs.1.lin3.weight_mask_train',
			  	'lin1.weight_mask_train','lin2.weight_mask_train','node_emb.weight_mask_train'):
			mask_dict[i] = get_each_mask(model, i, wei_thre)

	elif dataset=='ba2':
		for i in ('convs.0.lin1.weight_mask_train', 'convs.0.lin2.weight_mask_train',
			  	'convs.0.lin3.weight_mask_train', 'convs.1.lin1.weight_mask_train',
			  	'convs.1.lin2.weight_mask_train', 'convs.1.lin3.weight_mask_train',
				  'convs.2.lin1.weight_mask_train',
				  'convs.2.lin2.weight_mask_train', 'convs.2.lin3.weight_mask_train',
			  	'lin1.weight_mask_train','lin2.weight_mask_train','node_emb.weight_mask_train'):
			mask_dict[i] = get_each_mask(model, i, wei_thre)
	elif dataset=='mutag':
		for i in ('convs.0.nn.0.weight_mask_train', 'convs.0.nn.2.weight_mask_train',
			  	'convs.1.nn.0.weight_mask_train', 'convs.1.nn.2.weight_mask_train',
			  	'lin1.weight_mask_train','lin2.weight_mask_train','node_emb.weight_mask_train'):
			mask_dict[i] = get_each_mask(model, i, wei_thre)
	elif dataset=='reddit5k':
		for i in ('conv_block.0.lin_l.weight_mask_train', 'conv_block.0.lin_r.weight_mask_train',
			  	  'conv_block.1.lin_l.weight_mask_train', 'conv_block.1.lin_r.weight_mask_train',
				  'conv_block.2.lin_l.weight_mask_train', 'conv_block.2.lin_r.weight_mask_train',
			  	  'lin1.weight_mask_train','lin2.weight_mask_train'):
			mask_dict[i] = get_each_mask(model, i, wei_thre)
	return mask_dict


def get_each_mask(model, name, threshold):
	mask_weight_tensor = copy.deepcopy(model.state_dict()[name])
	ones = torch.ones_like(mask_weight_tensor)
	zeros = torch.zeros_like(mask_weight_tensor)
	mask = torch.where(mask_weight_tensor.abs() >= threshold, ones, zeros)
	return mask


def get_mask_distribution(model, dataset='ba3'):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	weight_mask_vector = torch.tensor([]).to(device)
	weight_mask_tensor= {}
	if dataset == 'ba3':
		weight_mask_tensor[1] = model.convs[0].lin1.weight_mask_train.flatten()
		weight_mask_tensor[2] = model.convs[0].lin2.weight_mask_train.flatten()
		weight_mask_tensor[3] = model.convs[0].lin3.weight_mask_train.flatten()
		weight_mask_tensor[4] = model.convs[1].lin1.weight_mask_train.flatten()
		weight_mask_tensor[5] = model.convs[1].lin2.weight_mask_train.flatten()
		weight_mask_tensor[6] = model.convs[1].lin3.weight_mask_train.flatten()
		weight_mask_tensor[7] = model.lin1.weight_mask_train.flatten()
		weight_mask_tensor[8] = model.lin2.weight_mask_train.flatten()
		weight_mask_tensor[9] = model.node_emb.weight_mask_train.flatten()

		for i in range(1, 10):
			nonzero = torch.abs(weight_mask_tensor[i]) > 1e-4
			weight_mask_vector = torch.cat([weight_mask_vector, weight_mask_tensor[i][nonzero]])
	elif dataset == 'ba2':
		weight_mask_tensor[1] = model.convs[0].lin1.weight_mask_train.flatten()
		weight_mask_tensor[2] = model.convs[0].lin2.weight_mask_train.flatten()
		weight_mask_tensor[3] = model.convs[0].lin3.weight_mask_train.flatten()
		weight_mask_tensor[4] = model.convs[1].lin1.weight_mask_train.flatten()
		weight_mask_tensor[5] = model.convs[1].lin2.weight_mask_train.flatten()
		weight_mask_tensor[6] = model.convs[1].lin3.weight_mask_train.flatten()
		weight_mask_tensor[7] = model.convs[2].lin1.weight_mask_train.flatten()
		weight_mask_tensor[8] = model.convs[2].lin2.weight_mask_train.flatten()
		weight_mask_tensor[9] = model.convs[2].lin3.weight_mask_train.flatten()
		weight_mask_tensor[10] = model.lin1.weight_mask_train.flatten()
		weight_mask_tensor[11] = model.lin2.weight_mask_train.flatten()
		weight_mask_tensor[12] = model.node_emb.weight_mask_train.flatten()

		for i in range(1, 13):
			nonzero = torch.abs(weight_mask_tensor[i]) > 1e-4
			weight_mask_vector = torch.cat([weight_mask_vector, weight_mask_tensor[i][nonzero]])
	elif dataset == 'mutag':
		weight_mask_tensor[1] = model.convs[0].nn[0].weight_mask_train.flatten()
		weight_mask_tensor[2] = model.convs[0].nn[2].weight_mask_train.flatten()
		weight_mask_tensor[3] = model.convs[1].nn[0].weight_mask_train.flatten()
		weight_mask_tensor[4] = model.convs[1].nn[2].weight_mask_train.flatten()
		weight_mask_tensor[5] = model.lin1.weight_mask_train.flatten()
		weight_mask_tensor[6] = model.lin2.weight_mask_train.flatten()
		weight_mask_tensor[7] = model.node_emb.weight_mask_train.flatten()
		for i in range(1, 8):
			nonzero = torch.abs(weight_mask_tensor[i]) > 1e-4
			weight_mask_vector = torch.cat([weight_mask_vector, weight_mask_tensor[i][nonzero]])
			# np.savez('mask', adj_mask=adj_mask_tensor.detach().cpu().numpy(), weight_mask=weight_mask_tensor.detach().cpu().numpy())
	elif dataset == 'reddit5k':
		weight_mask_tensor[1] = model.conv_block[0].lin_l.weight_mask_train.flatten()
		weight_mask_tensor[2] = model.conv_block[0].lin_r.weight_mask_train.flatten()
		weight_mask_tensor[3] = model.conv_block[1].lin_l.weight_mask_train.flatten()
		weight_mask_tensor[4] = model.conv_block[1].lin_r.weight_mask_train.flatten()
		weight_mask_tensor[5] = model.conv_block[2].lin_l.weight_mask_train.flatten()
		weight_mask_tensor[6] = model.conv_block[2].lin_r.weight_mask_train.flatten()
		weight_mask_tensor[7] = model.lin1.weight_mask_train.flatten()
		weight_mask_tensor[8] = model.lin2.weight_mask_train.flatten()
		for i in range(1, 9):
			nonzero = torch.abs(weight_mask_tensor[i]) > 1e-4
			weight_mask_vector = torch.cat([weight_mask_vector, weight_mask_tensor[i][nonzero]])
	return weight_mask_vector.detach().cpu()


def print_sparsity(model,dataset='ba3'):
	device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if dataset == 'ba3':
		path = 'trained/gnns/ba3_net.pt'
	elif dataset == 'mutag':
		path = 'trained/gnns/mutag_net.pt'
	else:
		raise ValueError('need new ')
	model_ = torch.load(path)
	weight_mask_vector = torch.tensor([]).to(device)
	for i in model_.state_dict():
		weight_mask_vector=torch.cat([weight_mask_vector,model_.state_dict()[i].flatten()])
	all_num=weight_mask_vector.shape[0]

	weight_tota=0
	weight_nonzero_tota = 0
	if dataset=='ba3':
		para_list=['convs.0.lin1.weight_mask_train', 'convs.0.lin2.weight_mask_train',
			  'convs.0.lin3.weight_mask_train', 'convs.1.lin1.weight_mask_train',
			  'convs.1.lin2.weight_mask_train', 'convs.1.lin3.weight_mask_train',
			  'lin1.weight_mask_train']
	elif dataset=='mutag':
		para_list = ['convs.0.nn.0.weight_mask_train', 'convs.0.nn.2.weight_mask_train',
					 'convs.1.nn.0.weight_mask_train', 'convs.1.nn.2.weight_mask_train',
					 'lin1.weight_mask_train','lin2.weight_mask_train','node_emb.weight_mask_train']
	else:
		raise ValueError('need new ')
	for i in para_list:
		para=model.state_dict()[i]
		weight_num= para.numel()
		weight_tota+=weight_num

		weight_nonzero = para.sum().item()
		weight_nonzero_tota+=weight_nonzero

	wei_zero= weight_tota-weight_nonzero_tota
	spa=round(1-wei_zero/all_num,3)
	print(spa)
	return spa

def binary_weight_qz(model, wei_percent, dataset='ba3'):
	wei_mask = get_mask_distribution_qz(model,dataset=dataset)
	wei_total = wei_mask.shape[0]
	# print('wei_total:',wei_total)
	### sort
	wei_y, wei_i = torch.sort(wei_mask.abs())
	### get threshold
	wei_thre_index = int(wei_total * wei_percent)
	wei_thre = wei_y[wei_thre_index]

	mask_dict = {}
	if dataset=='ba3':
		mask_dict['convs.0.lin1.weight_mask_train'] = get_each_mask(model, 'convs.0.lin1.weight_orig_weight', wei_thre)
		mask_dict['convs.0.lin2.weight_mask_train'] = get_each_mask(model, 'convs.0.lin2.weight_orig_weight',wei_thre)
		mask_dict['convs.0.lin3.weight_mask_train'] = get_each_mask(model, 'convs.0.lin3.weight_orig_weight',wei_thre)
		mask_dict['convs.1.lin1.weight_mask_train'] = get_each_mask(model, 'convs.1.lin1.weight_orig_weight',wei_thre)
		mask_dict['convs.1.lin2.weight_mask_train'] = get_each_mask(model, 'convs.1.lin2.weight_orig_weight', wei_thre)
		mask_dict['convs.1.lin3.weight_mask_train'] = get_each_mask(model, 'convs.1.lin3.weight_orig_weight', wei_thre)
		mask_dict['lin1.weight_mask_train'] = get_each_mask(model, 'lin1.weight_orig_weight', wei_thre)
		mask_dict['lin2.weight_mask_train'] = get_each_mask(model, 'lin2.weight_orig_weight', wei_thre)
		mask_dict['node_emb.weight_mask_train'] = get_each_mask(model, 'node_emb.weight_orig_weight', wei_thre)

	elif dataset=='mutag':
		for i in ('convs.0.nn.0.weight_mask_train', 'convs.0.nn.2.weight_mask_train',
			  	'convs.1.nn.0.weight_mask_train', 'convs.1.nn.2.weight_mask_train',
			  	'lin1.weight_mask_train','lin2.weight_mask_train','node_emb.weight_mask_train'):
			mask_dict[i] = get_each_mask(model, i, wei_thre)
	elif dataset=='reddit5k':
		for i in ('conv_block.0.lin_l.weight_mask_train', 'conv_block.0.lin_r.weight_mask_train',
			  	  'conv_block.1.lin_l.weight_mask_train', 'conv_block.1.lin_r.weight_mask_train',
				  'conv_block.2.lin_l.weight_mask_train', 'conv_block.2.lin_r.weight_mask_train',
			  	  'lin1.weight_mask_train','lin2.weight_mask_train'):
			mask_dict[i] = get_each_mask(model, i, wei_thre)
	return mask_dict

def get_mask_distribution_qz(model, dataset='ba3'):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	weight_mask_vector = torch.tensor([]).to(device)
	weight_mask_tensor= {}
	if dataset == 'ba3':
		weight_mask_tensor[1] = model.convs[0].lin1.weight_orig_weight.flatten()
		weight_mask_tensor[2] = model.convs[0].lin2.weight_orig_weight.flatten()
		weight_mask_tensor[3] = model.convs[0].lin3.weight_orig_weight.flatten()
		weight_mask_tensor[4] = model.convs[1].lin1.weight_orig_weight.flatten()
		weight_mask_tensor[5] = model.convs[1].lin2.weight_orig_weight.flatten()
		weight_mask_tensor[6] = model.convs[1].lin3.weight_orig_weight.flatten()
		weight_mask_tensor[7] = model.lin1.weight_orig_weight.flatten()
		weight_mask_tensor[8] = model.lin2.weight_orig_weight.flatten()
		weight_mask_tensor[9] = model.node_emb.weight_orig_weight.flatten()

		for i in range(1, 10):
			weight_mask_vector = torch.cat([weight_mask_vector, weight_mask_tensor[i]])

	elif dataset == 'mutag':
		weight_mask_tensor[1] = model.convs[0].nn[0].weight_orig_weight.flatten()
		weight_mask_tensor[2] = model.convs[0].nn[2].weight_orig_weight.flatten()
		weight_mask_tensor[3] = model.convs[1].nn[0].weight_orig_weight.flatten()
		weight_mask_tensor[4] = model.convs[1].nn[2].weight_orig_weight.flatten()
		weight_mask_tensor[5] = model.lin1.weight_orig_weight.flatten()
		weight_mask_tensor[6] = model.lin2.weight_orig_weight.flatten()
		weight_mask_tensor[7] = model.node_emb.weight_orig_weight.flatten()
		for i in range(1, 8):
			nonzero = torch.abs(weight_mask_tensor[i]) > 1e-4
			weight_mask_vector = torch.cat([weight_mask_vector, weight_mask_tensor[i][nonzero]])
			# np.savez('mask', adj_mask=adj_mask_tensor.detach().cpu().numpy(), weight_mask=weight_mask_tensor.detach().cpu().numpy())
	elif dataset == 'reddit5k':
		weight_mask_tensor[1] = model.conv_block[0].lin_l.weight_orig_weight.flatten()
		weight_mask_tensor[2] = model.conv_block[0].lin_r.weight_orig_weight.flatten()
		weight_mask_tensor[3] = model.conv_block[1].lin_l.weight_orig_weight.flatten()
		weight_mask_tensor[4] = model.conv_block[1].lin_r.weight_orig_weight.flatten()
		weight_mask_tensor[5] = model.conv_block[2].lin_l.weight_orig_weight.flatten()
		weight_mask_tensor[6] = model.conv_block[2].lin_r.weight_orig_weight.flatten()
		weight_mask_tensor[7] = model.lin1.weight_orig_weight.flatten()
		weight_mask_tensor[8] = model.lin2.weight_orig_weight.flatten()
		for i in range(1, 9):
			nonzero = torch.abs(weight_mask_tensor[i]) > 1e-4
			weight_mask_vector = torch.cat([weight_mask_vector, weight_mask_tensor[i][nonzero]])
	return weight_mask_vector.detach().cpu()



import os
import copy
import math
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

def visualize_all(graph,path,dataset='ba3'):

	G = nx.DiGraph()
	G.add_nodes_from(range(graph.num_nodes))
	G.add_edges_from(list(graph.edge_index.cpu().numpy().T))
		# if self.vis_dict is None:
		# 	self.vis_dict = vis_dict[self.model_name] if self.model_name in vis_dict.keys() else vis_dict['defult']
	pos = nx.kamada_kawai_layout(G) if graph.pos is None else graph.pos[0]
	edge_pos_mask = np.ones(graph.num_edges, dtype=np.bool_)
		# edge_pos_mask[np.array(graph.ground_truth_mask[0], dtype=np.bool_)] = True
	vmax = sum(edge_pos_mask)
	node_pos_mask = np.zeros(graph.num_nodes, dtype=np.bool_)
	node_neg_mask = np.zeros(graph.num_nodes, dtype=np.bool_)
	node_pos_idx = np.unique(graph.edge_index[:, edge_pos_mask].cpu().numpy()).tolist()
	node_neg_idx = list(set([i for i in range(graph.num_nodes)]) - set(node_pos_idx))
	node_pos_mask[node_pos_idx] = True
	node_neg_mask[node_neg_idx] = True
	nx.draw_networkx_nodes(G, pos={i: pos[i] for i in node_pos_idx},
							   nodelist=node_pos_idx,
							   node_color=graph.z[0][node_pos_idx],
							   alpha=1, cmap='winter',
							   edgecolors='red',
							   vmin=-max(graph.z[0]), vmax=max(graph.z[0])
							   )
	nx.draw_networkx_nodes(G, pos={i: pos[i] for i in node_neg_idx},
							   nodelist=node_neg_idx,
							   node_color=graph.z[0][node_neg_idx],
							   alpha=0.2, cmap='winter',
							   edgecolors='whitesmoke',
							   vmin=-max(graph.z[0]), vmax=max(graph.z[0])
							   )
	if 'ba3' in dataset:

		nx.draw_networkx_edges(G, pos=pos,
								   edgelist=list(graph.edge_index.cpu().numpy().T),
								   edge_color='grey',
								   arrows=False
								   )


	nx.draw_networkx_edges(G, pos=pos,
							   edgelist=list(graph.edge_index[:, edge_pos_mask].cpu().numpy().T),
							   # np.ones(len(edge_imp[edge_pos_mask])),
							   edge_cmap=cm.get_cmap('bwr'),
							   edge_vmin=-vmax, edge_vmax=vmax,
							   arrows=False
							   )
	ax = plt.gca()
	ax.set_facecolor('aliceblue')

	folder = Path(path)
	if not os.path.exists(folder):
		os.makedirs(folder)
	if isinstance(graph.name[0], str):
		plt.savefig(folder / Path(r'%d-%s.png' % (int(graph.y), str(graph.name[0]))), dpi=500,
					bbox_inches='tight')
	else:
		plt.savefig(folder / Path(r'%d-%d.png' % (graph.y, int(graph.name[0]))), dpi=500, bbox_inches='tight')

	plt.cla()


