import numpy as np
import gzip
import torch
from torch_geometric.data import Data
from data.MNIST.load_mnist_graph import load_mnist_graph


def Mnist(data_size=1000):
	return load_mnist_graph(data_size)
