import numpy as np


class SimDataset(object):
    """docstring for SimDataset"""
    def __init__(self,
                 num_samples,
                 num_nodes,
                 node_features,
                 edge_features):
        super(SimDataset, self).__init__()
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.edge_features = edge_features
        self.data = self.__generate_data()

    def __generate_data(self):
        sample_size = self.sample_size()
        return np.zeros((self.num_samples, sample_size + 1), dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index].flatten()

    def sample_size(self):
        number_edges = int((self.num_nodes*(self.num_nodes-1))/2)
        node_feat_mat = self.num_nodes * self.node_features
        edge_feat_mat = number_edges * self.edge_features
        edge_adj = self.num_nodes **3
        covalent_mat = self.num_nodes ** 2
        non_covalent_mat = self.num_nodes ** 2
        ligand_only_mat = self.num_nodes ** 2
        return node_feat_mat + edge_feat_mat + edge_adj + covalent_mat \
            + non_covalent_mat + ligand_only_mat + 1


dataset = SimDataset(100, 50, 19, 1)


def get_train(index):
    return dataset[index]


def num_train_samples():
    return len(dataset)


def sample_dims():
    return (dataset.sample_size(),)
