import numpy as np
import dgl
import os
import networkx as nx

from ..graph import DGLGraph

class Sync_pool_dataset():
    """A synthetic dataset for graph pooling.

    This dataset contains several subgraphs, with only sparse connections among
    them. The set of subgraphs consist of 2 types, A and B. If there are more A
    than in B, then the whole graph is classified as A; vice versa.


    Parameters
    ----------
    num_graphs: int
        Number of composite graph in this dataset
    gen_graph_type: string
        The type of graph we use to generate composite graph.
        For now, we assume both class are generated from the same graph
        generator (only the node feature is different!)
    num_sub_graphs: int
        Number of subgraphs in each component. For now we assume it's a fixed
        number, but it could change.
    split_ratio: float
        Split ratio between class A and class B.
    feature_type: string
        feature type of each sub graph. Could be gaussian with different mean
        and variance, tuned by subgraph type.
    graph_label: int
        the composite graph label
    data_split_ratio: list
        list of dataset split ratio that sums to 1.
    mode: string
        decide what to return


    Return
    ------
    1) If backend = default: return nx graph and np feature tensor.
    2) If backend = DGL: return DGL graph and DGL backend tensor.
    """

    def __init__(self, num_graphs, gen_graph_type='default', num_sub_graphs=10,
                 feature_type='gaussian', num_graph_type=2, data_split_ratio = [0.8,0.1,0.1]):
        super(Sync_pool_dataset, self).__init__()
        self.num_graphs = num_graphs
        self.gen_graph_type = gen_graph_type
        self.num_sub_graphs = num_sub_graphs
        self.split_ratio = split_ratio
        self.feature_type = feature_type
        self.graph_label = graph_label
        self.min_nodes = 30
        self.max_nodes = 50
        self.min_deg = 3
        self.max_deg = 5
        self.A_params = {'label':0,'mean':np.random.uniform(0,1),
                         'variance':np.random.uniform(0,1),
                         'dim':32}
        self.B_params = {'label':1, 'mean':np.random.uniform(0,1),
                         'variance':np.random.uniform(0,1),
                         'dim':32}
        self.graphs = []
        self.feature = []
        self.gen_graphs()
        label = [i for i in range(num_graph_type)]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def _get_all(self):
        return self.graphs, self.feature

    def gen_graphs(self):
        for n in range(self.num_graphs):
            graphs = []
            feats = []
            labels = []
            split_ratio = np.random.uniform(0,1)
            n_A = int(split_ratio * self.num_sub_graphs)
            n_B = self.num_sub_graphs - n_A
            for i in range(n_A):
                g, feat = gen_component(self.feature_type, self.A_params,
                                        self.min_nodes, self.max_nodes,
                                        self.min_deg, self.max_deg)
                graphs.append(g)
                feats.append(feat)
            for i in range(n_B):
                g, feat = gen_component(self.feature_type, self.B_params,
                                        self.min_nodes, self.max_nodes,
                                        self.min_deg, self.max_deg)
                graphs.append(g)
                feats.append(feat)

            if n_A > n_B:
                composite_label = 0
            else:
                composite_label = 1





    def gen_component(self, feature_type, feature_params,
                      min_nodes, max_nodes, min_deg, max_deg):
        num_n = np.random.randint(min_nodes, max_nodes)
        deg = np.random.randint(min_deg, max_deg)
        g = nx.random_regular_graph(deg, num_n)
        g = max(nx.connected_component_subgraphs(g), key=len)
        # ensure connected component
        if feature_type == 'gaussian':
            assert 'mean' in feature_params.keys()
            assert 'variance' in feature_params.keys()
            assert 'dim' in feature_params.keys()
            feat = np.random.normal(feature_params['mean'],
                                    feature_params['variance'],
                                    (g.number_of_nodes(),
                                     feature_params['dim']))

            feat_dict = {i: {'feat': feat[i,:]} for i in range(feat.shape[0])}
            nx.set_node_attributes(g, feat_dict)
        else:
            raise NotImplementedError

        return g, feat
