import torch
import dgl
from dgl.data import sync_pool

test_sync_pool = sync_pool.SyncPoolDataset(20, num_sub_graphs=10)
candidate_graph, feature, label, n2sub_label = test_sync_pool[1]
