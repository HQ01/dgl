import dgl
import torch

def test_to_device():
    g = dgl.DGLGraph()
    g.add_nodes(5, {'h': torch.ones((5,2))})
    g.add_edges([0,1],[1,2], {'m' : torch.ones((2,2))})
    device = torch.device('cuda')
    g.to(device)
    assert g.ndata['h'].is_cuda
    assert g.edata['m'].is_cuda


if __name__ == '__main__':
    test_to_device()