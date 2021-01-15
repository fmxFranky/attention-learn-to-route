import os
import pickle

import torch
from torch.utils.data import Dataset

from problems.tsp.state_tsp import StateTSP
from utils.beam_search import beam_search


class TSP(object):

    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi, sep=None):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (torch.arange(pi.size(1), out=pi.data.new()).view(
            1, -1).expand_as(pi) == pi.data.sort(1)[0]).all(), "Invalid tour"

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        if sep is None:
            torch.ones(d.size(0), 1, device=d.device) * d.size(1)

        # Generate the mask of fake actions
        indicators = sep.repeat(1, d.size(1)) - torch.arange(
            0, d.size(1))[None, :].repeat(d.size(0), 1).to(d.device)
        mask = indicators <= 0

        # Calculate the real cost
        last_node_coords = d[indicators == 1.]
        d[mask] = 0.0
        # add a zero column to d
        zeros = torch.zeros(d.size(0), 1, d.size(-1)).to(d.device)
        d_z = torch.cat([d, zeros], dim=1)
        length_a = (d_z[:, 1:] - d_z[:, :-1]).norm(p=2, dim=2).sum(1)
        length_b = last_node_coords.norm(p=2, dim=1)
        length_c = (last_node_coords - d_z[:, 0]).norm(p=2, dim=1)

        return length_a - length_b + length_c, mask

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input,
                    beam_size,
                    expand_size=None,
                    compress_mask=False,
                    model=None,
                    max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam,
                fixed,
                expand_size,
                normalize=True,
                max_calc_batch_size=max_calc_batch_size)

        state = TSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8)

        return beam_search(state, beam_size, propose_expansions)


class TSPDataset(Dataset):
    def __init__(self,
                 filename=None,
                 size=50,
                 num_samples=1000000,
                 offset=0,
                 distribution=None):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [
                    torch.FloatTensor(row)
                    for row in (data[offset:offset + num_samples])
                ]
        else:
            # Sample points randomly in [0, 1] square
            self.data = [
                torch.FloatTensor(size, 2).uniform_(0, 1)
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
