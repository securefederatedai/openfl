# Copyright (c) 2012-2022 Clement Vignac, Igor Krawczuk, Antoine Siraudin
# source: https://github.com/cvignac/DiGress/

import torch
from digress import utils


class ExtraMolecularFeatures:
    def __init__(self, dataset_infos):
        self.charge = ChargeFeature(remove_h=dataset_infos.remove_h, valencies=dataset_infos.valencies)
        self.valency = ValencyFeature()
        self.weight = WeightFeature(max_weight=dataset_infos.max_weight, atom_weights=dataset_infos.atom_weights)

    def __call__(self, noisy_data):
        charge = self.charge(noisy_data).unsqueeze(-1)      # (bs, n, 1)
        valency = self.valency(noisy_data).unsqueeze(-1)    # (bs, n, 1)
        weight = self.weight(noisy_data)                    # (bs, 1)

        extra_edge_attr = torch.zeros((*noisy_data['E_t'].shape[:-1], 0)).type_as(noisy_data['E_t'])

        return utils.PlaceHolder(X=torch.cat((charge, valency), dim=-1), E=extra_edge_attr, y=weight)


class ChargeFeature:
    def __init__(self, remove_h, valencies):
        self.remove_h = remove_h
        self.valencies = valencies

    def __call__(self, noisy_data):
        bond_orders = torch.tensor([0, 1, 2, 3, 1.5], device=noisy_data['E_t'].device).reshape(1, 1, 1, -1)
        weighted_E = noisy_data['E_t'] * bond_orders      # (bs, n, n, de)
        current_valencies = weighted_E.argmax(dim=-1).sum(dim=-1)   # (bs, n)

        valencies = torch.tensor(self.valencies, device=noisy_data['X_t'].device).reshape(1, 1, -1)
        X = noisy_data['X_t'] * valencies  # (bs, n, dx)
        normal_valencies = torch.argmax(X, dim=-1)               # (bs, n)

        return (normal_valencies - current_valencies).type_as(noisy_data['X_t'])


class ValencyFeature:
    def __init__(self):
        pass

    def __call__(self, noisy_data):
        orders = torch.tensor([0, 1, 2, 3, 1.5], device=noisy_data['E_t'].device).reshape(1, 1, 1, -1)
        E = noisy_data['E_t'] * orders      # (bs, n, n, de)
        valencies = E.argmax(dim=-1).sum(dim=-1)    # (bs, n)
        return valencies.type_as(noisy_data['X_t'])


class WeightFeature:
    def __init__(self, max_weight, atom_weights):
        self.max_weight = max_weight
        self.atom_weight_list = torch.tensor(list(atom_weights.values()))

    def __call__(self, noisy_data):
        X = torch.argmax(noisy_data['X_t'], dim=-1)     # (bs, n)
        X_weights = self.atom_weight_list.to(X.device)[X]           # (bs, n)
        return X_weights.sum(dim=-1).unsqueeze(-1).type_as(noisy_data['X_t']) / self.max_weight     # (bs, 1)