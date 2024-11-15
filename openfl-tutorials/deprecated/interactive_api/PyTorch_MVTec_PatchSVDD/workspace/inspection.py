# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inspection of images and patches."""

import os
import shutil

import ngtpy
import numpy as np
from data_transf import bal_acc_score
from data_transf import detection_auroc
from data_transf import segmentation_auroc
from sklearn.neighbors import KDTree
from utils import distribute_scores


def search_nn(test_emb, train_emb_flat, nn=1, method='kdt'):
    """Seach nearest neighbors."""
    if method == 'ngt':
        return search_nn_ngt(test_emb, train_emb_flat, nn=nn)

    kdt = KDTree(train_emb_flat)

    ntest, i, j, d = test_emb.shape
    closest_inds = np.empty((ntest, i, j, nn), dtype=np.int32)
    l2_maps = np.empty((ntest, i, j, nn), dtype=np.float32)

    for n_ in range(ntest):
        for i_ in range(i):
            dists, inds = kdt.query(test_emb[n_, i_, :, :], return_distance=True, k=nn)
            closest_inds[n_, i_, :, :] = inds[:, :]
            l2_maps[n_, i_, :, :] = dists[:, :]

    return l2_maps, closest_inds


def search_nn_ngt(test_emb, train_emb_flat, nn=1):
    """Search nearest neighbors."""
    ntest, i, j, d = test_emb.shape
    closest_inds = np.empty((ntest, i, j, nn), dtype=np.int32)
    l2_maps = np.empty((ntest, i, j, nn), dtype=np.float32)

    dpath = f'/tmp/{os.getpid()}'
    ngtpy.create(dpath, d)
    index = ngtpy.Index(dpath)
    index.batch_insert(train_emb_flat)

    for n_ in range(ntest):
        for i_ in range(i):
            for j_ in range(j):
                query = test_emb[n_, i_, j_, :]
                results = index.search(query, nn)
                inds = [result[0] for result in results]

                closest_inds[n_, i_, j_, :] = inds
                vecs = np.asarray([index.get_object(inds[nn_]) for nn_ in range(nn)])
                dists = np.linalg.norm(query - vecs, axis=-1)
                l2_maps[n_, i_, j_, :] = dists
    shutil.rmtree(dpath)

    return l2_maps, closest_inds


def assess_anomaly_maps(obj, anomaly_maps, masks, labels):
    """Assess anomaly maps."""
    auroc_seg = segmentation_auroc(obj, anomaly_maps, masks)

    anomaly_scores = anomaly_maps.max(axis=-1).max(axis=-1)
    auroc_det = detection_auroc(obj, anomaly_scores, labels)
    ba_score = bal_acc_score(obj, anomaly_scores, labels)
    return auroc_det, auroc_seg, ba_score


def eval_embeddings_nn_multik(obj, embs64, embs32, masks, labels, nn=1):
    """Evaluate embeddings."""
    emb_tr, emb_te = embs64
    maps_64 = measure_emb_nn(emb_te, emb_tr, method='kdt', nn=nn)
    maps_64 = distribute_scores(maps_64, (256, 256), k=64, s=16)
    det_64, seg_64, ba_64 = assess_anomaly_maps(obj, maps_64, masks, labels)

    emb_tr, emb_te = embs32
    maps_32 = measure_emb_nn(emb_te, emb_tr, method='ngt', nn=nn)
    maps_32 = distribute_scores(maps_32, (256, 256), k=32, s=4)
    det_32, seg_32, ba_32 = assess_anomaly_maps(obj, maps_32, masks, labels)

    maps_sum = maps_64 + maps_32
    det_sum, seg_sum, ba_sum = assess_anomaly_maps(obj, maps_sum, masks, labels)

    maps_mult = maps_64 * maps_32
    det_mult, seg_mult, ba_mult = assess_anomaly_maps(obj, maps_mult, masks, labels)

    return {
        'det_64': det_64,
        'seg_64': seg_64,
        'bal_acc_64': ba_64,

        'det_32': det_32,
        'seg_32': seg_32,
        'bal_acc_32': ba_32,

        'det_sum': det_sum,
        'seg_sum': seg_sum,
        'bal_acc_sum': ba_sum,

        'det_mult': det_mult,
        'seg_mult': seg_mult,
        'bal_acc_mult': ba_mult,

        'maps_64': maps_64,
        'maps_32': maps_32,
        'maps_sum': maps_sum,
        'maps_mult': maps_mult,
    }


def eval_embeddings_nn_maps(obj, embs64, embs32, masks, labels, nn=1):
    """Evaluate embeddings."""
    emb_tr, emb_te = embs64
    maps_64 = measure_emb_nn(emb_te, emb_tr, method='kdt', nn=nn)
    maps_64 = distribute_scores(maps_64, (256, 256), k=64, s=16)

    emb_tr, emb_te = embs32
    maps_32 = measure_emb_nn(emb_te, emb_tr, method='ngt', nn=nn)
    maps_32 = distribute_scores(maps_32, (256, 256), k=32, s=4)

    maps_mult = maps_64 * maps_32
    return maps_mult


def measure_emb_nn(emb_te, emb_tr, method='kdt', nn=1):
    """Measure embeddings."""
    d = emb_tr.shape[-1]
    train_emb_all = emb_tr.reshape(-1, d)

    l2_maps, _ = search_nn(emb_te, train_emb_all, method=method, nn=nn)
    anomaly_maps = np.mean(l2_maps, axis=-1)

    return anomaly_maps
