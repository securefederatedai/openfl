from data_transf import segmentation_auroc, detection_auroc
import os
import shutil
import numpy as np
from utils import distribute_scores
import ngtpy
from sklearn.neighbors import KDTree


def search_NN(test_emb, train_emb_flat, NN=1, method='kdt'):
    if method == 'ngt':
        return search_NN_ngt(test_emb, train_emb_flat, NN=NN)

    kdt = KDTree(train_emb_flat)

    Ntest, I, J, D = test_emb.shape
    closest_inds = np.empty((Ntest, I, J, NN), dtype=np.int32)
    l2_maps = np.empty((Ntest, I, J, NN), dtype=np.float32)

    for n in range(Ntest):
        for i in range(I):
            dists, inds = kdt.query(test_emb[n, i, :, :], return_distance=True, k=NN)
            closest_inds[n, i, :, :] = inds[:, :]
            l2_maps[n, i, :, :] = dists[:, :]

    return l2_maps, closest_inds


def search_NN_ngt(test_emb, train_emb_flat, NN=1):
    Ntest, I, J, D = test_emb.shape
    closest_inds = np.empty((Ntest, I, J, NN), dtype=np.int32)
    l2_maps = np.empty((Ntest, I, J, NN), dtype=np.float32)

    # os.makedirs('tmp', exist_ok=True)
    dpath = f'/tmp/{os.getpid()}'
    ngtpy.create(dpath, D)
    index = ngtpy.Index(dpath)
    index.batch_insert(train_emb_flat)

    for n in range(Ntest):
        for i in range(I):
            for j in range(J):
                query = test_emb[n, i, j, :]
                results = index.search(query, NN)
                inds = [result[0] for result in results]

                closest_inds[n, i, j, :] = inds
                vecs = np.asarray([index.get_object(inds[nn]) for nn in range(NN)])
                dists = np.linalg.norm(query - vecs, axis=-1)
                l2_maps[n, i, j, :] = dists
    shutil.rmtree(dpath)

    return l2_maps, closest_inds


def assess_anomaly_maps(obj, anomaly_maps, masks, labels):
    auroc_seg = segmentation_auroc(obj, anomaly_maps, masks)

    anomaly_scores = anomaly_maps.max(axis=-1).max(axis=-1)
    auroc_det = detection_auroc(obj, anomaly_scores, labels)
    return auroc_det, auroc_seg


def eval_embeddings_NN_multiK(obj, embs64, embs32, masks, labels, NN=1):
    emb_tr, emb_te = embs64
    maps_64 = measure_emb_NN(emb_te, emb_tr, method='kdt', NN=NN)
    maps_64 = distribute_scores(maps_64, (256, 256), K=64, S=16)
    det_64, seg_64 = assess_anomaly_maps(obj, maps_64, masks, labels)

    emb_tr, emb_te = embs32
    maps_32 = measure_emb_NN(emb_te, emb_tr, method='ngt', NN=NN)
    maps_32 = distribute_scores(maps_32, (256, 256), K=32, S=4)
    det_32, seg_32 = assess_anomaly_maps(obj, maps_32, masks, labels)

    maps_sum = maps_64 + maps_32
    det_sum, seg_sum = assess_anomaly_maps(obj, maps_sum, masks, labels)

    maps_mult = maps_64 * maps_32
    det_mult, seg_mult = assess_anomaly_maps(obj, maps_mult, masks, labels)

    return {
        'det_64': det_64,
        'seg_64': seg_64,

        'det_32': det_32,
        'seg_32': seg_32,

        'det_sum': det_sum,
        'seg_sum': seg_sum,

        'det_mult': det_mult,
        'seg_mult': seg_mult,

        'maps_64': maps_64,
        'maps_32': maps_32,
        'maps_sum': maps_sum,
        'maps_mult': maps_mult,
    }


########################

def measure_emb_NN(emb_te, emb_tr, method='kdt', NN=1):
    D = emb_tr.shape[-1]
    train_emb_all = emb_tr.reshape(-1, D)

    l2_maps, _ = search_NN(emb_te, train_emb_all, method=method, NN=NN)
    anomaly_maps = np.mean(l2_maps, axis=-1)

    return anomaly_maps
