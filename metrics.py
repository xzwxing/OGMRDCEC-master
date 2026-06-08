import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score
from scipy.optimize import linear_sum_assignment

nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(-w)
    correct = sum(w[i, j] for i, j in zip(row_ind, col_ind))
    return correct / y_pred.size


def f1(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(-w)
    mapping = dict(zip(row_ind, col_ind))
    y_pred_mapped = np.array([mapping.get(lbl, lbl) for lbl in y_pred])

    return f1_score(y_true, y_pred_mapped, average='macro')


def evaluate_clustering(y_true, y_pred):
    return {
        'ACC': acc(y_true, y_pred),
        'NMI': nmi(y_true, y_pred),
        'ARI': ari(y_true, y_pred),
        'F1': f1(y_true, y_pred)
    }


# ==================== test ====================
if __name__ == "__main__":
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
    y_pred = np.array([1, 1, 0, 0, 2, 2, 1, 0, 2])

    res = evaluate_clustering(y_true, y_pred)
    print(f"ACC: {res['ACC']:.4f}")
    print(f"NMI: {res['NMI']:.4f}")
    print(f"ARI: {res['ARI']:.4f}")
    print(f"F1:  {res['F1']:.4f}")
