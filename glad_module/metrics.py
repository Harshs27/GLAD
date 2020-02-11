import numpy as np
import sklearn
from sklearn import metrics

def report_metrics(G_true, G):
    """Compute FDR, TPR, and FPR for B
    Args:
        B_true: ground truth adj matrix
        B: predicted adj mat
    Returns:
        fdr: (false positive) / prediction positive = FP/P
        tpr: (true positive) / condition positive = TP/T
        fpr: (false positive) / condition negative = FP/F
        shd: undirected extra + undirected missing = E+M
        nnz: prediction positive
        ps : probability of success, sign match
    """
    B_true = G_true != 0
    B = G != 0
    d = B.shape[-1]

    # Probability of success : 1 = perfect match
    ps = int(np.all(np.sign(G)==np.sign(G_true)))

    # AUC
#    print('G , G_true', G, G_true, np.where(G_true>0, 1, 0).reshape(-1), G.reshape(-1))
    G_true_binary = np.where(G_true>0, 1, 0).reshape(-1)
    sk_fpr, sk_tpr, sk_th = metrics.roc_curve(G_true_binary.reshape(-1), G.reshape(-1))
    auc = metrics.auc(sk_fpr, sk_tpr)
    #print('auc = ', auc)
    #br

    # linear index of nonzeros
    pred = np.flatnonzero(B)
    cond = np.flatnonzero(B_true)
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    TP = (len(true_pos) - d)/2 + d
    # false pos
    false_pos = np.setdiff1d(pred, cond, assume_unique=True)
    FP = len(false_pos)/2
    # P = set of estimated edges
    P = max((len(pred)-d)/2+d, 1)
    # T = set of true edges
    T = max((len(cond)-d)/2+d, 1)
    # F = set of non-edges in ground truth graph
    F = max((d**2 - len(cond))/2, 1)
    # extra
    E = len(set(pred)-set(cond))/2
    # missing
    M = len(set(cond)-set(pred))/2
    # compute ratio
    fdr = float(FP) / P
    tpr = float(TP) / T
    fpr = float(FP) / F
    # structural hamming distance
    shd = E+M
#    print('FP=', FP, ' TP=',TP, ' P=', P, ' T=', T, ' F=',F, ' E=', E, ' M=', M)
    return fdr, tpr, fpr, shd, (len(pred)-d)/2 +d, (len(cond)-d)/2+d, ps#, auc


def main():
    a_pred = np.array([[0.74, 0.02, 0.01, 0, 0], [0.02, 1.25, 0,0,0], [0.01,0,0.79,0,0], [0,0,0,0.81,0], [0,0,0,0,0.78]])
    # sign match check
    a_pred2 = np.array([[1.33, 0.32, 0,0,0], [0.32,1.33,0.02,0,0.08], [0, 0.02,1.33,0,0], [0,0,0,1.33,0], [0,0.08,0,0,1.33]])
  
    a_true = np.array([[1.33, 0.32, 0,0,0], [0.32,1.33,-0.02,0,-0.08], [0, -0.02,1.33,0,0], [0,0,0,1.33,0], [0,-0.08,0,0,1.33]])
    print(a_pred, a_true)
    print(report_metrics(a_true, a_pred))
    print(report_metrics(a_true, a_pred2))
    print(report_metrics(a_true, a_true))

if __name__=="__main__":
    main()

