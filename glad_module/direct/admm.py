#import sys
#print(sys.path)

import argparse, random
from expts_glad.create_GGM import create_MN_vary_w


import torch, scipy
import networkx as nx
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau
import numpy as np 
from scipy.linalg import sqrtm
from expts_glad import metrics
from expts_glad.torch_sqrtm import MatrixSquareRoot
import pprint
import time

TRAIN = True #False# True

parser = argparse.ArgumentParser(description='Structure recovery of graphical model using admm')
parser.add_argument('--K_train', type=int, default=1, #1000,
                    help='Num of training examples for a fixed D')
parser.add_argument('--K_valid', type=int, default=1, #1000,
                    help='Number of valid examples for a fixed D ')
parser.add_argument('--K_test', type=int, default=10, #1000,
                    help='Number of testing examples for a fixed D')
parser.add_argument('--M', type=int, default= 4, #1000, #500, #6,#500,
                    help='number of samples M, R^(NxM)')
parser.add_argument('--SAMPLE_BATCHES', type=int, default= 5, #1000, #500, #6,#500,
                    help='number of batches of sample size M')
parser.add_argument('--N', type=int, default= 5, #250, #3,#250,
                    help='Number of nodes N, R^(NxM)')
parser.add_argument('--p_min', type=float, default=0.12,
                    help='the weights of values in true precision matrices')
parser.add_argument('--p_max', type=float, default=0.25,
                    help='the weights of values in true precision matrices')
parser.add_argument('--w_min', type=float, default=-1,
                    help='the weights of values in true precision matrices')
parser.add_argument('--w_max', type=float, default=1,
                    help='the weights of values in true precision matrices')
parser.add_argument('--prob', type=float, default=0.05,
                    help='sparsity = 2*prob: probability for the erdos-renyi true graph')
parser.add_argument('--MAX_DEG', type=int, default= 50, #250, #3,#250,
                    help='Max degree of the random erdos-renyi graph')
parser.add_argument('--SIGNS', type=int, default= 0, #250, #3,#250,
                    help='1: create random graphs with postive and negative signs')
parser.add_argument('--rho', type=float, default=0.1, #075,
                    help='penalty term for L1(theta)')
parser.add_argument('--lambda_init', type=float, default=0.6,
                    help='initial step sizes for unrolled architecture')
parser.add_argument('--alpha_lr', type=float, default=0.0001,
                    help='learning rate of lambda')
parser.add_argument('--theta_init_offset', type=float, default=0.1, #0.03, #075,
                    help='offset for setting the diagonal init theta approximation')
parser.add_argument('--L', type=int, default=50,
                    help='Unroll the network L times')
parser.add_argument('--cp', type=float, default=0.2,
                    help='penalty for cholesky decomposition')
parser.add_argument('--INIT_DIAG', type=int, default=0,
                    help='1 : initialize the theta0 diagonally')
parser.add_argument('--USE_CUDA_FLAG', type=int, default=1,
                    help='USE GPU if = 1')
parser.add_argument('--varyS', type=int, default=0,
                    help='experiment to do varyS')
parser.add_argument('--EDGE_RECOVERY', type=int, default=1,
                    help='experiment to get the probability of success plots')
parser.add_argument('--graph_type', type=str, default='random_maxd',
                    help='grid/chain/star')



args = parser.parse_args()

# Global Variables
USE_CUDA = False
if args.USE_CUDA_FLAG == 1:
    USE_CUDA = True


def convert_to_torch(data, TESTING_FLAG=False):# convert from numpy to torch variable 
    if USE_CUDA == False:
        data = torch.from_numpy(data.astype(np.float, copy=False)).type(torch.FloatTensor)
    else: # On GPU
        if TESTING_FLAG == False:
            data = torch.from_numpy(data.astype(np.float, copy=False)).type(torch.FloatTensor).cuda()
        else: # testing phase, no need to store the data on the GPU
            data = torch.from_numpy(data.astype(np.float, copy=False)).type(torch.FloatTensor).cuda()
            data.requires_grad = False
    return data


def prepare_data_helper(graphs):
    theta, s = [], [] # precision_mat, samples covariance mat
    for g_num in graphs:
        precision_mat, data = graphs[g_num] # data = M x N
        theta.append(precision_mat)
        s.append(np.matmul(data.T, data)/(args.M))
    theta = np.array(theta)
    s     = np.array(s)
    return [theta, s]


def prepare_data(mn):
    train_data = prepare_data_helper(mn.train_graphs)
#    valid_data = prepare_data_helper(mn.valid_graphs)
    test_data  = prepare_data_helper(mn.test_graphs)
    return train_data, test_data


def get_theta_pred(ll):
    return torch.matmul(ll, ll.transpose(-1, -2))

def my_cholesky_max(A, offset=args.cp): # formula taken from wiki
    off = convert_to_torch(np.array(offset), TESTING_FLAG=True)
    L = torch.zeros(args.N, args.N)
    if USE_CUDA:
        L = L.cuda()
    for i, (Ai, Li) in enumerate(zip(A, L)):# row wise i
        for j, Lj in enumerate(L[:i+1]): # for all rows j above i
            s = torch.sum(torch.Tensor([Li[k] * Lj[k] for k in range(j)]))
            if USE_CUDA:
                s = s.cuda()
#            print('e11: ', Ai[i], s, off)
            Li[j] = torch.sqrt(torch.max(Ai[i] - s, off)) if (i==j) else (1.0/ Lj[j] * (Ai[j]-s))
    return L 

def my_cholesky(A, offset=args.cp): # multiply by sign to keep positive
#    off = convert_to_torch(np.array(offset), TESTING_FLAG=True)
    L = torch.zeros(args.N, args.N)
    if USE_CUDA:
        L = L.cuda()
    for i, (Ai, Li) in enumerate(zip(A, L)):# row wise i
        for j, Lj in enumerate(L[:i+1]): # for all rows j above i
            #s = torch.sum(torch.Tensor([Li[k] * Lj[k] for k in range(j)]))
            #if USE_CUDA:
            #    s = s.cuda()
            s = torch.sum((Li * Lj)[:j]).detach()# for k in range(j))
#            print('e11: ', Ai[i], s, off)
#            Li[j] = torch.sqrt(torch.max(Ai[i] - s, off)) if (i==j) else (1.0/ Lj[j] * (Ai[j]-s))
            Li[j] = torch.sqrt(torch.sign(Ai[i] - s)*(Ai[i] - s)) if (i==j) else (1.0/ Lj[j] * (Ai[j]-s))
    return L 


def batch_cholesky(Ab): # TODO: Inplace?
    L = torch.zeros(Ab.shape)
    if USE_CUDA:
        L = L.cuda()
    for i, A in enumerate(Ab):
#        print('A = ', A)
        L[i, :, :] = my_cholesky(A)
    return L

def logdet_eig(A):
    return torch.sum(torch.log(torch.eig(A)[0][:, 0]))

def logdet_cholesky(A):
    ll = torch.cholesky(A)
    return 2*torch.sum(torch.log(ll.diag()))

def logdet_mycholesky(A):
    ll = my_cholesky(A)
    return 2*torch.sum(torch.log(ll.diag()))

def logdet_torch(A):
    #return torch.log(torch.det(A))
    return torch.logdet(A)

def get_logdet(A):
#    return logdet_eig(A)
#    return logdet_cholesky(A)
#    return logdet_mycholesky(A)
    return logdet_torch(A)



def get_duality_gap(theta, S, rho):
#    rho = torch.Tensor([args.rho])
#    if USE_CUDA:
#        rho = rho.cuda()
    U = torch.min(torch.max(torch.inverse(theta) - S, -1*rho), rho)
    #t1 = -1*torch.log(torch.det(S+U)) - args.N # term1 
    #t2 = -1*torch.log(torch.det(theta))
    t1 = -1*get_logdet(S+U) - args.N # term1 
    t2 = -1*get_logdet(theta)
    t3 = torch.trace(torch.matmul(S, theta))
    #t4 = args.rho*torch.max(torch.sum(torch.abs(theta), 0)) # L1 norm of mat is max abs column sum
    t4 = rho*torch.sum(torch.abs(theta)) # L1 norm of mat
#    print('DUALITY: ', t1, t2, t3, t4)
    return t1+t2+t3+t4


def batch_duality_gap(Ab, Sb, rho):
    v = 0
    for i, (A, S) in enumerate(zip(Ab, Sb)):
#        print('duality call = ', i)
        v += get_duality_gap(A, S, rho)
    return v/len(Ab)


def get_step_size(theta_pred, theta_prev):
    num = torch.trace(torch.matmul(theta_pred-theta_prev, theta_pred-theta_prev))
    den = torch.trace(torch.matmul(theta_pred-theta_prev, torch.inverse(theta_prev)-torch.inverse(theta_pred)))
#    step_size = num/den
    step_size = torch.exp(torch.log(num)-torch.log(den))
    zero = torch.Tensor([0]).type(step_size.type())
#    print('STEP size: ', num, den, step_size)
#    return torch.max(step_size, zero)
 
    if den !=0 :
#        #step_size[step_size!=step_size] = 0 # taking care of the NaN condition
        return torch.max(step_size, zero)
        #return step_size
    else:
        print('den =', den, ' :resetting the step size, possible fix: DECREASE the c_init_step value!')
        return torch.Tensor([0]).type(num.type()) # return 0


def batch_step_size_avg(theta_predb, theta_prevb):# returns the average over batch
    s = 0
    for i, (theta_pred, theta_prev) in enumerate(zip(theta_predb, theta_prevb)):
        s += get_step_size(theta_pred, theta_prev)
    return s/len(theta_predb)


def batch_step_size(theta_predb, theta_prevb):# returns the min over batch
    s = []
    for i, (theta_pred, theta_prev) in enumerate(zip(theta_predb, theta_prevb)):
        s.append(get_step_size(theta_pred, theta_prev))
    return min(s)


def get_min_eigval(theta):
    return torch.min(torch.eig(theta)[0][:, 0])

def get_init_step_size(theta_init):
    s = []
    for theta in theta_init:
        s.append(get_min_eigval(theta))
    return min(s)**2 #step_size


torch_sqrtm = MatrixSquareRoot.apply

def batch_matrix_sqrt(A):
    # A should be PSD
    n = A.shape[0]
    sqrtm_torch = torch.zeros(A.shape).type_as(A)
    for i in range(n):
        sqrtm_torch[i] = torch_sqrtm(A[i])
    return sqrtm_torch


def get_frobenius_norm(A):
    #return torch.mean(torch.sum(A**2, (1,2))) 
    return torch.sum(A**2) 

def get_convergence_loss(theta_pred, theta_true):
    num = get_frobenius_norm(theta_pred - theta_true)
    den = get_frobenius_norm(theta_true)
    return 10*torch.log10(num/den).data.cpu().numpy()

def get_obj_val(theta, S, rho):
    t1 = 0.5*get_f_theta(theta, S)
    t2 = rho*torch.sum(torch.abs(theta))
    #print('t1, t2', t1, t2)
    return (t1+t2).data.cpu().numpy()

def get_f_theta(theta, S):
    t1 = -1*get_logdet(theta)
#    print('err: ', S, theta)
    t2 = torch.trace(torch.matmul(S, theta))
    return t1 + t2

def glasso_predict(data, COLLECT=True):
    print('Running ADMM: augmented lagrangian')
    # predict as a complete batch?
    criterion = nn.MSELoss()# input, target
    m_sig = nn.Sigmoid()
    criterionBCE = nn.BCELoss()
    theta, S = data
    #theta, S = theta[0], S[0]
    # theta -> K_train x N x N (Matrix)
    # S -> K_train x N x N (observed vector)
    # train using ALISTA style training.
    lambda_f = args.lambda_init
    rho_l1 = args.rho
    alpha_lr = args.alpha_lr

    theta_true = convert_to_torch(theta, TESTING_FLAG=True)
    S = convert_to_torch(S, TESTING_FLAG=True)
   
    if args.INIT_DIAG == 1:
        print(' extract batchwise diagonals, add offset and take inverse')
        batch_diags = 1/(torch.diagonal(S, offset=0, dim1=-2, dim2=-1) + args.theta_init_offset)
        theta_init = torch.diag_embed(batch_diags)
    else:
        print('***************** (S+theta_offset*I)^-1 is used')
        theta_init = torch.inverse(S+args.theta_init_offset * torch.eye(args.N).expand_as(S).type_as(S))
    zero = torch.Tensor([0])#.type(self.dtype)
    if USE_CUDA == True:
        zero = zero.cuda()
    
    # batch size is fixed for testing as 1
    num_batches = 1 #int(len(theta_true)/args.batch_size)
#    print('num batches: ', num_batches)
    epoch_loss = []
    mse_binary_loss = []
    bce_loss = []
    frob_loss = []
    duality_gap = []
    ans = []
    #for batch_num in range(num_batches): # processing batchwise
        # Get a batch
        #ll = my_cholesky(theta_init[ridx][0])#(theta_pred) # lower triangular 
    theta_pred = theta_init#[batch_num*args.batch_size: (batch_num+1)*args.batch_size] #(theta_pred) # lower triangular



    Sb = S#[batch_num*args.batch_size: (batch_num+1)*args.batch_size]#[0]
    U = torch.zeros(Sb.shape).type(Sb.type())
#    print('sb type: ', Sb.dtype, Sb.type(), ' U:', U.dtype, U.type())

    identity_mat = torch.eye(Sb.shape[-1]).expand_as(Sb)
    #print('err: ', identity_mat, identity_mat.shape, Sb.shape, S.shape)
    if USE_CUDA == True:
        identity_mat = identity_mat.cuda()
#    print('ITR, conv.loss, obj_val_pred, obj_val_true', theta_pred)
    res_conv = []
    obj_true = get_obj_val(theta_true, S, rho_l1)
    for k in range(args.L):
        start = time.time()
        if COLLECT:
            theta_pred_diag = torch.diag_embed(torch.diagonal(theta_pred, offset=0, dim1=-2, dim2=-1))
            theta_true_diag = torch.diag_embed(torch.diagonal(theta_true, offset=0, dim1=-2, dim2=-1))
            cv_loss, cv_loss_off_diag, obj_pred = get_convergence_loss(theta_pred, theta_true), get_convergence_loss(theta_pred-theta_pred_diag, theta_true-theta_true_diag), get_obj_val(theta_pred, S, rho_l1)
#            print(k, '%.3f %.3f %.3f %.3f' %(cv_loss, obj_pred, obj_true_rho, obj_true_orig))
            res_conv.append([cv_loss, obj_pred, obj_true, cv_loss_off_diag])
        #print(k, '%.3f %.3f %.3f' %(get_convergence_loss(theta_pred, theta_true), get_obj_val(theta_pred, S, rho_l1), get_obj_val(theta_true, S, rho_l1)))
#       print('itr = ', itr, theta_pred)#, theta_true[ridx])
        # step 1 : ADMM
        b = 1.0/lambda_f * Sb - theta_pred + U
        b2_4ac = torch.matmul(b.transpose(-1, -2), b) + 4.0/lambda_f * identity_mat
        sqrt_term = torch_sqrtm(b2_4ac)#batch_matrix_sqrt(b2_4ac)
        theta_k1 = 1.0/2*(-1*b+sqrt_term)
        # step 2 : ADMM
#        theta_pred = model.eta_forward(theta_k1, k)
        theta_pred = torch.sign(theta_k1+U)*torch.max(zero, torch.abs(theta_k1+U)-rho_l1/lambda_f)#Z_k1
        # step 3: ADMM
        #lambda_f = lambda_f + alpha_lr * 0.5*get_frobenius_norm(theta_pred-theta_k1)
        U = U + theta_k1 - theta_pred

    print('fdr, tpr, fpr, shd, nnz, nnz/theta_pred.size, nnz_true,nnz_true/theta_true.size, ps, np.linalg.cond(theta_pred), np.linalg.cond(theta_true)')
    theta_pred = theta_pred.data.cpu().numpy()
    theta_true = theta_true.data.cpu().numpy()
#    print('ADMM: ', theta_pred, ' condition number = ', np.linalg.cond(theta_pred))#, ' nnz = ', np.count_nonzero(theta_pred), ' nnz% = ', np.count_nonzero(theta_pred)/theta_pred.size)
#    print('true: ', theta_true, ' condition_number = ', np.linalg.cond(theta_true))#, ' nnz = ', np.count_nonzero(theta_true), ' nnz% = ', np.count_nonzero(theta_true)/theta_true.size)
    fdr, tpr, fpr, shd, nnz, nnz_true, ps = metrics.report_metrics(theta_true, theta_pred)
    cond_theta_pred, cond_theta_true = np.linalg.cond(theta_pred), np.linalg.cond(theta_true)
    print(fdr, tpr, fpr, shd, nnz, nnz/theta_pred.size, nnz_true,nnz_true/theta_true.size, ps, cond_theta_pred, cond_theta_true)
    return [fdr, tpr, fpr, shd, nnz, nnz_true, ps, cond_theta_pred, cond_theta_true], res_conv

def isPSD(A, tol=1e-8):
    E,V = scipy.linalg.eigh(A)
    return np.all(E > -tol)

def check_sym(a, tol=1e-7):
    print('is PSD: ', isPSD(a))
    return np.allclose(a, a.T, atol=tol)

def main():
    print('creating the graph data for GLASSO ADMM ')
    # M = Samples, N = features :  out_data = M x N
    if args.EDGE_RECOVERY == 1:
#        mn = create_MN_edge_recovery(args.K_train, args.M, args.N, args.graph_type, args.SAMPLE_BATCHES, args.w, args.K_test)
#        mn = create_MN_vary_w(args.K_train, args.M, args.N, args.graph_type, args.SAMPLE_BATCHES, [args.w_min, args.w_max], args.K_test, args.K_valid)
        mn = create_MN_vary_w(args.K_train, args.M, args.N, args.graph_type, args.SAMPLE_BATCHES, [args.w_min, args.w_max], args.K_test, [args.prob, args.MAX_DEG, args.SIGNS], args.K_valid)
#        mn = create_MN_random(args.K_train, args.M, args.N, args.graph_type, args.SAMPLE_BATCHES, [args.p_min, args.p_max], args.K_test, args.K_valid)
    elif args.varyS == 1:
        mn = create_MN_varyS(args.K_train, args.M, args.N, [0.03, 0.1, 0.2], args.K_valid, args.K_test)
    else: # create graphs with varying sparsity
        mn = create_MN(args.K_train, args.M, args.N, args.prob, args.K_valid, args.K_test)

    train_data, test_data = prepare_data(mn)
    
#    print('****Train Data****')
#    glasso_predict(train_data)
#    print('****Valid Data****')
#    glasso_predict(valid_data)
    # Collect the results for test_data
    res_str = [] # structure learning metrics  
    res_conv_loss = {}
    print('****Test Data****')
#    true_theta = test_data[0][0]
    for i, data in enumerate(zip(test_data[0], test_data[1])):
#    for i, data in enumerate(zip(test_data[0], test_data[1])):
        print('test graph ', i)
#        res_str[i]  = []
#        res_conv[i] = []
#        print(i, data)
        #str_metric, res_conv_loss[i] = glasso_predict([true_theta, data])
        str_metric, res_conv_loss[i] = glasso_predict(data, COLLECT=False)
        res_str.append(str_metric)
    print('Optimization done, running analysis')
    # get the average results for analysis
    res_mean = np.mean(np.array(res_str), 0)
    res_std  = np.std(np.array(res_str), 0)
    res_mean = ["%.3f" %x for x in res_mean]
    res_std  = ["%.3f" %x for x in res_std]
    print('fdr, tpr, fpr, shd, nnz, nnz_true, ps, np.linalg.cond(theta_pred), np.linalg.cond(theta_true)')
    print(*sum(list(map(list, zip(res_mean, res_std))), []), sep=', ')
    # print the convergence loss analysis
    print('ITR, conv_loss, obj_val_pred, obj_val_true, conv_loss_off_diag')
    res_conv = []
    early_terminate = {}
    for itr in range(args.L):
        res_conv = []
        early_terminate[itr] = set()
        for k, v in res_conv_loss.items():
            if itr >= len(v):
                early_terminate[itr].add(k)
                #print('Some test graphs already converged at itr =',k, itr)
                continue
            res_conv.append(v[itr])
        res_conv = np.array(res_conv)
        if len(res_conv) == 0:
            print('all graphs converged by iteration = ', itr)
            break
        mean_vec = ["%.3f" %x for x in np.mean(res_conv, 0)]
        std_vec = ["%.3f" %x for x in np.std(res_conv, 0)]
        print(itr, *sum(list(map(list, zip(mean_vec, std_vec))), []), sep=', ')
    print('early optimization details')
    e=0
    for k, v in early_terminate.items():
        if len(v) > e :
            print('itr ', k, '# of early terminated graphs ', len(v))
            e = len(v)
#    pprint.pprint(early_terminate)

#    print('****Test Data****')
#    glasso_predict(test_data)
    return 

if __name__=="__main__":
    main()
