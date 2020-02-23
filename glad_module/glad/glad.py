#! /usr/bin/python -u
import argparse, random
from expts_glad.create_GGM import create_MN_vary_w

from expts_glad.glad.model import threshold_NN_lambda_single_model
import torch, scipy, copy
import networkx as nx
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau
import numpy as np 
from scipy.linalg import sqrtm
from expts_glad import metrics
from expts_glad.torch_sqrtm import MatrixSquareRoot
import matplotlib.pyplot as plt
import time

TRAIN = True #False# True
ONLY_ECOLI=0

parser = argparse.ArgumentParser(description='Structure learning of graphical model using glad')
parser.add_argument('--K_train', type=int, default=2, #1000,
                    help='Num of training examples for a fixed D')
parser.add_argument('--K_valid', type=int, default=0, #1000,
                    help='Number of valid examples for a fixed D ')
parser.add_argument('--K_test', type=int, default=100,
                    help='Number of testing examples for a fixed D')
parser.add_argument('--M', type=int, default= 4, #1000, #500, #6,#500,
                    help='number of samples M, R^(NxM)')
parser.add_argument('--SAMPLE_BATCHES', type=int, default=5, #1000, #500, #6,#500,
                    help='number of batches of sample size M')
parser.add_argument('--N', type=int, default= 5, #250, #3,#250,
                    help='Number of nodes N, R^(NxM)')
parser.add_argument('--prob', type=float, default=0.05,
                    help='sparsity = 2*prob: probability for the erdos-renyi true graph')
parser.add_argument('--MAX_DEG', type=int, default= 50, #250, #3,#250,
                    help='Max degree of the random erdos-renyi graph')
parser.add_argument('--SIGNS', type=int, default= 0, #250, #3,#250,
                    help='1: create random graphs with postive and negative signs')
parser.add_argument('--w_min', type=float, default=-1,
                    help='the weights of values in true precision matrices')
parser.add_argument('--w_max', type=float, default=1,
                    help='the weights of values in true precision matrices')
parser.add_argument('--p_min', type=float, default=0.12,
                    help='the weights of values in true precision matrices')
parser.add_argument('--p_max', type=float, default=0.25,
                    help='the weights of values in true precision matrices')
parser.add_argument('--rho_init', type=float, default=1, #075,
                    help='penalty term for L1(theta)')
parser.add_argument('--lambda_init', type=float, default=0.6,
                    help='initial step sizes for unrolled architecture')
parser.add_argument('--gamma_init', type=float, default=1e-2,
                    help='min eigenvalue correction term ')
parser.add_argument('--theta_init_offset', type=float, default=0.1, #0.03, #075,
                    help='offset for setting the diagonal init theta approximation')
parser.add_argument('--L', type=int, default=20,
                    help='Unroll the network L times')
parser.add_argument('--init_lr', type=float, default=0.001,
                    help='learning rate of phi')
parser.add_argument('--cp', type=float, default=0.2,
                    help='penalty for cholesky decomposition')
parser.add_argument('--INIT_DIAG', type=int, default=0,
                    help='1 : initialize the theta0 diagonally')
parser.add_argument('--USE_CUDA_FLAG', type=int, default=1,
                    help='USE GPU if = 1')
parser.add_argument('--use_optimizer', type=str,  default='adam',
                    help='can use either: adam, adadelta, rms, sgd')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size used for training')
parser.add_argument('--train_epochs', type=int, default=40,
                    help='number of epochs to be used for training')
parser.add_argument('--varyS', type=int, default=0,
                    help='experiment to do varyS')
parser.add_argument('--lossBCE', type=int, default=0,
                    help='change the loss to binary')
parser.add_argument('--lossL1', type=int, default=0,
                    help='change the loss L1 loss')

parser.add_argument('--loss_signed', type=int, default=0,
                    help='change the loss to signed frobenius norm')
parser.add_argument('--cost_sen_wt', type=int, default=1,
                    help='cost sensitive learning weight for BCE loss')
parser.add_argument('--EDGE_RECOVERY', type=int, default=0,
                    help='experiment to get the nmse plots')
parser.add_argument('--graph_type', type=str, default='random_maxd',
                    help='grid/chain/star/random_maxd')
parser.add_argument('--MODEL_type', type=str, default='th_NN',
                    help='th or th_NN (neural net threshold)')
parser.add_argument('--nF', type=int, default=2,
                    help='number of input features for NN rho')
parser.add_argument('--H', type=int, default=5,
                    help='Hidden layer size of NN rho')


args = parser.parse_args()

# Global Variables
USE_CUDA = False
if args.USE_CUDA_FLAG == 1:
    USE_CUDA = True


def convert_to_torch(data, TESTING_FLAG=False, USE_CUDA=USE_CUDA):# convert from numpy to torch variable 
    if USE_CUDA == False:
        data = torch.from_numpy(data.astype(np.float, copy=False)).type(torch.FloatTensor)
        if TESTING_FLAG==True:
            data.requires_grad = False
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
        # check whether the diagonals are all positive
        if np.all(s[-1].diagonal() > 0) == False:
            print('Diagonals of emp cov matrix are negative: CHECK', s, s[-1].diagonal())
#        else:
#            print('Diagonals of emp cov matrix are positive:', s[-1].diagonal())

    theta = np.array(theta)
    s     = np.array(s)
    return [theta, s]


def prepare_data(mn):
    train_data = prepare_data_helper(mn.train_graphs)
    valid_data = []
    if args.K_valid > 0:
        valid_data = prepare_data_helper(mn.valid_graphs)
    test_data  = prepare_data_helper(mn.test_graphs)
    return train_data, valid_data, test_data


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

def train_glasso(data, valid_data=[]):# tied lista
#    torch.set_grad_enabled(True)
    print('training GLASSO')
    theta, S = data
#    theta = theta[0]
    if len(valid_data)>0:
        valid_theta, valid_S = valid_data
#        valid_theta = valid_theta[0]
        valid_theta_true = convert_to_torch(valid_theta, TESTING_FLAG=True)
        valid_S = convert_to_torch(valid_S, TESTING_FLAG=True)
    # theta -> K_train x N x N (Matrix)
    # S -> K_train x N x N (observed vector)
    # train using ALISTA style training.
    if args.MODEL_type == 'th':
        model = threshold_model(args.L, args.rho_init, args.lambda_init, args.theta_init_offset, args.gamma_init, USE_CUDA=USE_CUDA)
    elif args.MODEL_type == 'th_NN':
        model = threshold_NN_lambda_single_model(args.L, args.rho_init, args.lambda_init, args.theta_init_offset, args.gamma_init, args.N, args.nF, args.H, USE_CUDA=USE_CUDA)
        #model = threshold_NN_lambda_unrolled_model(args.L, args.rho_init, args.lambda_init, args.theta_init_offset, args.gamma_init, args.N, args.nF, args.H, USE_CUDA=USE_CUDA)
#    model.train() 
    theta_true = convert_to_torch(theta, TESTING_FLAG=True)
    S = convert_to_torch(S, TESTING_FLAG=True)

    zero = torch.Tensor([0])#.type(self.dtype)

#    print('check: theta ', theta_init.shape)
#    print('true: ', theta_true)
    print('parameters to be learned')
    for name, param in model.named_parameters():
        print(name, param.shape, param.requires_grad)
    dtype = torch.FloatTensor
    if USE_CUDA == True:
        model = model.cuda()
        zero = zero.cuda()
        dtype = torch.cuda.FloatTensor

    lr = args.init_lr
    if args.use_optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, rho=0.9, eps=1e-06, weight_decay=0)# LR range = 5 -> 
    elif args.use_optimizer == 'rms':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.25, centered=False)
    elif args.use_optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,  dampening=0, weight_decay=0, nesterov=False)
    elif args.use_optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    else:
        print('Optimizer not found!')
    #scheduler = MultiStepLR(optimizer, milestones=[1000], gamma=0.5)
    scheduler = MultiStepLR(optimizer, milestones=[10, 15, 20, 25, 100, 200], gamma=0.25)
    #scheduler = MultiStepLR(optimizer, milestones=[10, 15, 20, 100, 200], gamma=0.25)
    #scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30, 200], gamma=0.5)
    criterion = nn.MSELoss()# input, target
    criterion_L1 = nn.L1Loss()
    m_sig = nn.Sigmoid()
    criterionBCE = nn.BCELoss()
    
    # batch size is fixed
#    num_batches = int(args.K_train/args.batch_size)
#    if args.SAMPLE_BATCHES > 0:
#        num_batches = int(args.K_train*args.SAMPLE_BATCHES/args.batch_size)

    num_batches = int(len(S)/args.batch_size)
 
#    if args.K_train >= 10:
#        args.batch_size = 10
    #best_shd_model = model
    best_valid_shd, best_valid_ps, best_valid_nmse = np.inf, -1*np.inf, np.inf
    EARLY_STOP=False
    for epoch in range(args.train_epochs):# 1 epoch is expected to go through complete data
        scheduler.step()
#        if epoch%1==0:
#            for param_group in optimizer.param_groups:
#                print('epoch: ', epoch, ' lr ', param_group['lr'])
        epoch_loss = []
        frob_loss = []
        duality_gap = []
        mse_binary_loss = []
        bce_loss = []
        if EARLY_STOP:
            break
#        print('ecpohc ', epoch)
        for batch_num in range(num_batches): # processing batchwise
            optimizer.zero_grad()
            # resetting the loss to zero
            loss = torch.Tensor([0]).type(dtype)
            # Get a batch
            #ridx = random.sample(list(range(args.K_train)), args.batch_size)
            ridx = random.sample(list(range(len(S))), args.batch_size)
            Sb = S[ridx]#[0]
#            print('errr train check : ', batch_num, theta_true, Sb, theta_true.expand_as(Sb))

            if args.INIT_DIAG == 1:
                #print(' extract batchwise diagonals, add offset and take inverse')
                batch_diags = 1/(torch.diagonal(Sb, offset=0, dim1=-2, dim2=-1) + model.theta_init_offset)
                theta_init = torch.diag_embed(batch_diags)
            else:
                #print('***************** (S+theta_offset*I)^-1 is used')
                theta_init = torch.inverse(Sb+model.theta_init_offset * torch.eye(Sb.shape[-1]).expand_as(Sb).type_as(Sb))
            #theta_pred = S_inv[r_idx]
            #ll = torch.cholesky(theta_init[ridx])#(theta_pred) # lower triangular  
            #ll = my_cholesky(theta_init[ridx][0])#(theta_pred) # lower triangular 
            #ll = batch_cholesky(theta_init[ridx])#(theta_pred) # lower triangular 
            theta_pred = theta_init#[ridx]
            #theta_pred = theta_init[ridx]
            #theta_pred.requires_grad = True
            #Sb = S[ridx][0]
            #step_size = get_init_step_size(theta_init[ridx])
            identity_mat = torch.eye(Sb.shape[-1]).expand_as(Sb)
            if USE_CUDA == True:
                identity_mat = identity_mat.cuda()
            #print('ERRR check: ', theta_pred.shape, get_frobenius_norm(theta_pred), get_frobenius_norm(theta_pred).shape)
            #lambda_k = model.lambda_f(get_frobenius_norm(theta_pred))
            lambda_k = model.lambda_forward(zero + args.lambda_init, zero,  k=0)
            for k in range(args.L):
#                print('itr = ', itr, theta_pred)#, theta_true[ridx])
                # step 1 : AM
                b = 1.0/lambda_k * Sb - theta_pred
                b2_4ac = torch.matmul(b.transpose(-1, -2), b) + 4.0/lambda_k * identity_mat
                sqrt_term = batch_matrix_sqrt(b2_4ac)
                theta_k1 = 1.0/2*(-1*b+sqrt_term)
                """
                # extract the diagonals of the matrices
                theta_diag = torch.diag_embed(torch.diagonal(theta_k1, offset=0, dim1=-2, dim2=-1))
                # soft threshold on remaining entries 
                theta_pred = model.eta_forward(theta_k1-theta_diag, k)
                # add the diagonals
                theta_pred = theta_pred + theta_diag
                """
                # softthresholding on all the entries
                #theta_pred = model.eta_forward(theta_k1, k)
                
                if args.MODEL_type == 'th':
                    # soft thresholding + eigenvalue correctness term
                    theta_pred = model.eta_forward(theta_k1, k) + torch.max(model.gamma_c[k], zero+1e-2) * identity_mat 
                elif args.MODEL_type == 'th_NN':
                    theta_pred = model.eta_forward(theta_k1, Sb, k, theta_pred) # 
                # update the lambda
                lambda_k = model.lambda_forward(torch.Tensor([get_frobenius_norm(theta_pred-theta_k1)]).type(dtype), lambda_k, k)
                # accumulating loss
#                print('k= ', k, ' lambda_value ', lambda_k, get_frobenius_norm(theta_pred-theta_k1))
                
                loss += criterion(theta_pred, theta_true[ridx])/args.L
                #loss += criterion(theta_pred, theta_true.expand_as(theta_pred))/args.L

#            print('k= ', k, ' lambda_value ', lambda_k)
            #print('thetapred: ', theta_pred, check_sym(theta_pred[0].data.cpu().numpy()))
            #delta = batch_duality_gap(theta_pred, Sb, model.rho)
            # NOTE: ******* IMP: Change thissss!@!! ****************************
            delta = torch.ones([1])*-1


            loss += criterion(theta_pred, theta_true[ridx])/args.L

            #loss += criterion(theta_pred, theta_true.expand_as(theta_pred))/args.L
            
            #lossf = get_frobenius_norm(theta_pred - theta_true[ridx])
            #total_loss = loss #+ delta
            #total_loss = lossB #+ loss #+ delta
            #total_loss = loss + lossB+ delta + lossBCE
#            total_loss = lossBCE
            #total_loss = delta
            if args.lossBCE == 1:# binary cross entropy
                total_loss = lossBCE
            elif args.loss_signed == 1:# signed loss
                #total_loss = criterion(torch.sign(theta_pred), torch.sign(theta_true.expand_as(theta_pred)))
                total_loss = criterion(theta_pred, torch.sign(theta_true.expand_as(theta_pred)))
            elif args.lossL1 == 1:# signed loss
                total_loss = criterion_L1(theta_pred, theta_true.expand_as(theta_pred))
            else:# frobenius norm
                total_loss = loss
#            total_loss.requires_grad = True
#            print('err: ', total_loss, total_loss.requires_grad)
 
            lv = loss.data.cpu().numpy() 
            if lv <= 1e-7:# loss value
                print('Early stopping as loss = ', lv)
                EARLY_STOP = True
                break

            total_loss.backward()
            #delta.backward()

#            for name, param in model.named_parameters():
#                print('befoer: ', name, param)
            optimizer.step()

#            for name, param in model.named_parameters():
#                print('after: ', name, param)

#            mse_binary_loss.append(lossB.data.cpu().numpy())
#            bce_loss.append(lossBCE.data.cpu().numpy())
#            duality_gap.append(delta.data.cpu().numpy())
#            frob_loss.append(lossf.data.cpu().numpy())
            epoch_loss.append(loss.data.cpu().numpy())
        if epoch % 1 == 0 and EARLY_STOP==False:
#            print('loss_summary: MSE: ', sum(epoch_loss)/len(epoch_loss), ' Mean Frobenius loss: ',sum(frob_loss)/len(frob_loss), ' MSE_binary loss: ', sum(mse_binary_loss)/len(mse_binary_loss), 'BCE_loss: ', sum(bce_loss)/len(bce_loss), ' duality gap = ', sum(duality_gap)/len(duality_gap)) 
            if args.lossBCE == 1: 
                print(epoch, sum(epoch_loss)/len(epoch_loss), sum(bce_loss)/len(bce_loss))
            else:
                print('loss_values: ', epoch, sum(epoch_loss)/len(epoch_loss))#, sum(duality_gap)/len(duality_gap))
        # Checking the results on valid data and updating the best model
        if len(valid_data)>0:
            # get the SHD on the valid data and the train data
            #curr_valid_shd, curr_valid_nmse = glasso_predict(model, valid_data)
            curr_valid_shd, curr_valid_ps, curr_valid_nmse = glasso_predict(model, valid_data)
            curr_train_shd, curr_train_ps, curr_train_nmse = glasso_predict(model, data)
            print('valid/train: shd %0.2f/%0.2f ps %0.2f/%0.2f nmse %0.2f/%0.2f' %(curr_valid_shd, curr_train_shd, curr_valid_ps, curr_train_ps, curr_valid_nmse, curr_train_nmse))
#            if curr_valid_shd <= best_valid_shd:
            if curr_valid_ps >= best_valid_ps:
                print('epoch = ', epoch, ' Updating the best ps model with valid ps = ', curr_valid_ps)
                best_ps_model = copy.deepcopy(model)
                best_valid_ps = curr_valid_ps

            if curr_valid_shd <= best_valid_shd:
                print('epoch = ', epoch, ' Updating the best shd model with valid shd = ', curr_valid_shd)
                best_shd_model = copy.deepcopy(model)
                best_valid_shd = curr_valid_shd

            if curr_valid_nmse <= best_valid_nmse:
                print('epoch = ', epoch, ' Updating the best nmse model with valid nmse = ', curr_valid_nmse)
                best_nmse_model = copy.deepcopy(model)
                best_valid_nmse = curr_valid_nmse
            model.train()
            #print('loss_summary:: epoch: ', epoch, ' loss: ', sum(epoch_loss)/len(epoch_loss))#, ' NMSE loss: ', 10*np.log10( (np.sum(np.array(epoch_loss)))/(len(epoch_loss)*E_norm_xtrue)))
#    print('ans: ', theta_pred)
#    print('true: ', theta_true)
    for name, param in model.named_parameters():
        print(name, param)
    #return best_ps_model # model
    return best_nmse_model, best_shd_model, best_ps_model # model


def get_frobenius_norm(A, single=False):
    if single:
        return torch.sum(A**2)
    return torch.mean(torch.sum(A**2, (1,2)))

def get_convergence_loss(theta_pred, theta_true):
    num = get_frobenius_norm(theta_pred - theta_true, single=True)
    den = get_frobenius_norm(theta_true, single=True)
    #print('n d ', num, den)
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


def glasso_predict(model, data, flagP=False, SAVE_GRAPH=False, eM=0, name='', mn=''):
    with torch.no_grad(): 
        print('Running unrolled ADMM predict')
        # predict as a complete batch?
        model.eval()
        criterion = nn.MSELoss()# input, target
        m_sig = nn.Sigmoid()
        criterionBCE = nn.BCELoss()
        theta, S = data
    #    theta = theta[0] 
        # theta -> K_train x N x N (Matrix)
        # S -> K_train x N x N (observed vector)
        #theta_true = convert_to_torch(theta, TESTING_FLAG=True, USE_CUDA=False)
        theta_true = convert_to_torch(theta, TESTING_FLAG=True, USE_CUDA=True)
        S = convert_to_torch(S, TESTING_FLAG=True, USE_CUDA=False)
       
        zero = torch.Tensor([0])#.type(self.dtype)
        dtype = torch.FloatTensor
        if USE_CUDA == True:
            zero = zero.cuda()
            model = model.cuda()
            dtype = torch.cuda.FloatTensor
        
        # batch size is fixed for testing as 1
        batch_size = 1
        print('CEHCKK: Total graphs = ', len(S)) 
        num_batches = int(len(S)/batch_size)
    #    print('num batches: ', num_batches)
        epoch_loss = []
        mse_binary_loss = []
        bce_loss = []
        frob_loss = []
        duality_gap = []
        ans = []
        if flagP:
            res_conv = {}
            for k in range(args.L+1):
                res_conv[k] = []
    #        print('ITR, conv.loss, obj_val_pred, obj_val_true_model_rho, obj_val_pred_args_rho')#, theta_pred)

        res = []
        for batch_num in range(num_batches): # processing batchwise
            # Get a batch
            #ll = my_cholesky(theta_init[ridx][0])#(theta_pred) # lower triangular 
            #theta_pred = theta_init[batch_num*batch_size: (batch_num+1)*batch_size] #(theta_pred) # lower triangular
            #theta_true_b = theta_true[batch_num*batch_size: (batch_num+1)*batch_size]
            theta_true_b = theta_true[batch_num*batch_size: (batch_num+1)*batch_size]
            Sb = S[batch_num*batch_size: (batch_num+1)*batch_size]#[0]
            identity_mat = torch.eye(Sb.shape[-1]).expand_as(Sb)
            if USE_CUDA == True:
                identity_mat = identity_mat.cuda()
                Sb = Sb.cuda()
            #    theta_pred = theta_pred.cuda()
            #    theta_true_b = theta_true_b.cuda()

            if args.INIT_DIAG == 1:
                #print(' extract batchwise diagonals, add offset and take inverse')
                batch_diags = 1/(torch.diagonal(Sb, offset=0, dim1=-2, dim2=-1) + model.theta_init_offset)
                theta_pred = torch.diag_embed(batch_diags)
            else:
                #print('***************** (S+theta_offset*I)^-1 is used')
                theta_pred = torch.inverse(Sb+model.theta_init_offset * torch.eye(Sb.shape[-1]).expand_as(Sb).type_as(Sb))

            #lambda_k = model.lambda_f(get_frobenius_norm(theta_pred))
            lambda_k = model.lambda_forward(zero + args.lambda_init, zero, k=0)

    #        if flagP:
    #            print('ITR, conv.loss, obj_val_pred, obj_val_true_model_rho, obj_val_pred_args_rho')#, theta_pred)
            for k in range(args.L):
    #            start = time.time()
                if flagP:
                    theta_pred_diag = torch.diag_embed(torch.diagonal(theta_pred[0], offset=0, dim1=-2, dim2=-1))
                    #theta_true_b_diag = torch.diag_embed(torch.diagonal(theta_true_b[0], offset=0, dim1=-2, dim2=-1))
                    theta_true_b_diag = torch.diag_embed(torch.diagonal(theta_true_b, offset=0, dim1=-2, dim2=-1))
                    if args.MODEL_type == 'th':
                        cv_loss, cv_loss_off_diag, obj_pred, obj_true_rho, obj_true_orig = get_convergence_loss(theta_pred[0], theta_true_b), get_convergence_loss(theta_pred[0]-theta_pred_diag, theta_true_b-theta_true_b_diag), get_obj_val(theta_pred[0], Sb[0], model.rho_l1[k]), get_obj_val(theta_true_b, Sb[0], model.rho_l1[k]), get_obj_val(theta_true_b, Sb[0], args.rho_init)
                        res_conv[k].append([cv_loss, obj_pred, obj_true_rho, obj_true_orig, cv_loss_off_diag])
                    elif args.MODEL_type == 'th_NN':
                        cv_loss, cv_loss_off_diag = get_convergence_loss(theta_pred[0], theta_true_b), -1 #get_convergence_loss(theta_pred[0]-theta_pred_diag, theta_true_b-theta_true_b_diag)
                        res_conv[k].append([cv_loss, cv_loss_off_diag])
                # step 1 : AM
                b = 1.0/lambda_k * Sb - theta_pred
                b2_4ac = torch.matmul(b.transpose(-1, -2), b) + 4.0/lambda_k * identity_mat
                sqrt_term = batch_matrix_sqrt(b2_4ac)
                theta_k1 = 1.0/2*(-1*b+sqrt_term)

                # step 2 : AM  
                """ 
                # extract the diagonals of the matrices
                theta_diag = torch.diag_embed(torch.diagonal(theta_k1, offset=0, dim1=-2, dim2=-1))
                # soft threshold on remaining entries 
                theta_pred = model.eta_forward(theta_k1-theta_diag, k)
                # add the diagonals
                theta_pred = theta_pred + theta_diag
                """
                # soft thresholding on all the entries
                #theta_pred = model.eta_forward(theta_k1, k)
                if args.MODEL_type == 'th':
                    # soft thresholding + eigenvalue correctness term
                    theta_pred = model.eta_forward(theta_k1, k) + torch.max(model.gamma_c[k], zero+1e-2) * identity_mat 
                elif args.MODEL_type == 'th_NN':
                    #theta_pred = model.eta_forward(theta_k1, Sb, k) + torch.max(model.gamma_c[k], zero+1e-2) * identity_mat 
                    #theta_pred = model.eta_forward(theta_k1, Sb, k, theta_pred) + torch.max(model.gamma_c[k], zero+1e-2) * identity_mat 
                    theta_pred = model.eta_forward(theta_k1, Sb, k, theta_pred) #+ torch.max(model.gamma_c[k], zero+1e-2) * identity_mat 
                    
                # updating lambda 
                lambda_k = model.lambda_forward(torch.Tensor([get_frobenius_norm(theta_pred-theta_k1)]).type(dtype), lambda_k, k)
                #lambda_k = model.lambda_f(get_frobenius_norm(theta_pred-theta_k1))
    #            print('k= ', k, ' lambda_value ', lambda_k, get_frobenius_norm(theta_pred-theta_k1))
    #            stop = time.time()
    #            print('Walltimes: ', k, stop-start)
    #        br
            if flagP:
                theta_pred_diag = torch.diag_embed(torch.diagonal(theta_pred[0], offset=0, dim1=-2, dim2=-1))
                # Getting the final predicted convergence loss
                if torch.min(torch.eig(theta_pred[0])[0][:, 0]) == 0:
                    adjust_eval_identity = torch.eye(theta_pred.shape[-1]).expand_as(theta_pred[0]).type_as(theta_pred[0])
                    print('Adjusting the minimum eigenvalue to 1, SHOULD NOT BE CALLED AFTER THE GAMMA ADDITION!')
                    theta_pred[0] += adjust_eval_identity # change the eigenval to 1
                if args.MODEL_type == 'th':
                    cv_loss, cv_loss_off_diag, obj_pred, obj_true_rho, obj_true_orig = get_convergence_loss(theta_pred[0], theta_true_b), get_convergence_loss(theta_pred[0]-theta_pred_diag, theta_true_b-theta_true_b_diag), get_obj_val(theta_pred[0], Sb[0], model.rho_l1[k]), get_obj_val(theta_true_b, Sb[0], model.rho_l1[k]), get_obj_val(theta_true_b, Sb[0], args.rho_init)
                    res_conv[k+1].append([cv_loss, obj_pred, obj_true_rho, obj_true_orig, cv_loss_off_diag])
                elif args.MODEL_type == 'th_NN':
                    cv_loss, cv_loss_off_diag = get_convergence_loss(theta_pred[0], theta_true_b), -1 # get_convergence_loss(theta_pred[0]-theta_pred_diag, theta_true_b-theta_true_b_diag)
                    res_conv[k+1].append([cv_loss, cv_loss_off_diag])

            final_nmse = get_convergence_loss(theta_pred[0], theta_true_b)
            theta_pred = theta_pred[0].data.cpu().numpy()
    #        theta_true_b = theta_true_b[0].data.cpu().numpy()
            theta_true_b = theta_true_b.data.cpu().numpy()

            fdr, tpr, fpr, shd, nnz, nnz_true, ps = metrics.report_metrics(theta_true_b, theta_pred)
            cond_theta_pred, cond_theta_true_b = np.linalg.cond(theta_pred), -1 #np.linalg.cond(theta_true_b)
            res.append([fdr, tpr, fpr, shd, nnz, nnz_true, ps, cond_theta_pred, cond_theta_true_b])

        res_mean = np.mean(np.array(res), 0)
        res_std  = np.std(np.array(res), 0)
        res_mean = ["%.3f" %x for x in res_mean]
        res_std  = ["%.3f" %x for x in res_std]
        
        if flagP:
            print('Structure learning Metrics')
            print('Average result over test graphs')    
            #print('fdr, tpr, fpr, shd, nnz, nnz_true, np.linalg.cond(theta_pred), np.linalg.cond(theta_true)')
            print('fdr, tpr, fpr, shd, nnz, nnz_true, ps, np.linalg.cond(theta_pred), np.linalg.cond(theta_true)')
            print(*sum(list(map(list, zip(res_mean, res_std))), []), sep=', ')

            #print('ITR, conv.loss, obj_val_pred, obj_val_true_model_rho, obj_val_pred_args_rho')#, theta_pred)
            print('ITR, conv_loss"ecoli_M"+str(eM), obj_val_pred, obj_val_true_model_rho, obj_val_pred_args_rho, conv_loss_off_diag')#, theta_pred)
            for i in res_conv:
                mean_vec = ["%.3f" %x for x in np.mean(res_conv[i], 0)]
                std_vec = ["%.3f" %x for x in np.std(res_conv[i], 0)]
                print(i, *sum(list(map(list, zip(mean_vec, std_vec))), []), sep=', ')
    #            print(i, np.mean(res_conv[i], 0), np.std(res_conv[i], 0))

        if SAVE_GRAPH:
            x = np.where(theta_pred>0, 1, 0)
            A = np.matrix(x-np.eye(x.shape[0]))
            G = nx.from_numpy_matrix(A)
            fig = plt.figure(figsize=(15, 15))
            mapping = {n1:n2 for n1, n2 in zip(G.nodes(), mn.nodes)}
            G = nx.relabel_nodes(G, mapping)
            #nx.draw_networkx(G, pos=nx.spring_layout(G), with_labels = True)
            #nx.draw_networkx(G, pos=nx.circular_layout(G), with_labels = True)
            nx.draw_networkx(G, pos=nx.shell_layout(G), with_labels = True)
            plt.savefig(name+'_'+str(eM)+".pdf", bbox_inches='tight')
            nx.draw_networkx(mn.G_true, pos=nx.shell_layout(G), with_labels = True)
            plt.savefig(name+'_true_'+str(eM)+".pdf", bbox_inches='tight')
            # saving the graph
            #nx.write_gpickle(G, name+'_'+str(eM)+'.gpickle')
            nx.write_adjlist(G, name+'_'+str(eM)+'.adjlist')

        return np.float(res_mean[3]), np.float(res_mean[6]), final_nmse  # The PS mean value, final NMSE obtained

def isPSD(A, tol=1e-7):
    E,V = scipy.linalg.eigh(A)
    print('min_eig = ', np.min(E) , 'max_eig = ', np.max(E), ' min_diag = ', np.min(np.diag(A)))
    return np.all(E > -tol)

def check_sym(a, tol=1e-7):
    print('is PSD: ', isPSD(a))
    return np.allclose(a, a.T, atol=tol)

def main():
    if ONLY_ECOLI == 0:
        print('creating the graph data for GLASSO UNROLLED ADMM ')
        # M = Samples, N = features :  out_data = M x N
        if args.EDGE_RECOVERY == 1:
            print('Getting the graphs for edge recovery expt')
#            mn = create_MN_edge_recovery(args.K_train, args.M, args.N, args.graph_type, args.SAMPLE_BATCHES, args.w, args.K_test, args.K_valid)
            mn = create_MN_vary_w(args.K_train, args.M, args.N, args.graph_type, args.SAMPLE_BATCHES, [args.w_min, args.w_max], args.K_test, [args.prob, args.MAX_DEG, args.SIGNS], args.K_valid)
#            mn = create_MN_random(args.K_train, args.M, args.N, args.graph_type, args.SAMPLE_BATCHES, [args.p_min, args.p_max], args.K_test, args.K_valid)
        elif args.varyS == 1:
            print('Getting the graphs for varying sparsity')
            mn = create_MN_varyS(args.K_train, args.M, args.N, [0.03, 0.1, 0.2], args.K_valid, args.K_test)
        else: # create graphs with varying sparsity
            print('Getting the graphs for random structure')
            mn = create_MN(args.K_train, args.M, args.N, args.prob, args.K_valid, args.K_test)
        
        train_data, valid_data, test_data = prepare_data(mn)
  
#    print('chkk: ', valid_data, len(valid_data))
    
        if TRAIN == True:
            print('Training the glasso model')
            print('check: ', train_data[0].shape, train_data[1].shape)
            nmse_model, shd_model, ps_model = train_glasso(train_data, valid_data) 

        model = nmse_model #shd_model # or nmse_model
        print('TIMING check:')
        glasso_predict(model, train_data)



        print('model trained: Predicting on...')
        #torch.save(model.state_dict(), 'Gista_model.pt')
        print('****Train Data, same pred matrix, different samples****')
        #glasso_predict(nmse_model, train_data, True)
        glasso_predict(model, train_data, True)
        if len(valid_data)>0:
            print('****Valid Data****')
            #glasso_predict(nmse_model, valid_data, True)
            glasso_predict(model, valid_data, True)
        print('****Test Data, model_NMSE: average results over different samples****')
        glasso_predict(model, test_data, True)
        print('****Test Data, tag: model_SHD : average results over different samples****')
        glasso_predict(ps_model, test_data, True)
        #glasso_predict(shd_model, test_data, True)

#    mn_ecoli = create_MN_ecoli(1, eM, 'ecoli')
     
    """
    print('ECOLI: predict and plot')
    #for eM in [10, 30, 100]:#, 5000, 1000, 400, 100]:
    for eM in [20, 50, 100, 500]:#, 5000, 1000, 400, 100]:
        name = 'ecoli_sub_30' #30 edges selected(43 nodes)    +str(eM) # ecoli samples M
        mn_ecoli = create_MN_ecoli(1, eM, name)
        ecoli_data = prepare_data_helper(mn_ecoli.train_graphs)
        print('S_matrix', ecoli_data)
        
        # training for the ECOLI data
#        ecoli_nmse_model, ecoli_shd_model, ecoli_ps_model = train_glasso(ecoli_data, ecoli_data) 
        print('predict for M = ', eM)
         
        glasso_predict(model, ecoli_data, True, SAVE_GRAPH=True, eM=eM, name=name, mn=mn_ecoli)
#        glasso_predict(ecoli_nmse_model, ecoli_data, True, SAVE_GRAPH=True, eM=eM, name=name, mn=mn_ecoli)
    """
    return 

if __name__=="__main__":
    main()
