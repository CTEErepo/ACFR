from __future__ import division
import torch
SQRT_CONST = 1e-10

def safe_sqrt(x, lbound=SQRT_CONST):
    return torch.sqrt(torch.clip(x, lbound, 1e+5))

def pdist2sq(X,Y):

    C = -2*torch.matmul(X,torch.transpose(Y, 0, 1))
    nx = torch.sum(torch.square(X), dim=1, keepdim=True)
    ny = torch.sum(torch.square(Y),dim=1, keepdim=True)
    D = (C + torch.transpose(ny,  0, 1)) + nx
    return D


def wasserstein(Xf,Xc,p,lam=10,its=10,sq=False,backpropT=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nc = Xc.shape[0]
    nf = Xf.shape[0]

    if sq:
        M = pdist2sq(Xf,Xc)
    else:
        M = safe_sqrt(pdist2sq(Xf,Xc))
    M_mean = torch.mean(M)
    # M_drop = torch.nn.Dropout(M,10/(nc*nt))
    delta = torch.max(M).detach()
    eff_lam = (lam/M_mean).detach()

    Mt = M
    num_row = M[:, 0:1].shape[0]
    num_col = M[0:1, :].shape[1]
    row = delta * (torch.ones([1, num_col]).to(device))
    col = torch.cat([delta * (torch.ones([num_row, 1]).to(device)), torch.zeros((1, 1)).to(device)], dim=0)
    Mt = torch.cat([M, row], dim=0)
    Mt = torch.cat([Mt, col], dim=1)


    a = torch.cat([torch.unsqueeze(p * torch.ones(nf) / nf, 1), (1 - p) * torch.ones((1, 1))], dim=0).to(device)
    b = torch.cat([torch.unsqueeze((1 - p) * torch.ones(nc) / nc, 1), p * torch.ones((1, 1))], dim=0).to(device)

    Mlam = eff_lam.to(device)*Mt
    K = torch.exp(-Mlam) + 1e-6 # added constant to avoid nan
    U = K*Mt
    ainvK = K/a

    u = a
    for i in range(0,its):
        u = 1.0/(torch.matmul(ainvK,(b/torch.transpose(torch.matmul(torch.transpose(u, 0, 1),K), 0, 1))))
    v = b/(torch.transpose(torch.matmul(torch.transpose(u, 0, 1),K), 0, 1))

    T = u*(torch.transpose(v, 0, 1)*K)

    E = T*Mt
    D = 2*torch.sum(E)
    return D, Mlam


"""
python implementation of Hilbert Schmidt Independence Criterion
hsic_gam implements the HSIC test using a Gamma approximation
Python 2.7.12

Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Scholkopf, B., 
& Smola, A. J. (2007). A kernel statistical test of independence. 
In Advances in neural information processing systems (pp. 585-592).

Shoubo (shoubo.sub AT gmail.com)
09/11/2016

Inputs:
X 		n by dim_x matrix
Y 		n by dim_y matrix
alph 		level of test

Outputs:
testStat	test statistics
thresh		test threshold for level alpha test
"""



def rbf_dot(pattern1, pattern2, deg):
    size1 = pattern1.shape
    size2 = pattern2.shape

    G = torch.sum(pattern1*pattern1, 1).reshape(size1[0],1)
    H = torch.sum(pattern2*pattern2, 1).reshape(size2[0],1)

    Q = torch.tile(G, (1, size2[0]))
    R = torch.tile(H.T, (size1[0], 1))

    H = Q + R - 2* torch.matmul(pattern1, pattern2.T)

    H = torch.exp(-H/2/(deg**2))
    return H


def hsic_gam(X, Y, alph = 0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = X.shape[0]
    Xmed = X

    G = torch.sum(Xmed*Xmed, 1).reshape(n,1)
    Q = torch.tile(G, (1, n) )
    R = torch.tile(G.T, (n, 1) )

    dists = Q + R - 2 * torch.matmul(Xmed, Xmed.T)
    dists = dists - torch.tril(dists)
    dists = dists.reshape(n**2, 1)

    width_x = torch.sqrt(0.5 * torch.median(dists[dists>0]))
    Ymed = Y

    G = torch.sum(Ymed*Ymed, 1).reshape(n,1)
    Q = torch.tile(G, (1, n) )
    R = torch.tile(G.T, (n, 1) )
    dists = Q + R - 2* torch.matmul(Ymed, Ymed.T)
    dists = dists - torch.tril(dists)
    dists = dists.reshape(n**2, 1)
    width_y = torch.sqrt( 0.5 * torch.median(dists[dists>0]))


    bone = torch.ones((n, 1)).to(device)
    H = (torch.eye(n) - torch.ones((n,n)) / n).to(device)

    K = rbf_dot(X, X, width_x)
    L = rbf_dot(Y, Y, width_y)
    Kc = torch.matmul(torch.matmul(H, K), H)
    Lc = torch.matmul(torch.matmul(H, L), H)

    testStat = torch.sum(Kc.T * Lc) / n
    return testStat





