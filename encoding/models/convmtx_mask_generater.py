import torch
import numpy as np
from torch.autograd import Variable
from scipy import signal
import torch.nn as nn
import time
import torch.nn.functional as F
def convmtx2_matlab(H=torch.ones(3,3), M=5, N=5):
    '''convmtx2 2-D convolution matrix. this script is modyfied according to MATLAB convmtx2.m
    T = convmtx2(H,M,N) returns the convolution matrix for the matrix H.
    If the input matrix X is a M-by-N matrix, then reshape(T*X(:), size(H)+[M N]-1)
    is the same as conv2(X, H).
    '''
    # brute-force method, P , Q should be odd numbers which represent receptive filed of view P*Q
    P, Q = H.size()[0], H.size()[1]
    rc = int((P - 1) / 2)
    rw = int((Q - 1) / 2)
    # X = torch.zeros(M, N)
    # T = torch.zeros(M * N, M * N)

    blockHeight = M+P-1
    blockWidth = M
    blockNonZeros = P*M
    totalNonZeros = Q*N*blockNonZeros

    THeight = (N+Q-1)*blockHeight
    TWidth = N*blockWidth

    Tvals =torch.zeros(totalNonZeros,1)
    Trows = torch.zeros(totalNonZeros,1)
    Tcols = torch.zeros(totalNonZeros,1)

    c= torch.mm(torch.diag(torch.FloatTensor(range(0,M))),torch.ones(M,P))
    r= torch.add(c,torch.FloatTensor(range(0,P)).view((1,P)).repeat(M,1))
    r=r.t().contiguous().view((-1,1))
    c=c.t().contiguous().view((-1,1))

    r=r.repeat(1,N)
    c=c.repeat(1,N)

    colOffsets = (torch.FloatTensor(range(1,N+1))-1)*M
    colOffsets = colOffsets.repeat(M*P,1)
    colOffsets = torch.add(colOffsets, c)
    colOffsets = colOffsets.t().contiguous().view((-1,1))

    rowOffsets = (torch.FloatTensor(range(1,N+1))-1)*blockHeight
    rowOffsets = rowOffsets.repeat(M*P,1)
    rowOffsets = torch.add(rowOffsets, r)
    rowOffsets = rowOffsets.t().contiguous().view((-1,1))

    for k in range(Q):
        val=H[:,k]
        val=val.repeat(M,1)
        val=val.view((-1,1))

        first = k*N*blockNonZeros+1
        last = first+ N*blockNonZeros-1
        Trows[first-1:last] = rowOffsets
        Tcols[first-1:last] = colOffsets
        Tvals[first-1:last] = val.repeat(N,1)

        rowOffsets = rowOffsets+blockHeight

    T=torch.sparse.FloatTensor(torch.LongTensor(torch.stack((Trows,Tcols))[:,:,0].numpy()),Tvals.view((-1)),torch.Size([THeight,TWidth])).to_dense()
    mask = torch.zeros(M+P-1,N+Q-1)
    mask[rc:M+rc,rw:N+rw]=1
    mask_indx=mask.t().contiguous().view((-1,1))
    T=T[np.where(mask_indx[:].numpy()[:,0]>0),:][0]
    return T

def convmtx2_bf2(H=torch.ones(3,3), M=5, N=5):
    P, Q = H.size()[0], H.size()[1]
    rc = int((P - 1) / 2)
    rw = int((Q - 1) / 2)
    X = torch.zeros(M+P-1, N+Q-1)
    T = torch.zeros(M , N, M , N)
    for i in range(M):
        for j in range(N):
            X[i:i+P,j:j+Q]=1
            T[i,j]=X[rc:M+P-1-rc,rw:N+Q-1-rw]
            X = torch.zeros(M + P - 1, N + Q - 1)
    return T.view(M*N,M*N)

def convmtx2_bf2MV(mviews=[13,25,49,96], M=96, N=96):
    mviews0 = mviews
    mv=len(mviews)
    Tmv = torch.zeros(mv, M, N, M, N)
    if mviews[-1]==M:
        Tmv[mv-1]=torch.ones(M , N, M , N)
        mviews=mviews[:-1]

    for n, view in enumerate(mviews):
        H = torch.ones(view, view)
        P, Q = H.size()[0], H.size()[1]
        rc = int((P - 1) / 2)
        rw = int((Q - 1) / 2)
        X = torch.zeros(M + P - 1, N + Q - 1)
        # T = torch.zeros(M , N, M , N)
        for i in range(M):
            for j in range(N):
                X[i:i+P,j:j+Q]=1
                Tmv[n,i,j]=X[rc:M+P-1-rc,rw:N+Q-1-rw]
                X = torch.zeros(M + P - 1, N + Q - 1)
    Mva = Tmv.sum(dim=0)
    if mviews0[-1] == M:
        Mva=1./Mva
    else:
        Mva2=1./(Mva+1e-9)
        Mva=torch.where(Mva>0.5,Mva2,Mva)
    return Tmv.view((mv,M*N,M*N)), Mva.view((M*N,M*N))


def convmtx2_bf(H=torch.ones(3,3), M=5, N=5):
    '''convmtx2 2-D convolution matrix. this script is modyfied according to MATLAB convmtx2.m
    T = convmtx2(H,M,N) returns the convolution matrix for the matrix H.
    If the input matrix X is a M-by-N matrix, then reshape(T*X(:), size(H)+[M N]-1)
    is the same as conv2(X, H).
    '''
    # brute-force method, P , Q should be odd numbers which represent receptive filed of view P*Q
    P, Q = H.size()[0], H.size()[1]
    rc = int((P - 1) / 2)
    rw = int((Q - 1) / 2)
    X = torch.zeros(M, N)
    T = torch.zeros(M * N, M * N)
    if P==M and Q==N:
        T=torch.ones(M * N, M * N)
        return T
    for i in range(M):
        for j in range(N):
            if i < rc:
                if j < rw:
                    X[0:i + rc+1, 0:j + rw+1] = 1
                elif j <= (N - 1 - rw):
                    X[0:i + rc+1, j - rw:j + rw+1] = 1
                else:
                    X[0:i + rc+1, j - rw:N] = 1

            elif i <= (M - 1 - rc):
                if j < rw:
                    X[i - rc:i + rc+1, 0:j + rw+1] = 1
                elif j <= (N - 1 - rw):
                    X[i - rc:i + rc + 1, j - rw:j + rw + 1] = 1
                else:
                    X[i - rc:i + rc + 1, j - rw:N] = 1

            else:
                if j < rw:
                    X[i - rc:M, 0:j + rw+1] = 1
                elif j <= (N - 1 - rw):
                    X[i - rc:M, j - rw:j + rw + 1] = 1
                else:
                    X[i - rc:M, j - rw:N] = 1
            T[j + i * N, :] = X.view((1,-1))
            X = torch.zeros(M, N)
    return T

def convmtx2_torch(H=torch.ones(3,3), M=5, N=5):
    P, Q = H.size()[0], H.size()[1]
    rc = int((P - 1) / 2)
    rw = int((Q - 1) / 2)
    X = torch.ones(1,1,M, N).cuda()
    T = torch.zeros(M * N, M * N).cuda()
    unfold = nn.Unfold(kernel_size=(P, Q), padding=(rc,rw))
    output = unfold(X)
    output_diag = torch.diag_embed(output, offset=0, dim1=-2, dim2=-1)

    fold = nn.Fold(output_size=(M, N), kernel_size=(P, Q), padding=(rc,rw))
    # input = torch.randn(1, 3 * 3 * 3, 1)
    output2 = fold(output_diag.permute(0, 3, 1, 2).contiguous().view((1,P*Q*M*N, M*N)))

def im2col():
    stride = (1, 1)
    kernel_size = (3, 3)
    # inp = torch.randn(1, 1, 5, 5)
    x = torch.arange(0, 25).resize_(1,1, 5, 5).double()
    xpad=F.pad(x,pad=(1,1,1,1), mode='constant', value=0)

    # unfold = nn.Unfold(kernel_size=(2, 3))
    # input = torch.randn(2, 5, 3, 4)
    # output = unfold(input)
## torch implementation
    # xnn = x.view((1,1,5,5))
    xnn_unfold = torch.nn.functional.unfold(xpad, kernel_size, dilation=1, padding=0, stride=1)

## numpy implementation
    ynp=np.zeros((25,9))
    for i in range(5):
        for j in range(5):
            ynp[i*5+j,:]=xpad.numpy()[0,0,i:i+3,j:j+3].ravel()
    # err=xnn_unfold.view(9, 25).t().numpy() - ynp
    return xnn_unfold



if __name__ == '__main__':
    tic=time.time()
    output1=convmtx2_bf(H=torch.ones(13, 13), M=96, N=96)
    toc1=time.time()
    output2=convmtx2_bf2(H=torch.ones(13,13), M=96, N=96)
    toc2=time.time()
    print(toc1-tic)
    print(toc2-toc1)
    print((output1-output2).sum())
    print((output1-output1.t()).sum())
    unfold = nn.Unfold(kernel_size=(3, 3),padding=1)
    input = torch.randn(1, 1, 4, 5)
    output = unfold(input)
    output_diag=torch.diag_embed(output,offset=0,dim1=-2,dim2=-1)

    fold = nn.Fold(output_size=(4, 5), kernel_size=(3, 3),padding=1)
    # input = torch.randn(1, 3 * 3 * 3, 1)
    output2 = fold(output_diag.permute(0,3,1,2).contiguous().view((1,180,20)))

    output.size()

    inp = torch.randn(1, 1, 10, 12)
    w = torch.randn(1, 1, 4, 5)
    inp_unf = torch.nn.functional.unfold(inp, (4, 5))
    out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
    out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
    # or equivalently (and avoiding a copy),
    # out = out_unf.view(1, 1, 7, 8)
    conv2dout = torch.nn.functional.conv2d(inp, w)
    err=(conv2dout - out).abs().max()
    y = im2col()
    M=96
    N=96
    P=13
    Q=13
    X=torch.randn(M,N).cuda()
    H=torch.ones(P,Q).cuda()
    tic=time.time()
    T=convmtx2_bf(H,M,N)
    toc = time.time()
    print(toc-tic)
    Tbf = T.numpy()
    tic = time.time()
    T2 = convmtx2_matlab(H,M,N)
    toc = time.time()
    print(toc - tic)
    T2matlab = T2.numpy()

    convmtx_res=torch.mm(T,X.view((-1,1))).view((M,N))
    np_convmtx_res=convmtx_res.numpy()
    conv_res=signal.convolve2d(X.numpy(), H.numpy(), boundary='fill', fillvalue=0, mode='same')

    print(T.shape)


    x = X.view(1, 1, M,N)


    h=H.view(1,1,P,Q)

    weights = Variable(h)  # torch.randn(1,1,2,2)) #out_channel*in_channel*H*W


    y = F.conv2d(x, weights, padding=6)
    res = (y - convmtx_res).numpy()
    res2= conv_res-np_convmtx_res
    print("y:", y-convmtx_res)