# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 22:47:07 2021

@author: cyw
"""

import numpy as np
import scipy as sp
import pandas as pd
from scipy import interpolate
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import measure
import skimage.io
import networkx as nx
from scipy.spatial.distance import cdist
import math
import copy

# figure2knot-V1.0中的自定义函数
def func(arr,m,n):#输出 m x n 矩阵的转置
            return [[row[i] for row in arr] for i in range(n)]

def find_index(arr,m,n): #根据点的index找到其在 m x n 矩阵中的坐标
    arr=np.array(arr)
    if len(arr.shape)==1:
        idx = [np.unravel_index(int(arr[i]), (m, n)) for i in range(len(arr))]
    else:
        [l,k]=arr.shape
        arr = arr.reshape((l* k,1),order='F')
        idx = [np.unravel_index(int(arr[i][0]), (m, n)) for i in range(l * k)]
    idx=np.array(idx)
    return [idx[:,0],idx[:,1]]

def find_path(P): #根据0-1矩阵找出纽结路径
            '''
            1 得到节点与邻接矩阵
            '''
            [m, n] = P.shape
            ID = np.linspace(1, m * n, m * n).reshape(1, m*n)  # ID——节点

            P_1 = P.reshape(1, m*n)
            arr = np.where(P_1 == 0)[1]#所有0元素的index

            ID = np.delete(ID, arr, axis=1)
            nPixel=len(list(ID)[0])

            [i_1,i_2]=find_index(ID,m,n)

            A1 = np.array([[abs(i_1[i] - i_1[j]) for j in range(nPixel)] for i in range(nPixel)])
            A2 = np.array([[abs(i_2[i] - i_2[j]) for j in range(nPixel)] for i in range(nPixel)])
            A = A1+A2
            A[(A1 > 1) | (A2 > 1)] = 0

            '''
            2 graph
            '''
            G=nx.Graph()
            d=nPixel
            G.add_nodes_from(np.linspace(0,d-1,d))#添加节点

            [x1, y1] = np.where(A == 1)#赋权
            [x2, y2] = np.where(A == 2)
            C1 = np.array(np.vstack((x1, y1))).transpose()
            C2 = np.array(np.vstack((x2, y2))).transpose()
            G.add_edges_from(C1, weight=1)
            G.add_edges_from(C2, weight=2)

            # dist=np.zeros((d,d))
            dist=dict(nx.all_pairs_shortest_path_length(G))#最短路径

            D=np.array([[dist[node][nodei] for nodei in range(d)]for node in range(d)])
            I=np.argmax(D.reshape(1, d*d))

            [i, j] = np.unravel_index(I, (nPixel, nPixel), order='F')
            path = nx.shortest_path(G, i, j)#最短路径的index

            return [ID[0][index] for index in path]

def knot_smooth(R,NUM_POINTS):
            R1=R-np.mean(R,axis=0)
            R2=np.array(R1)
            R3=np.std(np.array(R))
            R=(R2)/np.array(R3)
            x = R[:,0]
            y = R[:,1]
            z = R[:,2]
            N=len(x)
            theta=np.zeros((1,N))
            for i in range(1,N):
                theta[0,i] = theta[0,i-1] + np.linalg.norm( R[i,:]-R[i-1,:])   #求范数 
            theta = theta / theta[0,-1] * 2 * math.pi
             #plt.plot(theta,R)
            theta = np.asarray(theta).squeeze()
            theta_iso = np.linspace(0,2*math.pi,NUM_POINTS)
            f1 = sp.interpolate.interp1d(theta,x,kind='slinear')
            x=f1(theta_iso)
            f2 = sp.interpolate.interp1d(theta,y,kind='slinear')
            y=f2(theta_iso)
            f3 = sp.interpolate.interp1d(theta,z,kind='slinear')
            z=f3(theta_iso)
            
            return [x,y,z]

# figure2knot-V2.0中的自定义函数
def knot2figure(R):
#R=pd.read_csv('R.csv')
    
    X=np.random.randn(3,3)#正态随机数组
    #X=np.ones((3,3))
    #X=X/3
    #print(X)
    def SchmitOrth(mat:np.array):#qr分解
        cols = mat.shape[1]
    
        Q = np.copy(mat)
        r = np.zeros((cols, cols))
    
        for col in range(cols):
            for i in range(col):
                k =  np.sum(mat[:, col] * Q[:, i]) / np.sum( np.square(Q[:, i]) )
                Q[:, col] -= k*Q[:, i]
            Q[:, col] /= np.linalg.norm(Q[:, col])
    
            for i in range(cols):
                r[col, i] = Q[:, col].dot( mat[:, i] )
    
        return Q
    
    
    Q = SchmitOrth(X)

    ar=np.array(Q) 
    #ar=np.ones((3,3))
    RR=np.zeros((256,3)) 
    for i in range(256):
        for j in range(3):
            RR[i][j]=R[i][0]*ar[:,j][0]+R[i][1]*ar[:,j][1]+R[i][2]*ar[:,j][2]    
    
    x=RR[:,0]
    y=RR[:,1]
    z=RR[:,2]
    
    x=np.array(x)    
    y=np.array(y)
    z=np.array(z)          

    Rxy=RR[:,:2]
    
    
    N=len(x)
    import math
    theta=np.zeros((1,N))
    for i in range(1,N):
        theta[:,i] = theta[:,i-1]+np.linalg.norm(Rxy[i]-Rxy[i-1]) 
    theta = theta / theta[:,-1] * 2 * math.pi;
    theta=np.array(theta).squeeze()
       
    theta_iso = np.linspace(0,2*math.pi,128*6)
     
    from scipy import interpolate
    
    f1=interpolate.interp1d(theta,x)
    x=f1(theta_iso)
    f2=interpolate.interp1d(theta,y)
    y=f2(theta_iso)
    f3=interpolate.interp1d(theta,z)
    z=f3(theta_iso)
    
    Rxy=np.concatenate((x,y)).conj().transpose()
    Rxy=np.concatenate((y,x)).conj().transpose()
        
    Rxy=Rxy.reshape((2,768))
     
    N=len(x)
    crossID = []
    
    from shapely.geometry import LineString  #判断两线段相交
    #LineString([(0, 0), (1, 1), (1, -1), (0, 1)]).is_simple
    ii=[]
    for j in range (1,N):
        for i in range (1,N):
            if not(LineString([(x[i-1], y[i-1]), (x[i], y[i]),(x[j-1], y[j-1]), (x[j], y[j])]).is_simple):
                ii.append([i,j])
    nC=len(ii)
    for i in range(nC-1,-1,-1):
        if ii[i][0] >= ii[i][1]: # filt
            del ii[i]
        elif (ii[i][0] == 1 or ii[i][1] == N-1):# filt
            del ii[i]
    nC=len(ii)
   
    dx=np.diff(x)
    dy=np.diff(y)
    for k in range(nC):
        i = ii[k][0]; j = ii[k][1];
        x1 = (x[i-1],x[i]); y1 = (y[i-1],y[i]);
        x2 = (x[j-1],x[j]); y2 = (y[j-1],y[j]);
        
        x11=x1[0];y11=y1[0];
        x12=x1[1];y12=y1[1];
        x3=x2[0];y3=y2[0];
        x4=x2[1];y4=y2[1];
        
        xc= ( (x11*y12-y11*x12)*(x3-x4)-(x11-x12)*(x3*y4-y3*x4) ) / ( (x11-x12)*(y3-y4)-(y11-y12)*(x3-x4) ) #求交点位置
        yc= ( (x11*y12-y11*x12)*(y3-y4)-(y11-y12)*(x3*y4-y3*x4) ) / ( (x11-x12)*(y3-y4)-(y11-y12)*(x3-x4) )
        
        norm1 = np.linalg.norm([np.diff(x1), np.diff(y1)]);
        norm2 = np.linalg.norm([np.diff(x2), np.diff(y2)]);
        normv1 = np.linalg.norm([xc - x1[0], yc - y1[0]]);
        normv2 = np.linalg.norm([xc - x2[0], yc - y2[0]]);
        t1 = normv1 / norm1;
        t2 = normv2 / norm2;
        zc1 = z[i-1] + t1 * ( z[i] - z[i-1]);
        zc2 = z[j-1] + t1 * ( z[j] - z[j-1]);
        
        plt.plot(xc,yc,marker='o',c='none')
        
        if zc1 > zc2:
            crossID.append(j);
        else:
            crossID.append(i);
    # # step 3: plot
  

    RN = Rxy;

    for k in range(len(crossID)-1,-1,-1):
        #i = int(crossID[0][k]);
        i=int(crossID[k])
        RN[:,i-2:i+2] = RN[:,i-2:i+2] * float("nan")
    x = RN[1,:]
    y = RN[0,:]
    
  
   # plt.legend()
    plt.plot(x,y,ls="-",lw=2)
    #plt.title(knot2figure)
    #plt.legend()
    plt.show()
def load_figure(figname):
    
    img = skimage.io.imread(figname)    
    img1=rgb2gray(img)
    img2=img1*0
    img2[img1>0.4] = 1
    img2=1-img2
    P= measure.label(img2, background = None, return_num = False,connectivity = None)
    
    return [img,P]
def collect_path(path_array,m,n):
    path_collections = {}
    nPart = len(path_array)
    endInd_array = np.zeros((nPart,2))
    for k in range(1,nPart+1):
        endInd_array[k-1,:] = [path_array[k][0],path_array[k][-1]]
    E=endInd_array.astype(int)
    
    x=[]
    y=[]
    for i in range(2):
        for j in range(nPart):
            [x_,y_]=np.unravel_index(E[j][i],(m,n))
            x=np.append(x,x_)
            y=np.append(y,y_)
    z=[x,y]

    #输出z的转置
    def func(arr,m,n):
        res=[[row[i] for row in arr ] for i in range(n)]
        return res
    z= func(z,len(z),len(z[0]))

    D = squareform(pdist(z)) + m * np.eye(2*nPart)
    I =np.argmin(D,axis=0)
    I=I.reshape([nPart,2],order='F')
    
    past_array = np.zeros((nPart,1))
    
    
    for loop in range(nPart):
        if int(min(past_array)) == 1:
            break
        path_array_sorted = {}
        ne = np.where(past_array==0)
        nex=ne[0][0]
        j=nex
        jend=1
        for inner_loop in range(nPart):
            j=j+1
            if jend == 1:
                path_array_sorted[inner_loop]=copy.deepcopy(path_array[j])
            else:#if jend == 0
                path_array_sorted[inner_loop] = np.flipud(path_array[j])
            past_array[j-1] = 1
            i = j-1
            iend = jend
            mj = I[i,iend]
            [j,jstart] = np.unravel_index(mj,(nPart,2),order='F')
            jend = 1 - jstart
            if past_array[j] == 1 :
                path_collections[loop]=path_array_sorted
                break
        
    return path_collections
def assign_z_end(path_collections,m,n):
    path_array_sorted = {}
    i=0
    for loop in range(len(path_collections)):
        # path_array_sorted =np.hstack((path_array_sorted,path_collections[loop]))
        for j in range(len(path_collections[loop])):   
            path_array_sorted[i]=path_collections[loop][j]
            i=i+1
    nPart = len(path_array_sorted)
    paths = copy.deepcopy(path_array_sorted)
    paths[nPart]=path_collections[0][0]
    z_paths =copy.deepcopy(path_array_sorted)
    for k in range(nPart):
        for g in range(len(z_paths[k])):
            z_paths[k][g]=np.nan
        z_paths[k][0] = -1  
        
    for k in range(nPart):
        [i1,j1]=np.unravel_index(int(paths[k][-1]),(m,n))
        [i2,j2]=np.unravel_index(int(paths[k+1][0]),(m,n))

        im = (i1+i2)/2
        jm = (j1+j2)/2#各段首尾像素点的中点
        
        nearp = 1
        ip = 1
        mindist = m
        for v in range(nPart):
            iv=[]
            jv=[]
            for i in range(len(paths[v])):
                [iv_,jv_]=np.unravel_index(int(paths[v][i]),(m,n))
                iv=np.append(iv,iv_)
                jv=np.append(jv,jv_)
            A=np.array([(im,jm)])
            B=np.array(np.vstack((iv,jv))).transpose()
            dists=cdist(A,B)#寻找(im,jm)与各点的距离
        
            mindist_=dists.min(1)
            ip_=np.argmin(dists,axis=1)#最近点的索引
            
            
            if mindist_ < mindist:
                nearp = v
                ip = ip_
                mindist = mindist_
        z_paths[nearp][int(ip)] = 1
    
    nCurve = len(path_collections)
    curve_collection = {}
    start = 1;
    for loop in range(nCurve):
        final = start + len(path_collections[loop]) - 1
        x = []; y = []; z = [];
        for k in range(start-1,final):
            i=[]
            j=[]
            for hh in range(len(paths[k])):
                [i_,j_] =np.unravel_index(int(paths[k][hh]), (m,n))
                i=np.append(i,i_)
                j=np.append(j,j_)

            x=np.append(x,i)
            y=np.append(y,j)
            z=np.append(z,z_paths[k])
        
        x=np.append(x,x[0])
        y=np.append(y,y[0])
        z=np.append(z,z[0])
        curve_collection[loop] =np.array(np.vstack((x,y,z))).transpose()
        start = final+1
    
    return curve_collection
def normalize_curves(curve_collection):
    nCurve = len(curve_collection)
    R_all = copy.deepcopy(curve_collection[0])
    for loop in range (1,nCurve):
        R = curve_collection[loop]
        R_all =np.vstack((R_all,R))
    meanR = np.mean(R_all,axis=0)
    stdR=np.std(R_all,0,ddof = 1)
    meanR[2] = 0
    stdR[2]= 1
    for loop in range(nCurve):
        R = curve_collection[loop]
        R = (R - meanR) / stdR
        curve_collection[loop]= R
    return curve_collection
def complete_by_interp(R):
    Rxy=R[:,:2]
    z=R[:,2]
    
    N = len(z)
    z=np.reshape(z,N)
    
    theta =np.zeros((1,N))
    for i in range(1,N):
        theta[0,i] = theta[0,i-1]+np.linalg.norm(Rxy[i]-Rxy[i-1]) 
    theta = theta / theta[-1,-1]
    
    xe = np.hstack((theta[0,:-2]-1,theta[0,:],theta[0,1:-1]+1)) 
    z=np.hstack((z[:-2]-1,z,z[1:-1]))
        
    t=np.argwhere(np.isnan(z))#NaN点   
    xe=np.delete(xe,t,0)
    z=np.delete(z,t,0)

    cs=sp.interpolate.splrep(xe,z,k=3)
    R[:,2]=sp.interpolate.splev(theta,cs)
    
    return R
def polish_curve(R, num_points, nearN):
    
    #step 1: reparameterization
    x = R[:,0]; y = R[:,1]; z = R[:,2]; 
    N = len(x);
    theta=[0]*N
    for i in range(1,N):
        theta[i] = theta[i-1]+np.linalg.norm(R[i]-R[i-1]) 
    theta = theta / theta[-1] 
    
    
    theta_iso =np.linspace(0.0,1.0,int(num_points))
    theta_iso =np.reshape(theta_iso,(int(num_points),1))
    
    f1=interpolate.interp1d(theta,x)
    x=f1(theta_iso)
    f2=interpolate.interp1d(theta,y)
    y=f2(theta_iso)
    f3=interpolate.interp1d(theta,z)
    z=f3(theta_iso)
    
    #step 2: smooth， 利用 smooth 函数进行光滑化，并且，让曲线更好看一些
    def smooth(a, SMOOTH_NEAR):
        '''
        a: 需要平滑的一维向量
        SMOOTH_NEAR: 平滑窗口的长度,必须为奇数
        参考：https://www.codenong.com/40443020/
        '''
        out0 = np.convolve(a, np.ones(SMOOTH_NEAR, dtype=int), 'valid') / WSZ
        r = np.arange(1, SMOOTH_NEAR - 1, 2)
        start = np.cumsum(a[:SMOOTH_NEAR - 1])[::2] / r
        stop = (np.cumsum(a[:-SMOOTH_NEAR:-1])[::2] / r)[::-1]
        return np.concatenate((start, out0, stop))

    x = smooth(x.flatten(), nearN)
    y = smooth(y.flatten(), nearN)
    z = smooth(z.flatten(), nearN)
    
    R = [x,y,z]
    
    return R
def get_curve_lengths(curve_collection, NUM_POINTS):
    nCurve = len(curve_collection)
    lengths =np.zeros((nCurve,1))
    num_points = np.zeros((nCurve,1))

    for loop in range(nCurve):
        R = curve_collection[loop]
        N=R.shape[0]
        theta=np.zeros((1,N))
        for i in range(1,N):
            theta[0,i] = theta[0,i-1]+np.linalg.norm(R[i]-R[i-1]) 
        lengths[loop,0] = theta[0,-1]

    total_length=sum(lengths)

    for loop in range(0,nCurve-1):
        num_points[loop,0] = np.round( NUM_POINTS * lengths[loop,0] / total_length,0)
    num_points[-1,0] = NUM_POINTS - sum(num_points)

    return [lengths, num_points]

class f2k:      
    def __init__(self,pth):
        self.pth=pth#path
        self.img=skimage.io.imread(pth)#image

        [self.row,self.col]=rgb2gray(self.img).shape

        img2=rgb2gray(self.img)*0
        img2[rgb2gray(self.img)<0.4]=1
        self.pic=measure.label(img2, background = None, return_num = False,connectivity = None)

    def fun1(self):#figure2knot-V1.0主函数
        nPart=np.max(self.pic)
        P=self.pic
        m,n=self.row,self.col

        endInd_array = np.zeros((nPart,2))#4x2，每条曲线的两端index

        path_array = {}
        for k in range(1,nPart+1):
            M=np.ones((m,n))
            M[P!=k]=0
            path_ind = find_path(M)
            path_array[k] = path_ind
            endInd_array[k-1,:] = [path_ind[0],path_ind[-1]]                      

            # [i,j]=find_index(path_ind,m,n)

        z=find_index(endInd_array,m,n)
        z= func(z,len(z),len(z[0]))

        D = squareform(pdist(z)) + m * np.eye(2*nPart)

        I=np.argmin(D,0)#各线段连接顺序
        I = I.reshape([nPart,2],order='F')

        path_array_updated =copy.deepcopy(path_array)#依序index列
        path_array_connected = path_array_updated[1]

        i = 0
        iend = 1
        PP = P

        for k in range(2,nPart+1):
            mj=I[i,iend]
            #[j,jstart] = ind2sub([nPart,2],mj)
            [j,jstart] = np.unravel_index(mj,[nPart,2],order='F')

            j=j+1    
            jend = 1 - jstart
            path_array_updated[k] = copy.deepcopy(path_array[j])
            if jend==0:
                path_array_updated[k] = np.flipud(path_array[j])
            i = j-1
            iend=jend
            path_array_connected =np.hstack((path_array_connected,path_array_updated[k]));

            PP[np.where(P==j)]=k

        # path_array_connected = np.hstack((path_array_connected, path_array_updated[1]))

        paths = path_array_updated
        paths[nPart+1] = path_array_updated[1]
        zpaths = copy.deepcopy(path_array_updated)

        for i in range(1,nPart+1):
            for k in range(len(zpaths[i])):
                zpaths[i][k]=np.nan

            zpaths[i][0] = -1
            zpaths[i][-1] = -1

        for k in range(1,nPart+1):
            [i1,j1]=np.unravel_index(int(paths[k][-1]),(m,n))
            [i2,j2]=np.unravel_index(int(paths[k+1][0]),(m,n))

            im = (i1+i2)/2
            jm = (j1+j2)/2#各段首尾像素点的中点

            nearp = 1
            ip = 1
            mindist = m

            for v in range(1,nPart+1):
                [iv,jv]=find_index(paths[v],m,n)
                A=np.array([(im,jm)])
                B=np.array(np.vstack((iv,jv))).transpose()
                dists=cdist(A,B)#寻找(im,jm)与各点的距离

                mindist_=dists.min(1)
                ip_=np.argmin(dists,axis=1)#最近点的索引
                if mindist_ < mindist:
                    nearp = v
                    ip = ip_
                    mindist = mindist_

            zpaths[nearp][int(ip)] = 1   #交叉点赋值为1     

        for k in range(1,nPart+1):
            z = zpaths[k]

            id1=np.linspace(1,len(z),len(z))
            id_ = copy.copy(id1)

            t=np.argwhere(np.isnan(z))#NaN点   
            id1 = np.delete(id1,t,0)
            z=np.delete(z,t,0)

            x = copy.copy(id1)
            xv = np.linspace(1,x[-1],len(id_))

            cs=interpolate.splrep(x,0+z+0,k=1)
            zpaths[k]=interpolate.splev(xv,cs,der=0)

        x = []
        y = []
        z = []

        for k in range(1,nPart+1):
            [i,j]=find_index(paths[k],m,n)
            x = np.append(x, i)
            y = np.append(y, j)
            z = np.append(z, zpaths[k])

        x = np.append(x,x[1])
        y = np.append(y,y[1])
        z = np.append(z,z[1])

        # step 6 smooth
        # R=np.vstack((x,y,z)).transpose()
        #
        # [x,y,z]=knot_smooth(R,256)
        # R=np.vstack((x,y,z)).transpose()
        # knot2figure(R)
        #绘制图像
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(x, y, z, label='parametric curve')
        ax.legend()
        plt.show()##
        return np.array((np.array(x),np.array(y),np.array(z)))

    # figure2knot-V2.0主函数
    # NUM_POINTS为输出三维参数曲线点的个数；SMOOTH_NEAR为smooth函数滑动窗口长度
    def fun2(self,NUM_POINTS =256,SMOOTH_NEAR = 3):
        P=self.pic
        nPart = np.max(P)
        
        [m,n]=P.shape

        # # step 2: 将每一个连通部分转化为曲线 -- function Find_Path --> path_array
        
        path_array={}
        for k in range(1,nPart+1):#
            M=np.ones((m,n))
            M[P!=k]=0
            path_ind =find_path(M)
            path_array[k] = path_ind

        #% 链环情形：path_collections 会有多个 path_array_sorted， 做成一个 cell，每个元素是一个 path_array_sorted
        path_collections =collect_path(path_array,m,n)
        path_array_connected = {}
        for k in range(len(path_collections)): #K=0,1
            path_array_ = copy.deepcopy(path_collections[k])
            for i in range(len(path_array_)):
                path_array_connected[k] = np.hstack((path_array_connected, path_array_[i]))#连接list
            path_array_connected[k] = np.hstack((path_array_connected, path_array_[0][0]))
        
        # # step 4: 转化为 3D 曲线, 其中 complete_by_interp 是影响 3D 效果的关键

        curve_collection = assign_z_end(path_collections, m, n)
        curve_collection = normalize_curves(curve_collection)
        nCurve = len(path_collections)
        for i in range(nCurve):
            R = curve_collection[i]
            R = complete_by_interp(R)
            curve_collection[i] = R

        # # step 5：美化一下
        [lengths, num_points] = get_curve_lengths(curve_collection, NUM_POINTS)
        for i in range(nCurve):
            R = curve_collection[i]
            num_point = num_points[i]
            smooth_near = SMOOTH_NEAR
            R = polish_curve(R, num_point, smooth_near)
            curve_collection[i] = R

        # # step 6：画图保存
        # 绘制图像
        fig = plt.figure() 
        ax = fig.add_subplot(projection='3d')

        x1,y1,z1=[],[],[]
        for i in range(nCurve):
            R = curve_collection[i]
            x = list(R[0])
            y = list(R[1])
            z = list(R[2])
            tmp = pd.DataFrame(x,columns=['X'])
            tmp['Y']=y
            tmp['Z']=z
            ax.plot(tmp['X'], tmp['Y'], tmp['Z'])

            x1.extend(x)
            y1.extend(y)
            z1.extend(z)
        plt.show()

        return np.array((np.array(x1), np.array(y1), np.array(z1)))
           
   