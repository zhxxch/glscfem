# -*- coding: utf-8 -*-  
import sae
import ref
import numpy as np

'''
FEMinputFormat =  dict({
    'XY': np.array(['节点编码','节点X坐标', '节点Y坐标']),
    'TEVW': dict({
            't': '厚度',
            'E': '弹性模量',
            'pr': '泊松比'
        }),
    'IJMH': [[单元编号,np.array(['节点单元编号', '节点编码'])]],#节点编码
    'HIP': np.array(['节点编码', 'X向荷载', 'Y向荷载', 'Z向荷载']),
    'BIU': np.array(['节点编码', 'X向位移约束', 'Y向位移约束', 'Z向位移约束'])
    '''

def FEM(inputDict):
#有限元
    HIP = ref.arrIndexSort(inputDict['HIP'])
    HIP = np.matrix(HIP[:,1:3].reshape(-1,1))
    KS = STIFF(inputDict)
    KS = INSCD(KS, inputDict['BIU'])
    Solution = np.linalg.solve(KS, HIP)
    Solution = np.array(Solution.reshape((-1,2)))
    return Solution

def area(ENXY):
#计算三角单元面积   #ENXY: np.array([['节点编码','节点X坐标', '节点Y坐标'],..,...])  
    if ENXY.shape == (3,3):#三角单元
        triArea = np.absolute(0.5 * np.cross(ENXY[1,[1,2]]-ENXY[0,[1,2]], ENXY[2,[1,2]]-ENXY[0,[1,2]]))
        return triArea
    else: return 0

def DTE(E, PR):
#生成平面应力弹性常数矩阵   #E/弹性模量; PR/泊松比
    D = np.zeros((3,3))
    D[0,0] = E / (1 - PR*PR)
    D[0,1] = PR * D[0,0]
    D[1,0] = D[0,1]
    D[1,1] = D[0,0]
    D[2,2] = 0.5 * E / (1 + PR)
    D = np.matrix(D)
    return D

def BTE(ENXY):
#计算常应变T3单元几何矩阵
    B = np.zeros((3,6))
    B[0,0] = ENXY[1,2] - ENXY[2,2]
    B[0,2] = ENXY[2,2] - ENXY[0,2]
    B[0,4] = ENXY[0,2] - ENXY[1,2]
    B[1,1] = ENXY[2,1] - ENXY[1,1]
    B[1,3] = ENXY[0,1] - ENXY[2,1]
    B[1,5] = ENXY[1,1] - ENXY[0,1]
    B[2,0] = B[1,1]
    B[2,1] = B[0,0]
    B[2,2] = B[1,3]
    B[2,3] = B[0,2]
    B[2,4] = B[1,5]
    B[2,5] = B[0,4]
    B = np.matrix(B)
    B = (0.5/area(ENXY)) * B
    return B

def STE(ENXY, E, PR, t, NE,IJMH):
#计算单元刚度矩阵
    B = BTE(ENXY)
    D = DTE(E, PR)
    S = D * B   #内力矩阵
    KE = B.T * S * area(ENXY) * t    #单元刚度矩阵
    KE = np.matrix(KE)
    KEl = [IJMH[NE][1], KE]  #[单元编号对应, 单元刚度矩阵]
    return KEl

def AINCKE(lGKS, lKE):
#lGKS = [[m,n],KS]
#lKE = [[i,j],KE]
    o=2 #offset
    m = lGKS[0][0]
    n = lGKS[0][1]
    KS = lGKS[1]
    i = lKE[0][0]
    j = lKE[0][1]
    KE = lKE[1]
    KS[m*o: m*o+o, n*o: n*o+o] = KS[m*o: m*o+o, n*o: n*o+o] + KE[i*o: i*o+o, j*o: j*o+o]
    return KS

def INCKE(KS, KEl):
#STIFF下位程序, 将单元刚度矩阵迭代插入总体刚度矩阵
    mn = ref.arrIndexSort(KEl[0])[:,1].T
    ij = ref.arrIndexSort(KEl[0])[:,0].T
    KE = KEl[1]
    map(lambda i: map(lambda j: AINCKE([[mn[i], mn[j]],KS], [[i,j],KE]), ij), ij)
    return KS

def STIFF(FEMinput):
#生成总体刚度矩阵
    IJMH = FEMinput['IJMH']
    E = FEMinput['TEVW']['E']
    PR = FEMinput['TEVW']['pr']
    t = FEMinput['TEVW']['t']
    XY = FEMinput['XY']
    KS = np.zeros((FEMinput['XY'].shape[0]*2, FEMinput['XY'].shape[0]*2))
    XY = ref.arrIndexSort(XY)
    IJMH = ref.lstIndexSort(IJMH)
    KEs = map(lambda EN: STE(XY[EN[1][:,1],:], E, PR, t, EN[0], IJMH), IJMH)#单元刚度矩阵List
    KS = np.matrix(reduce(INCKE, KEs, KS))
    return  KS

def EQUPE():
#载荷等效换算
    pass

def INSCD(KS, BIU):
#插入支承条件, 修改总体刚度矩阵
    BIU = ref.arrIndexSort(BIU)
    KS = np.matrix(reduce(INSCDS, BIU, KS))
    return KS
    
def INSCDS(KS, BIU):
#INSCD支承条件插入下位迭代程序
    NN = BIU[0]
    lX = BIU[1]
    lY = BIU[2]
    if lX == 0:
        KS[:,NN*2] = 0
        KS[NN*2,:] = 0
        KS[NN*2, NN*2] = 1
    if lY == 0:
        KS[:,NN*2+1] = 0
        KS[NN*2+1,:] = 0
        KS[NN*2+1, NN*2+1] = 1
    return KS
