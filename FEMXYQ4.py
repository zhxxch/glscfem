import sae
import ref
import pylibmc as mc
cache = mc.Client()
import numpy as np

'''
FEMinputFormat =  dict({
    'XY': np.array(['NodeLable','XCoordinate', 'YCoordinate']),
    'TEV': dict({
            'T': 'Thickness',
            'E': 'ElasticModulus',
            'V': 'v(P.R.)'
        }),
    'IJ': [[ElementLable, np.array(['NodeLableInElement', 'NodeLable'])]],
    'LD': np.array(['NodeLable', 'XLoad', 'Y....', 'Z...']),
    'LM': np.array(['NodeLable', 'XLimit', 'Y....', 'Z...'])#Limit=0 or else
    '''

def FEM(inputDict):
    #MAIN
    LD = cmpLoad(inputDict['LD'], inputDict['XY'].shape[0])
    KS = GKS(inputDict)
    KS = insLmt(KS, inputDict['LM'])
    DP = np.linalg.solve(KS, LD)
    DP = np.array(DP.reshape((-1,2)))
    return DP

def KEsquare(E, v, T):
#Calculate element stiffness matrix
    KE = np.zeros((8,8))
    map(lambda i: map(lambda j: insKEIJ([[i,j], KE], [[0,0], KEij(i, j, v)]), range(4)), range(4))
    KE = np.matrix(KE)
    KE *= (E*T)/(4.0*(1.0-v*v))
    return KE

def KEij(i, j, v):
    eta = (-1.0,-1.0,1.0,1.0)
    xi = (-1.0,1.0,1.0,-1.0)
    KEij = np.zeros((2,2))
    KEij[0,0] = (1.0+(1.0/3.0)*eta[i]*eta[j])*xi[i]*xi[j] + ((1.0-v)/2.0)*(1.0+(1.0/3.0)*xi[i]*xi[j])*eta[i]*eta[j]
    KEij[0,1] = v*xi[i]*eta[j] + ((1.0-v)/2.0)*eta[i]*xi[j]
    KEij[1,0] = v*eta[i]*xi[j] + ((1.0-v)/2.0)*xi[i]*eta[j]
    KEij[1,1] = (1.0+(1.0/3.0)*xi[i]*xi[j])*eta[i]*eta[j] + ((1.0-v)/2.0)*(1.0+(1.0/3.0)*eta[i]*eta[j])*xi[i]*xi[j]
    return KEij

def KEdcr(KE, El, IJ):
    return [IJ[El][1], KE]

def GKS(FEMinput):
    #Calculate global stiffness matrix
    IJ = FEMinput['IJ']
    E = FEMinput['TEV']['E']
    V = FEMinput['TEV']['V']
    T = FEMinput['TEV']['T']
    XY = FEMinput['XY']

    KS = np.zeros((XY.shape[0]*2, XY.shape[0]*2))
    XY = ref.arrIndexSort(XY)
    IJ = ref.lstIndexSort(IJ)
    KE = KEsquare(E, V, T)

    KEsl = [KEdcr(KE, EN[0], IJ) for EN in IJ]#List of Element stiffness matrix
    KS = np.matrix(reduce(insKE, KEsl, KS))
    return KS

def EQUNL():
#Equivalent load for node
    pass

def insLmt(KS, LM):
#Insert support condition
    KS = np.matrix(reduce(subInsLmt, LM, KS))
    return KS

def subInsLmt(KS, LM):
#subInsLmt() sub reduce() function, insert limitations for a single node
    NL = LM[0]
    lX = LM[1]
    lY = LM[2]
    if lX == 0:
        KS[:,NL*2] = 0
        KS[NL*2,:] = 0
        KS[NL*2, NL*2] = 1
    if lY == 0:
        KS[:,NL*2+1] = 0
        KS[NL*2+1,:] = 0
        KS[NL*2+1, NL*2+1] = 1
    return KS

def cmpLoad(LDs, NN):
    LD = np.zeros((NN,2))
    reduce(subInsLoad, LDs, LD)
    LD = LD.reshape((-1,1))
    return LD

def subInsLoad(LD, LDsl):
    LD[LDsl[0],:] = LDsl[1:3]
    return LD

def insKEIJ(lGKS, lKE):
#insKE() sub map() function, insert a single block (ij) of KE
#lGKS = [[m,n],KS]
#lKE = [[i,j],KE]
    o=2 #= offset = size of a block = degree of node's freedom
    m = lGKS[0][0]
    n = lGKS[0][1]
    KS = lGKS[1]
    i = lKE[0][0]
    j = lKE[0][1]
    KE = lKE[1]
    KS[m*o: m*o+o, n*o: n*o+o] += KE[i*o: i*o+o, j*o: j*o+o]
    return KS

def insKE(KS, lKE):
#KS() sub reduce() function, insert KE of a single element
#lKE = [[[LableInElement, NodeLable],[0,91],[1,92], ...], KE]
    mn = ref.arrIndexSort(lKE[0])[:,1].T
    ij = ref.arrIndexSort(lKE[0])[:,0].T
    KE = lKE[1]
    map(lambda i: map(lambda j: insKEIJ([[mn[i], mn[j]],KS], [[i,j],KE]), ij), ij)
    return KS

def Bc(r, xi, eta):
    cache = mc.Client()
    if not cache.get('FEMXYQ4.B.'+str(xi)+'.'+str(eta))==None:
        B = cache.get('B.'+str(xi)+'.'+str(eta))
    else:
        B = BTE(r, xi, eta)
        cache.set('FEMXYQ4.B.'+str(xi)+'.'+str(eta), B)
    return B

def BTE(r, xi, eta):
    B = np.zeros((3,8))
    #B0
    B[0,0] = -r+r*eta
    B[1,1] = -r+r*xi
    B[2,0] = B[1,1]
    B[2,1] = B[0,0]
    #B1
    B[0,2] = r-r*eta
    B[1,3] = -r-r*xi
    B[2,2] = B[1,3]
    B[2,3] = B[0,2]
    #B2
    B[0,4] = r+r*eta
    B[1,5] = r+r*xi
    B[2,4] = B[1,5]
    B[2,5] = B[0,4]
    #B3
    B[0,6] = -r-r*eta
    B[1,7] = r-r*xi
    B[2,6] = B[1,7]
    B[2,7] = B[0,6]
    B = B/(4.0*r*r)
    #B = np.matrix(B)
    return np.matrix(B)

def ESIGME(DP, r, parts, E ,v , El, XY, IJ, T=0):
    #DP:np.array([[u, v],[],[],[]])#node0/1/2/3
    step = 2.0/parts
    xis = np.arange(-1,1,step)
    etas = np.arange(-1,1,step)
    DP = DP.reshape((-1,1))
    D = ref.D0(E,v)
    ES = [[ [ref.xietaToXY(El, XY, IJ, xi, eta), ((D*Bc(r, xi, eta))*DP).A1] for eta in etas] for xi in xis]
    return ES

def SIGME(DP, FEMinput, parts):
    cache = mc.Client()
    if not cache.get('FEMXYQ4.S') == None:
        S = cache.get('FEMXYQ4.S')
    else:
        XY = FEMinput['XY']
        IJ = FEMinput['IJ']
        E = FEMinput['TEV']['E']
        V = FEMinput['TEV']['V']
        T = FEMinput['TEV']['T']
        r = ref._r(XY)
        S = [ESIGME(DP[IJ[El[0]][1][:,1],:], r, parts, E, V, El[0], XY, IJ, T) for El in IJ]
        S = ref.flatter(ref.flatter(S))
        cache.set('FEMXYQ4.S', S)
    return S

def fixedLMx(length, start, XY):
#@start [x,y]
    X = start[0]
    Y = start[1]
    r = ref._r(XY)
    startNL = ref.XYToNL(X, Y, XY)
    endNL = ref.XYToNL(X + length, Y, XY)
    if length < 0: step = -1
    if length > 0: step = 1
    LMs = [[N,0,0,0] for N in range(startNL, endNL + step, step)]
    return LMs

def fixedLMy(length, start, XY):
#@start [x,y]
    X = start[0]
    Y = start[1]
    r = ref._r(XY)
    startNL = ref.XYToNL(X, Y, XY)
    endNL = ref.XYToNL(X, Y + length, XY)
    step = ref.XYToNL(0, 2.0*r, XY)
    if length < 0: step*= -1
    LMs = [[N,0,0,0] for N in range(startNL, endNL + step, step)]
    return LMs

def simplyLMx(length, start, XY):
#@start [x,y]
    X = start[0]
    Y = start[1]
    r = ref._r(XY)
    startNL = ref.XYToNL(X, Y, XY)
    endNL = ref.XYToNL(X + length, Y, XY)
    if length < 0: step = -1
    if length > 0: step = 1
    LMs = [[N,1,0,1] for N in range(startNL, endNL + step, step)]
    return LMs

def simplyLMy(length, start, XY):
#@start [x,y]
    X = start[0]
    Y = start[1]
    r = ref._r(XY)
    startNL = ref.XYToNL(X, Y, XY)
    endNL = ref.XYToNL(X, Y + length, XY)
    step = ref.XYToNL(0, 2.0*r, XY)
    if length < 0: step*= -1
    LMs = [[N,0,1,1] for N in range(startNL, endNL + step, step)]
    return LMs


def toimshowtr(arrS, scale, width, height, XY):
    #@width, height: from mesh()
    cache = mc.Client()
    if not cache.get('FEMXYQ4.imgS')==None:
        dataArr = cache.get('FEMXYQ4.imgS')
    else:
        scale_pro = scale / (ref._r(XY)*2.0)
        scaledData = [[[int(round(l[0][0]*scale_pro)),int(round(l[0][1]*scale_pro))], l[1]] for l in arrS]
        dataArr = reduce(ref.coordcopy, scaledData, np.zeros((height*scale,width*scale,3)))
        cache.set('FEMXYQ4.imgS', dataArr)
    return dataArr
