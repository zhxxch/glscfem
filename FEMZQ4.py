import sae
import ref
import pylibmc as mc
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
    'LM': np.array(['NodeLable', 'ThetaXLimit', 'ThetaY....', 'wZ...'])#Limit=0 or else
    '''

def FEM(inputDict):
    #MAIN
    LD = np.matrix(cmpLoad(inputDict['LD'], inputDict['XY'].shape[0]))
    KS = GKS(inputDict)
    KS = insLmt(KS, inputDict['LM'])
    DP = np.linalg.solve(KS, LD)
    DP = np.array(DP.reshape((-1,3)))
    return DP

def KEACM(E, v, T, r):
    KE = np.zeros((12,12))
    k1 = 81.0 - 6.0*v
    k2 = r*r*(48.0-8.0*v)#=k3
    k4 = r*(33.0+12.0*v)#=k5
    k6 = 30.0*r*r*v
    k7 = -36.0+6.0*v
    k8 = r*r*(12.0+8.0*v)
    k9 = r*r*(18.0+2.0*v)#=k18
    k10 = r*12.0*(1.0-v)#=k21
    k11 = r*(33.0-3.0*v)
    k12 = -9.0-6.0*v
    k13 = r*r*(12.0-2.0*v)#=k14
    k15 = 3.0*r*(4.0+v)#=k16
    k17 = -36.0+6.0*v
    k19 = r*r*(12.0+8.0*v)
    k20 = 3.0*r*(11.0-v)

    KE[(0,3,6,9), (0,3,6,9)] = k1
    KE[(1,4,7,10), (1,4,7,10)] = k2
    KE[(2,5,8,11), (2,5,8,11)] = k2
    KE[(1,4), (0,3)] = k4
    KE[(7,10), (6,9)] = -k4
    KE[(5,8), (3,6)] = k4
    KE[(2,11), (0,9)] = -k4
    KE[(2,8), (1,7)] = -k6
    KE[(5,11), (4,10)] = k6
    KE[(3,9), (0,6)] = k7
    KE[(4,10), (1,7)] = k8
    KE[(10,5,7,11), (1,2,4,8)] = k9
    KE[(4,8,6,3),(0,3,5,1)] = k10
    KE[(9,9,11,10),(7,2,0,6)] = -k10
    KE[(3,11), (2,6)] = k11
    KE[(5,9), (0,8)] = -k11
    KE[(6,9), (0,3)] = k12
    KE[(7,8,10,11), (1,2,4,5)] = k13
    KE[(7,10,6,11), (0,3,2,3)] = k15
    KE[(6,9,8,9), (1,4,0,5)] = -k15
    KE[(6,9), (3,0)] = k17
    KE[(8,11), (5,2)] = k19
    KE[(7,10), (3,0)] = k20
    KE[(6,9), (4,1)] = -k20

    KE *= (E*T*T*T)/(360.0*(1.0-v*v)*r*r)
    KE = ref.matsym(KE)

    return np.matrix(KE)

def KEdcr(KE, El, IJ):
    return [IJ[El][1], KE]

def GKS(FEMinput):
    #Calculate global stiffness matrix
    IJ = FEMinput['IJ']
    E = FEMinput['TEV']['E']
    V = FEMinput['TEV']['V']
    T = FEMinput['TEV']['T']
    XY = FEMinput['XY']

    KS = np.zeros((XY.shape[0]*3, XY.shape[0]*3))
    XY = ref.arrIndexSort(XY)
    IJ = ref.lstIndexSort(IJ)

    r = ref._r(XY)
    KE = KEACM(E, V, T, r)
    KEsl = [KEdcr(KE, EN[0], IJ) for EN in IJ]#List of Element stiffness matrix
    KS = np.matrix(reduce(insKE, KEsl, KS))
    return KS

def cntLD(X, Y, XY, IJ, F, r):
#Equivalent load for node
    r = ref._r(XY)
    EL = ref.XYToEL(ref.XYToNL(X, Y, XY), X, Y, XY, IJ)
    N = IJ[EL][1][:,1].reshape(1,-1)[0]
    LD = [[N[0], F*r/8.0, -F*r/8.0, F/4.0], [N[1], F*r/8.0, F*r/8.0, F/4.0], [N[2], -F*r/8.0, F*r/8.0, F/4.0], [N[3], -F*r/8.0, -F*r/8.0, F/4.0]]
    return LD

def areaLD(X, Y, XY, IJ, F, r):
    r = ref._r(XY)
    EL = ref.XYToEL(ref.XYToNL(X, Y, XY), X, Y, XY, IJ)
    N = IJ[EL][1][:,1].reshape(1,-1)[0]
    LD = [[N[0], F*r*r*r/3.0, -F*r*r*r/3.0, F*r*r], [N[1], F*r*r*r/3.0, F*r*r*r/3.0, F*r*r], [N[2], -F*r*r*r/3.0, F*r*r*r/3.0, F*r*r], [N[3], -F*r*r*r/3.0, -F*r*r*r/3.0, F*r*r]]
    return LD

def insLmt(KS, LM):
#Insert support condition
    KS = np.matrix(reduce(subInsLmt, LM, KS))
    return KS

def subInsLmt(KS, LM):
#subInsLmt() sub reduce() function, insert limitations for a single node
    NL = LM[0]
    lX = LM[1]
    lY = LM[2]
    lZ = LM[3]
    if lZ == 0:
        KS[:,NL*3] = 0
        KS[NL*3,:] = 0
        KS[NL*3, NL*3] = 1
    if lX == 0:
        KS[:,NL*3+1] = 0
        KS[NL*3+1,:] = 0
        KS[NL*3+1, NL*3+1] = 1
    if lY == 0:
        KS[:,NL*3++2] = 0
        KS[NL*3+2,:] = 0
        KS[NL*3+2, NL*3+2] = 1
    return KS

def cmpLoad(LDs, NN):
    LD = np.zeros((NN,3))
    reduce(subInsLoad, LDs, LD)
    LD = LD.reshape((-1,1))
    return LD

def subInsLoad(LD, LDsl):
    LD[LDsl[0],:] = LDsl[[3,1,2]]
    return LD

def insKEIJ(lGKS, lKE):
#insKE() sub map() function, insert a single block (ij) of KE
#lGKS = [[m,n],KS]
#lKE = [[i,j],KE]
    o=3 #= offset = size of a block = degree of node's freedom
    m = lGKS[0][0]
    n = lGKS[0][1]
    KS = lGKS[1]
    i = lKE[0][0]
    j = lKE[0][1]
    KE = lKE[1]
    KS[m*o: m*o+o, n*o: n*o+o] += KE[i*o: i*o+o, j*o: j*o+o]
    return KS

def insKE(KS, lKE):
#GKS() sub reduce() function, insert KE of a single element
#lKE = [[[LableInElement, NodeLable],[0,91],[1,92], ...], KE]
    mn = ref.arrIndexSort(lKE[0])[:,1].T
    ij = ref.arrIndexSort(lKE[0])[:,0].T
    KE = lKE[1]
    map(lambda i: map(lambda j: insKEIJ([[mn[i], mn[j]],KS], [[i,j],KE]), ij), ij)
    return KS

def Bc(r, xi, eta):
    cache = mc.Client()
    if not cache.get('FEMZQ4.B.'+str(xi)+'.'+str(eta))==None:
        B = cache.get('FEMZQ4.B.'+str(xi)+'.'+str(eta))
    else:
        B = BTE(r, xi, eta)
        cache.set('FEMZQ4.B.'+str(xi)+'.'+str(eta), B)
    return B

def BTE(r, xi, eta):
    B = np.zeros((3,12))
    #B0
    B[0,0] = -3.0*xi*(1.0-eta)
    B[1,0] = -3.0*eta*(1.0-xi)
    B[2,0] = 3.0*xi*xi + 3.0*eta*eta - 4.0
    #B[0,1] = 0
    B[1,1] = r*(1.0-3.0*eta)*(1.0-xi)
    B[2,1] = r*(3.0*eta*eta - 2.0*eta - 1)
    B[0,2] = -r*(1.0-3.0*xi)*(1.0-eta)
    #B[1,2] = 0
    B[2,2] = -r*(3.0*xi*xi-2.0*xi-1.0)
    #B1
    B[0,3] = 3.0*xi*(1.0-eta)
    B[1,3] = -3.0*eta*(1.0+xi)
    B[2,3] = -3.0*xi*xi - 3.0*eta*eta + 4.0
    B[1,4] = r*(1.0-3.0*eta)*(1.0+xi)
    B[2,4] = -r*(3.0*eta*eta - 2.0*eta - 1)
    B[0,5] = r*(1.0+3.0*xi)*(1.0-eta)
    B[2,5] = -r*(3.0*xi*xi+2.0*xi-1.0)
    #B2
    B[0,6] = 3.0*xi*(1.0+eta)
    B[1,6] = 3.0*eta*(1.0+xi)
    B[2,6] = 3.0*xi*xi + 3.0*eta*eta - 4.0
    B[1,7] = -r*(1.0+3.0*eta)*(1.0+xi)
    B[2,7] = -r*(3.0*eta*eta + 2.0*eta - 1)
    B[0,8] = r*(1.0+3.0*xi)*(1.0+eta)
    B[2,8] = r*(3.0*xi*xi+2.0*xi-1.0)
    #B3
    B[0,9] = -3.0*xi*(1.0+eta)
    B[1,9] = 3.0*eta*(1.0-xi)
    B[2,9] = -3.0*xi*xi - 3.0*eta*eta + 4.0
    B[1,10] = -r*(1.0+3.0*eta)*(1.0-xi)
    B[2,10] = r*(3.0*eta*eta + 2.0*eta - 1)
    B[0,11] = -r*(1.0-3.0*xi)*(1.0+eta)
    B[2,11] = r*(3.0*xi*xi-2.0*xi-1.0)
    B *= 1.0/(4.0*r*r)
    B = np.matrix(B)
    return B

def ESIGME(DP, r, scale, E ,v , El, XY, IJ, T):
    #DP:np.array([[w,thetax,thetay],[],[],[]])#node0/1/2/3
    step = 2.0/scale
    xis = np.arange(-1,1,step)
    etas = np.arange(-1,1,step)
    DP = DP.reshape((-1,1))
    D = ref.D0(E,v)
    ES = [[ [ref.xietaToXY(El, XY, IJ, xi, eta), ((D*Bc(r, xi, eta)*DP).A1)*(T/2.0)] for eta in etas] for xi in xis]
    return ES

def SIGME(DP, FEMinput, scale):
    cache = mc.Client()
    if not cache.get('FEMZQ4.S') == None:
        S = cache.get('FEMZQ4.S')
    else:
        XY = FEMinput['XY']
        IJ = FEMinput['IJ']
        E = FEMinput['TEV']['E']
        V = FEMinput['TEV']['V']
        T = FEMinput['TEV']['T']
        r = ref._r(XY)
        S = [ESIGME(DP[IJ[El[0]][1][:,1],:], r, scale, E, V, El[0], XY, IJ, T) for El in IJ]
        S = ref.flatter(ref.flatter(S))
        #cache.set('FEMZQ4.S', S)
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
    LMs = [[N,0,1,1] for N in range(startNL, endNL + step, step)]
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
    LMs = [[N,1,0,1] for N in range(startNL, endNL + step, step)]
    return LMs


def toimshowtr(arrS, scale, width, height, XY):
    #@width, height: from mesh()
    cache = mc.Client()
    if not cache.get('FEMZQ4.imgS')==None:
        dataArr = cache.get('FEMZQ4.imgS')
    else:
        scale_pro = scale / (ref._r(XY)*2.0)
        scaledData = [[[int(round(l[0][0]*scale_pro)),int(round(l[0][1]*scale_pro))], l[1]] for l in arrS]
        dataArr = reduce(ref.coordcopy, scaledData, np.zeros((height*scale,width*scale,3)))
        cache.set('FEMZQ4.imgS', dataArr)
    return dataArr
