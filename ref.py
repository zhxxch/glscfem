import sae
import numpy as np
import pylibmc as mc
from itertools import chain


arrIndexSort = lambda arr: np.array(sorted(arr, key= lambda n: n[0]))
lstIndexSort = lambda arr: sorted(arr, key= lambda n: n[0])
#IJMHNNSort = lambda arr: np.array(sorted(arr, key= lambda n: n[1]))

flatter = lambda l: list(chain.from_iterable(l))

def arrToCSVComma(row, element):
    row = str(row) + ',' + str(element)
    return str(row)

def arrToCSVReturn(column, row):
    column = str(column) + '<br />' + str(row)
    return str(column)

def arrToCSV(arr):
    csvstr = reduce(arrToCSVReturn, map(lambda r:reduce(arrToCSVComma, r), arr))
    return csvstr

def _r(XY):
    cache = mc.Client()
    if cache.get('r.r')==None:
        r = np.absolute(XY[0,1] - XY[1,1])/2.0
        if r == 0: r = np.absolute(XY[0,2] - XY[1,2])/2.0
        cache.set('r.r', r)
    else:
        r = cache.get('r.r')
    return r

def matsym(m):
#Symmetric matrix
    shape = m.shape[0]
    t = np.zeros(m.shape) + m.T
    t[range(shape), range(shape)] = 0
    sm = m+t
    return sm

def mesh(mwidth, mheight, mr):

    height = np.int_(np.floor(mheight/(2.0*mr)))
    width = np.int_(np.floor(mwidth/(2.0*mr)))
    mesh = dict({'XY':[], 'IJ': []})
    nodeLables = np.arange((height+1) * (width+1)).T
    nodeXC = (nodeLables % (width+1)).T * mr * 2.0
    nodeYC = (np.floor(nodeLables/(width+1))).T * mr * 2.0
    XY = np.zeros(((height+1) * (width+1),3))
    XY[:,0] = nodeLables
    XY[:,1] = nodeXC
    XY[:,2] = nodeYC
    es = lambda e: (width+1)*np.floor(e/(width))+e%width
    #IJ = map(lambda e: [e, np.array([range(4), [es(e), es(e)+1, es(e)+width+2, es(e)+width+1]], np.int64).T], range(height*width))
    IJ = [[e, np.array([range(4), [es(e), es(e)+1, es(e)+width+2, es(e)+width+1]], np.int64).T] for e in range(height*width)]
    mesh['XY'] = XY
    mesh['IJ'] = IJ
    mesh['W'] = width
    mesh['H'] = height
    return mesh

def cmmesh(mwidth, mheight):
#r = 0.005m
    height = np.int_(np.floor(mheight*100))
    width = np.int_(np.floor(mwidth*100))
    mesh = dict({'XY':[], 'IJ': []})
    nodeLables = np.arange((height+1) * (width+1)).T
    nodeXC = (nodeLables % (width+1)).T * 0.01
    nodeYC = (np.floor(nodeLables/(width+1))).T * 0.01
    XY = np.zeros(((height+1) * (width+1),3))
    XY[:,0] = nodeLables
    XY[:,1] = nodeXC
    XY[:,2] = nodeYC
    es = lambda e: (width+1)*np.floor(e/(width))+e%width
    #IJ = map(lambda e: [e, np.array([range(4), [es(e), es(e)+1, es(e)+width+2, es(e)+width+1]], np.int64).T], range(height*width))
    IJ = [[e, np.array([range(4), [es(e), es(e)+1, es(e)+width+2, es(e)+width+1]], np.int64).T] for e in range(height*width)]
    mesh['XY'] = XY
    mesh['IJ'] = IJ
    return mesh

def mmmesh(mwidth, mheight):
#r = 0.0005m
    height = np.int_(mheight*1000)
    width = np.int_(mwidth*1000)
    mesh = dict({'XY':[], 'IJ': []})
    nodeLables = np.arange((height+1) * (width+1)).T
    nodeXC = (nodeLables % (width+1)).T * 0.001
    nodeYC = (np.floor(nodeLables/(width+1))).T * 0.001
    XY = np.zeros(((height+1) * (width+1),3))
    XY[:,0] = nodeLables
    XY[:,1] = nodeXC
    XY[:,2] = nodeYC
    es = lambda e: (width+1)*np.floor(e/(width))+e%width
    IJ = [[e, np.array([range(4), [es(e), es(e)+1, es(e)+width+2, es(e)+width+1]], np.int64).T] for e in range(height*width)]
    mesh['XY'] = XY
    mesh['IJ'] = IJ
    return mesh

def D0(E, v):
    cache = mc.Client()
    if cache.get('D0')==None:
        D = np.array([[1.0,v,0.0],[v,1.0,0.0],[0.0,0.0,((1.0-v*v)/2.0)]])
        D *= E/(1.0-v*v)
        cache.set('D0', D)
    else:
        D = cache.get('D0')
    return np.matrix(D)

def xietaToXY(El, XY, IJ, xi, eta):
#(XI, ETA) -> (X, Y)
    SW = XY[IJ[El][1][0,1], 1:3]
    NE = XY[IJ[El][1][2,1], 1:3]
    X = (NE[0]+SW[0])/2.0 + (NE[0]-SW[0])*xi/2.0
    Y = (NE[1]+SW[1])/2.0 + (NE[1]-SW[1])*eta/2.0
    return np.array([X, Y])

def XYToNL(X, Y, XY):
#(X, Y) -> [ElementLable, NodeLable]
    cache = mc.Client()
    r = _r(XY)
    if not cache.get('XYToNL.xIndex')==None:
        xIndex = cache.get('XYToNL.xIndex')
        yIndex = cache.get('XYToNL.yIndex')
    else:
        xIndex = XY[:,1].reshape((1,-1))
        yIndex = XY[:,2].reshape((1,-1))
        cache.set('XYToNL.xIndex', xIndex)
        cache.set('XYToNL.yIndex', yIndex)
    nodeLable = np.argwhere((np.absolute(xIndex-X)<=r)*(np.absolute(yIndex-Y)<=r))
    return nodeLable[0,1]

def XYToEL(NL, X, Y, XY, IJ):
    cache = mc.Client()
    r = _r(XY)
    if not cache.get('XYToEL.elementIndex')==None:
        elementIndex = cache.get('XYToEL.elementIndex')
    else:
        elementIndex = np.array(flatter(np.array([element[1][:,1].reshape(1,-1) for element in IJ])))
        cache.set('XYToEL.elementIndex', elementIndex)
    elementLables = np.argwhere((elementIndex[:,0]==NL)+(elementIndex[:,1]==NL)+(elementIndex[:,2]==NL)+(elementIndex[:,3]==NL))
    elementLables = flatter(elementLables) # [SW, SE, NW, NE]
    _xi = X - XY[NL][1]
    _eta = Y - XY[NL][2]
    if _xi<=0 and _eta<=0: elementLable = elementLables[0]
    elif _xi>0 and _eta<0: elementLable = elementLables[1]
    elif _xi<0 and _eta>0: elementLable = elementLables[2]
    elif _xi>0 and _eta>0: elementLable = elementLables[3]
    else: elementLable = elementLables[0]
    return elementLable

def toimshowtr(arrS, scale, width, height, XY):
    #@width, height: from mesh()
    cache = mc.Client()
    if not cache.get('imgS')==None:
        dataArr = cache.get('imgS')
    else:
        scale_pro = scale / (_r(XY)*2.0)
        scaledData = [[[int(round(l[0][0]*scale_pro)),int(round(l[0][1]*scale_pro))], l[1]] for l in arrS]
        dataArr = reduce(coordcopy, scaledData, np.zeros((height*scale, width*scale, 3)))
        cache.set('imgS', dataArr)
    return dataArr

def coordcopy(dataArr, data):
    dataArr[data[0][1], data[0][0]] = data[1]
    return dataArr

def KIC(S, theta, a=0.0001):
    #@theta: angle of crack
    #S->toimshowtr->KICTE
    Sxy = S[:,:,0:2]
    thetaS = np.deg2rad(theta-90)
    Ki = np.absolute( 1.1 * np.sqrt(a) * np.inner(Sxy, [np.sin(thetaS), np.cos(thetaS)]) )
    return Ki

