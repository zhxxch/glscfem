import sae
import sys
import os
import ch
import ref
import FEMZQ4
import FEMXYQ4
import numpy as np
import matplotlib
import StringIO
import pylibmc as mc
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sae.ext.shell import ShellMiddleware
cache = mc.Client()

def index(environ, start_response):
    urlpath = ch.urlPathDecode(environ)
    querys = ch.urlQueryDecode(environ)
    status = '200 OK'
    if urlpath[0] == 'img':
        response_headers = [('Content-type', 'image/png; charset=utf-8')]
    elif urlpath[0] == 'csv':
        response_headers = [('Content-type', 'text/html; charset=utf-8')]
    else:
        response_headers = [('Content-type', 'text/plain; charset=utf-8')]

    femzq4imesh = ref.mesh(0.8,1.2, 0.01)
    FEMZQ4i = dict({
        'XY': femzq4imesh['XY'],
        'TEV': dict({
            'T': 0.005,
            'E': 7.17e+10,
            'V': 0.26
        }),
        'IJ': femzq4imesh['IJ'],
        'LD': np.array(FEMZQ4.cntLD(0.05,0.05,femzq4imesh['XY'],femzq4imesh['IJ'],500,ref._r(femzq4imesh['XY']))+[[0,0,0,0]]),
        'LM': np.array(FEMZQ4.fixedLMx(0.8,[0.0,0.0],femzq4imesh['XY'])+FEMZQ4.fixedLMx(0.8,[0.0,1.2],femzq4imesh['XY'])+FEMZQ4.fixedLMy(1.2,[0.0,0.0],femzq4imesh['XY'])+FEMZQ4.fixedLMy(1.2,[0.8,0.0],femzq4imesh['XY']))
        })
    
    DP = FEMZQ4.FEM(FEMZQ4i)
    S = FEMZQ4.SIGME(DP, FEMZQ4i, 3)
    #plt.imsave(imgdata, S, format="png")
    
    imgS = ref.toimshowtr(S, 3, femzq4imesh['W'], femzq4imesh['H'], femzq4imesh['XY'])
    
    Kic = ref.KIC(imgS, 45)
    
    criterion = Kic - 0.1e+6
    cr = np.array(criterion > 0, dtype=int)
    imgdata = StringIO.StringIO()
    plt.imsave(imgdata, cr, format="png", origin='lower', cmap='binary')
    imgdata.seek(0)  # rewind the data
    start_response(status, response_headers)
    return [imgdata.getvalue()]
    '''
    start_response(status, response_headers)
    return [ref.arrToCSV(Kic)]
    '''
application = sae.create_wsgi_app(ShellMiddleware(index,'XXX'))
