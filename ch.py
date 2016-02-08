import sae
import urlparse
import ref
import numpy as np
import time
import hashlib
import math

class Router(object):  
    def __init__(self):  
        self.path_info = {}  
    def route(self, environ, start_response):  
        application = self.path_info[environ['PATH_INFO']]  
        return application(environ, start_response)  
    def __call__(self, path):  
        def wrapper(application):  
            self.path_info[path] = application  
        return wrapper

def urlPathDecode(environ):
    paths = filter(None,environ['PATH_INFO'].split('/'))
    return paths

def urlQueryDecode(environ):
    return dict([(k,v[0]) for k,v in urlparse.parse_qs(environ['QUERY_STRING']).items()])

def paramMapping(url, query):
    pass

def urlToken():
    struct_time = time.gmtime()
    year = struct_time.tm_year
    month = struct_time.tm_mon
    date = struct_time.tm_mday
    hour = struct_time.tm_hour
    minute = struct_time.tm_min
    second = struct_time.tm_sec
    a = math.floor((14.0-month)/12.0)
    y = year + 4800.0 - a
    m = month + 12.0*a - 3.0
    JDN = -0.5 + date + math.floor((153.0*m + 2.0)/5.0) + 365.0*y + math.floor(y/4.0) - math.floor(y/100.0) + math.floor(y/400.0) - 32045.0 + hour/24.0 + minute/1440.0 + second/86400.0
    srJDN = str(round(JDN, 4))
    md5JDN = hashlib.md5(srJDN).hexdigest().upper() + srJDN
    shaJDN = hashlib.sha1(md5JDN).hexdigest() + md5JDN
    token =  hashlib.md5(shaJDN).hexdigest()
    return token

def opid():
    struct_time = time.gmtime()
    year = struct_time.tm_year
    month = struct_time.tm_mon
    date = struct_time.tm_mday
    hour = struct_time.tm_hour
    minute = struct_time.tm_min
    second = struct_time.tm_sec
    a = math.floor((14.0-month)/12.0)
    y = year + 4800.0 - a
    m = month + 12.0*a - 3.0
    JDN = -0.5 + date + math.floor((153.0*m + 2.0)/5.0) + 365.0*y + math.floor(y/4.0) - math.floor(y/100.0) + math.floor(y/400.0) - 32045.0 + hour/24.0 + minute/1440.0 + second/86400.0
    srJDN = str(JDN)
    return srJDN
