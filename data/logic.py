import numpy as np

xor_fcn = lambda x : (x.sum(axis=1, keepdims=True)==1).astype(np.float16)
or_fcn  = lambda x : (x.sum(axis=1, keepdims=True) >0).astype(np.float16)
and_fcn = lambda x : (x.sum(axis=1, keepdims=True)==2).astype(np.float16)

def make(n, type):
    if type == 'xor':
        f = xor_fcn
    if type == 'or':
        f = or_fcn
    if type == 'and':
        f = and_fcn
    data = np.random.randint(2, size=(n,2))
    targets = f(data)
    return data, targets

def make_xor(n):
    return make(n,'xor')
def make_or(n):
    return make(n,'or')
def make_and(n):
    return make(n,'and')
