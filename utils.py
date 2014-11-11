import pickle
import numpy as np
import theano


def ones(shape):
    return np.ones(shape).astype(np.float32)


def zeros(shape):
    return np.zeros(shape).astype(np.float32)


def initial_weights(*argv):
    return np.asarray(
        np.random.uniform(
            low  = -np.sqrt(6. / sum(argv)),
            high =  np.sqrt(6. / sum(argv)),
            size =  argv
        ),
        dtype=np.float32
    )



class Parameters():
    def __init__(self):
        self.__dict__['params'] = {}

    def __setattr__(self,name,array):
        params = self.__dict__['params']
        if name not in params:
            params[name] = theano.shared(
                value = np.asarray(
                    array,
                    dtype = np.float32
                ),
                name = name
            )
        else:
            print "%s already assigned"%name
            params[name].set_value(np.asarray(
                    array,
                    dtype = np.float32
                ))
    def __setitem__(self,name,array):
        self.__setattr__(name,array)
    def __getitem__(self,name):
        return self.__getattr__(name)

    def __getattr__(self,name):
        params = self.__dict__['params']
        return self.params[name]

    def remove(self,name):
        del self.__dict__['params'][name]


    def values(self):
        params = self.__dict__['params']
        return params.values()

    def save(self,filename):
        params = self.__dict__['params']
        pickle.dump({p.name:p.get_value() for p in params.values()},open(filename,'wb'),2)

    def load(self,filename):
        params = self.__dict__['params']
        loaded = pickle.load(open(filename,'rb'))
        for k in params:
            params[k].set_value(loaded[k])
