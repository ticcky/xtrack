import os
#os.environ['THEANO_FLAGS'] = 'optimizer=None,exception_verbosity=high,\
# linker=py'
os.environ['THEANO_FLAGS'] = 'optimizer=None,exception_verbosity=high'

import climate
import cPickle as pickle
import logging
import numpy as np
import random
import theano
import theano.gradient
from theano import (function, tensor as tt)
from theano import printing


def initial_weights(*argv):
    return np.asarray(
        np.random.uniform(
            low  = -np.sqrt(6. / sum(argv)),
            high =  np.sqrt(6. / sum(argv)),
            size =  argv
        ),
        dtype=theano.config.floatX
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


class LSTM:
    input_size = 2
    n_cells = 3
    clf_size = 1

    def __init__(self):
        self.params = Parameters()

    def build_accum_model(self):
        self.params.w_x = initial_weights(self.n_cells, self.input_size)
        self.params.w_f = initial_weights(self.n_cells, self.input_size)
        self.params.w_i = initial_weights(self.n_cells, self.input_size)

        self.s0 = tt.vector('s0')

        def step(x, s):
            c = tt.tanh(tt.tensordot(self.params.w_x, x,
                                     [[1], [0]]))
            g_f =  tt.nnet.sigmoid(tt.tensordot(self.params.w_f, x,
                                                [[1], [0]]))
            g_i =  tt.nnet.sigmoid(tt.tensordot(self.params.w_i, x,
                                                [[1], [0]]))

            return c * g_i + s * g_f

        self.x = tt.matrix('x')
        self.n_steps = tt.iscalar('n_steps')

        res, updates = theano.scan(
            fn=step,
            n_steps=self.n_steps,
            outputs_info=self.s0,
            sequences=[self.x]
        )
        res = res[-1]

        return res + 0*self.n_steps

    def build_clf_model(self, curr_s):
        self.params.w_clf = initial_weights(self.clf_size, self.n_cells)
        self.params.b_clf = initial_weights(self.clf_size, 1)
        return tt.nnet.sigmoid(
            tt.tensordot(self.params.w_clf, curr_s, [[1], [0]]) +
                    self.params.b_clf)

    def build_loss(self, clf):
        tgt = self.tgt = tt.vector('tgt')

        return tt.mean((clf - tgt)**2)


def update(params, grads, lr):
    for i, p in enumerate(params.values()):
        p.set_value(p.get_value() - lr * grads[i])


def generate_data(n):
    res = []
    for _ in range(n):
        n_steps = random.randint(2, 5)
        # Toss a coin whether this is a positive or negative example.
        if random.random() > 0.5:
            # Positive.
            x = []
            el = [[random.random(), random.random()]]
            x = el * n_steps
            y = np.asarray([1.0], dtype=np.float32)
            res.append((np.asarray(x, dtype=np.float32), y, n_steps))
        else:
            # Negative.
            x = []
            for _ in range(n_steps):
                x.append([random.random(), random.random()])
            y = np.asarray([0.0], dtype=np.float32)
            res.append((np.asarray(x, dtype=np.float32), y, n_steps))

    return res





def main():
    n_steps = 5
    n_epochs = 1000
    # Generate some data.
    data = generate_data(100)
    validation_data = generate_data(20)

    lstm = LSTM()
    clf = lstm.build_clf_model(lstm.build_accum_model())
    f_clf = function([lstm.s0, lstm.n_steps, lstm.x], clf)
    """
    x = np.ones((5, lstm.input_size), dtype=np.float32)
    x[0,0] = 0
    x[1,0] = 1
    x[2,0] = 2
    x[3,0] = 3
    x[4,0] = 4
    print f_clf(
        np.zeros((lstm.n_cells, ), dtype=np.float32),
        5,
        x
    )
    return"""


    loss = lstm.build_loss(clf)
    f_loss = function([lstm.s0, lstm.n_steps, lstm.x, lstm.tgt], loss)

    input = np.ndarray((lstm.input_size), dtype=np.float32)
    input[:] = 1.0

    loss_prime = theano.grad(loss, wrt=lstm.params.values())
    f_loss_prime = function([lstm.s0, lstm.n_steps, lstm.x, lstm.tgt],
                            loss_prime)

    for e in range(n_epochs):
        logging.info("Epoch #%d" % e)
        g = {}

        total_loss = 0.0
        for x, y, n_steps in data:
            g_point = f_loss_prime(
                np.zeros((lstm.n_cells, ), dtype=np.float32),
                n_steps,
                x,
                y)
            total_loss += f_loss(
                np.zeros((lstm.n_cells, ), dtype=np.float32),
                n_steps,
                x,
                y) * 1.0 / len(data)

            for i in range(len(g_point)):
                if not i in g:
                    g[i] = np.zeros_like(g_point[i], dtype=np.float32)

            for i in range(len(g_point)):
                g[i] += g_point[i] #* 1.0 / len(data)

        validation_loss = 0.0
        for i, (x, y, n_steps) in enumerate(validation_data):
            args = [
                np.zeros((lstm.n_cells, ), dtype=np.float32),
                n_steps,
                x,
                y
            ]
            validation_loss += f_loss(*args) * 1.0 / len(validation_data)

            if e % 50 == 0:
                logging.info("%d: tgt(%.0f) clf(%.2f)" % (i, y, f_clf(*args[
                                                                       :-1])))

        logging.info("train_loss(%.5f) valid_loss(%.5f)" % (
            total_loss, validation_loss))

        update(lstm.params, g, 0.1)


if __name__ == '__main__':
    climate.call(main)