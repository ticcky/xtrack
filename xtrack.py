import os
#os.environ['THEANO_FLAGS'] = 'optimizer=None,exception_verbosity=high,\
# linker=py'
os.environ['THEANO_FLAGS'] = 'mode=FAST_COMPILE,floatX=float32'

import climate
import cPickle as pickle
import logging
import numpy as np
import random
import theano
import theano.gradient
from theano import (function, tensor as tt)
from theano import printing


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
    rprop_plus = 1.4
    rprop_minus = 0.5

    def __init__(self, learning_rate):
        self.params = Parameters()
        self.learning_rate = learning_rate

        self.s0 = tt.vector('s0')
        self.params.w_x = initial_weights(self.n_cells, self.input_size)
        self.params.w_f = initial_weights(self.n_cells, self.input_size)
        self.params.w_i = initial_weights(self.n_cells, self.input_size)

        self.params.w_clf = initial_weights(self.clf_size, self.n_cells)
        self.params.b_clf = initial_weights(self.clf_size, 1)





    def build_train(self):
        # Build training function.
        data_x = tt.tensor3('data_x')
        data_y = tt.vector('data_y')

        # Input to this function is a matrix with a sequence of inputs.
        def process_input(x_input):
            def step(x, s):
                c = tt.tanh(tt.tensordot(self.params.w_x, x,
                                         [[1], [0]]))
                g_f =  tt.nnet.sigmoid(tt.tensordot(self.params.w_f, x,
                                                    [[1], [0]]))
                g_i =  tt.nnet.sigmoid(tt.tensordot(self.params.w_i, x,
                                                    [[1], [0]]))

                return c * g_i + s * g_f

            res, updates = theano.scan(
                fn=step,
                outputs_info=self.s0,
                n_steps=x_input.shape[0],
                sequences=[x_input]
            )
            res = res[-1]

            return res

        def clf_state(curr_s):
            return tt.nnet.sigmoid(
                tt.tensordot(self.params.w_clf, curr_s, [[1], [0]]) +
                        self.params.b_clf)

        def classify(x):
            curr_s = process_input(x)
            clf = clf_state(curr_s)
            return clf

        def build_loss(x, y):
            clf = classify(x)
            return tt.mean((clf - y)**2)

        #f = function([self.s0, data_x, data_y], build_loss(0, data_x, data_y))
        #import ipdb; ipdb.set_trace()

        total_loss, _ = theano.scan(
            fn=build_loss,
            sequences=[data_x, data_y]
        )

        total_loss = tt.mean(total_loss)


        self.shapes = []
        grads = []
        grads_history = []
        self.grads_rprop = grads_rprop = []
        grads_rprop_new = []
        for param in self.params.values():
            logging.info('param %s', param.name)

            shape = param.shape.eval()
            self.shapes.append(shape)
            grad = tt.grad(total_loss, wrt=param)
            grads.append(grad)

            # Save gradients histories for RProp.
            grad_hist = theano.shared(ones(shape), name="%s_hist" % param)
            grads_history.append(
                grad_hist
            )

            # Create variables where rprop rates will be stored.
            grad_rprop = theano.shared(ones(shape) * self.learning_rate,
                                       name="%s_rprop" % param)
            grads_rprop.append(grad_rprop)

            # Compute the new RProp coefficients.
            rprop_sign = tt.sgn(grad_hist * grad)
            grad_rprop_new = grad_rprop * (
                tt.eq(rprop_sign, 1) * self.rprop_plus
                + tt.neq(rprop_sign, 1) * self.rprop_minus
            )
            grads_rprop_new.append(grad_rprop_new)


        self._train = function(
            inputs=[self.s0, data_x, data_y],
            outputs=[total_loss],
            updates=[
                        # Update parameters according to the RProp update rule.
                        (p, p - rg * tt.sgn(g)) for p, g, rg in zip(
                    self.params.values(),
                    grads,
                    grads_rprop_new)
                    ] + [
                        # Save current gradient for the next step..
                        (hg, g) for hg, g in zip(grads_history, grads)
                    ] + [
                        # Save the new rprop grads.
                        (rg, rg_new) for rg, rg_new in zip(grads_rprop, grads_rprop_new)
                    ]
        )

        x = tt.matrix('x')
        self.classify = function([self.s0, x], classify(x))


def update(params, grads, lr):
    for i, p in enumerate(params.values()):
        p.set_value(p.get_value() - lr * grads[i])


def generate_data(n):
    res_x = []
    res_y = []
    for _ in range(n):
        n_steps = 5 #random.randint(2, 5)
        # Toss a coin whether this is a positive or negative example.
        if random.random() > 0.5:
            # Positive.
            x = []
            el = [[random.random(), random.random()]]
            x = el * n_steps
            y = 1.0  #np.asarray([1.0], dtype=np.float32)
            res_x.append(np.asarray(x, dtype=np.float32))
            res_y.append(y)
        else:
            # Negative.
            x = []
            for _ in range(n_steps):
                x.append([random.random(), random.random()])
            y = 0.0  #np.asarray([0.0], dtype=np.float32)
            res_x.append(np.asarray(x, dtype=np.float32))
            res_y.append(y)

    return res_x, np.asarray(res_y, dtype=np.float32)





def main():
    n_steps = 5
    n_epochs = 1000
    # Generate some data.
    data_x, data_y = generate_data(100)
    print data_x
    print data_y
    validation_data_x, validation_data_y = generate_data(20)

    lstm = LSTM(
        learning_rate=0.1
    )
    lstm.build_train()

    s0 = np.zeros((lstm.n_cells, ), dtype=np.float32)
    for e in range(1000):
        (loss, ) = lstm._train(s0, data_x, data_y)
        logging.info("Epoch #%d: loss(%.5f)" % (e, loss, ))
        for i, (x, y) in enumerate(zip(validation_data_x, validation_data_y)):
            logging.info("%d: lbl(%.2f) clf(%.2f)" % (i, y, lstm.classify(
                s0, x), ))

    return

    clf = lstm.build_clf_model(lstm.process_input())
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