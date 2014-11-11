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

from lstm import LSTM

def update(params, grads, lr):
    for i, p in enumerate(params.values()):
        p.set_value(p.get_value() - lr * grads[i])


def generate_data(n, tricky=False):
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
            if tricky:
                n_random = 1
                n_same = n_steps - n_random
            else:
                n_random = n_steps
                n_same = 0

            for _ in range(n_random):
                x.append([random.random(), random.random()])


            el = [[random.random(), random.random()]]
            x = x + el * n_same

            y = 0.0  #np.asarray([0.0], dtype=np.float32)
            res_x.append(np.asarray(x, dtype=np.float32))
            res_y.append(y)

    return zip(*res_x), np.asarray(res_y, dtype=np.float32)





def main():
    n_steps = 5
    n_epochs = 1000
    n_data = 1000
    n_valid_data = 20
    # Generate some data.
    data_x, data_y = generate_data(n_data, tricky=True)
    logging.info(("First 10 data samples:"))
    logging.info(zip(*data_x)[:10])
    logging.info(data_y[:10])

    validation_data_x, validation_data_y = generate_data(n_valid_data,
                                                         tricky=True)

    lstm = LSTM(
        learning_rate=0.1
    )
    lstm.build_train()

    s0 = np.zeros((n_data, lstm.n_cells, ), dtype=np.float32)
    s0_valid = np.zeros((n_valid_data, lstm.n_cells, ),
                        dtype=np.float32)
    for e in range(1000):
        (loss, ) = lstm.train_step(s0, data_x, data_y)

        validation_data_est, validation_loss = lstm.validation_loss(
                s0_valid, validation_data_x, validation_data_y)
        logging.info("Epoch #%d: loss(%.5f) valid_loss(%.5f)" % (e, loss, validation_loss))
        res_seq = enumerate(zip(validation_data_est[0], validation_data_y))
        for i, (y_est, y_t) in res_seq:
            logging.info("%d: lbl(%.2f) clf(%.2f)" % (i, y_est, y_t, ))


        #for i, (x, y) in enumerate(zip(validation_data_x, validation_data_y)):


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