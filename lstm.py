import logging
import theano
from theano import function
import theano.tensor as tt

from utils import Parameters, initial_weights, ones





class LSTM(object):
    input_size = 2
    n_cells = 3
    clf_size = 1
    rprop_plus = 1.4
    rprop_minus = 0.5

    test_x = [
        [
            [0, 1],
            [1, 0],
            [1, 1],
            [1, 1],
        ],
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ]
    ]

    test_y = [
        0,
        1,
        0,
        0
    ]

    test_s0 = [[0, 0, 0]] * len(test_x[0])

    def __init__(self, learning_rate):
        self.params = Parameters()
        self.learning_rate = learning_rate

        self.s0 = tt.matrix('s0')
        self.s0.tag.test_value = self.test_s0
        self.params.w_x = initial_weights(self.n_cells, self.input_size)
        self.params.w_f = initial_weights(self.n_cells, self.input_size)
        self.params.w_i = initial_weights(self.n_cells, self.input_size)

        self.params.w_clf = initial_weights(self.clf_size, self.n_cells)
        self.params.b_clf = initial_weights(self.clf_size)

    def build_train(self):
        # Build training function.
        # data_x[t,i,:] = input number i at time t
        data_x = tt.tensor3('data_x')
        data_x.tag.test_value = self.test_x

        # Desired outputs.
        data_y = tt.vector('data_y')
        data_y.tag.test_value = self.test_y

        def process_input(x):
            """
            Go in time over the input sequence encoded by x, return final state.

            Args:
                x: A 3 dimensional tensor [time, example_id, example_dims].

            Return: Matrix with final states for all examples.
            """
            def step(x, s):
                """

                Args:
                    x: A 2 dimensional matrix with input [example_id,
                        example_dims].
                    s: A 2 dimensional matrix with current state [example_id,
                        state_dim].
                """
                c = tt.tanh(tt.tensordot(x, self.params.w_x,
                                         [[1], [1]]))
                g_f =  tt.nnet.sigmoid(tt.tensordot(x, self.params.w_f,
                                                    [[1], [1]]))
                g_i =  tt.nnet.sigmoid(tt.tensordot(x, self.params.w_i,
                                                    [[1], [1]]))

                new_cell = c * g_i
                new_state = s * g_f

                return new_cell + new_state

            # For each timestamp compute the new state. Result should be a 3
            # dimensional tensor.
            res, updates = theano.scan(
                fn=step,
                outputs_info=[self.s0],
                n_steps=x.shape[0],
                sequences=[x]
            )
            res = res[-1]

            return res

        #final_state = process_input(data_x)
        #return function([data_x, self.s0], final_state)

        def clf_state(curr_s):
            """
            For each example, get the classifier output.

            Args:
                curr_s: Matrix of final states for each example [example_id,
                    example_dims.
            """
            return tt.nnet.sigmoid(
                tt.tensordot(self.params.w_clf, curr_s, [[1], [1]]) +
                        self.params.b_clf.dimshuffle(0, 'x'))

        def classify(data_x):
            final_state = process_input(data_x)
            clf = clf_state(final_state)
            return clf

        def build_loss(data_x, y):
            clf = classify(data_x)
            return tt.mean((clf - y)**2)

        def validate(data_x, y):
            return (classify(data_x), build_loss(data_x, y))

        total_loss = build_loss(data_x, data_y)

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


        self.train_step = function(
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

        #x = tt.matrix('x')
        self.classify = function([self.s0, data_x], classify(data_x))
        self.validation_loss = function([self.s0, data_x, data_y], validate(
            data_x, data_y))


if __name__ == '__main__':
    import os
    #os.environ['THEANO_FLAGS'] = 'optimizer=None,exception_verbosity=high,\
    # linker=py'
    os.environ['THEANO_FLAGS'] = 'mode=FAST_COMPILE,floatX=float32'

    theano.config.compute_test_value = 'raise'
    lstm = LSTM(0.1)
    lstm.build_train()
    print lstm.train_step(lstm.test_s0, lstm.test_x, lstm.test_y)
    print lstm.train_step(lstm.test_s0, lstm.test_x, lstm.test_y)
    print lstm.train_step(lstm.test_s0, lstm.test_x, lstm.test_y)
    print lstm.train_step(lstm.test_s0, lstm.test_x, lstm.test_y)
    #print fn_final_state(test_x, s0)
