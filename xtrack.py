import climate
import theano
import theano.gradient
from theano import (function, tensor as T)

from theano_toolkit.parameters import Parameters


class LSTM:
    def __init__(self):
        self.params = Parameters()



def main():
    LSTM()


if __name__ == '__main__':
    climate.call(main)