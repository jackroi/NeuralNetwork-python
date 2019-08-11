"""
neuralnetwork.py
Author: Giacomo Rosin
(based on Daniel Shiffman Toy-Neural-Network-JS, https://github.com/CodingTrain/Toy-Neural-Network-JS)

Simple NeuralNetwork library
"""

import math
from matrix import Matrix


class ActivationFunction(object):
  def __init__(self, func, dfunc):
    self.func = func
    self.dfunc = dfunc

sigmoid = ActivationFunction(
  lambda x: 1 / (1 + math.exp(-x)),
  lambda y: y * (1 - y)
)

tanh = ActivationFunction(
  lambda x: math.tanh,
  lambda y: 1 - (y * y)
)


class NeuralNetwork(object):
  def __init__(self, input_nodes, hidden_nodes, output_nodes):
    self.input_nodes = input_nodes
    self.hidden_nodes = hidden_nodes
    self.output_nodes = output_nodes

    self.weights_ih = Matrix(self.hidden_nodes, self.input_nodes).randomize()
    self.weights_ho = Matrix(self.output_nodes, self.hidden_nodes).randomize()

    self.bias_h = Matrix(self.hidden_nodes, 1).randomize()
    self.bias_o = Matrix(self.output_nodes, 1).randomize()

    # TODO set_learning_rate
    # TODO set_activation_ function
