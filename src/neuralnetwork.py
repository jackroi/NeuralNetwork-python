"""
neuralnetwork.py
Author: Giacomo Rosin
(based on Daniel Shiffman Toy-Neural-Network-JS, https://github.com/CodingTrain/Toy-Neural-Network-JS)

Simple NeuralNetwork library
"""

import math, json, random
from matrix import Matrix

# TODO error checking

class ActivationFunction(object):
  def __init__(self, func, dfunc):
    """
    ActivationFunction constructor: creates an activation function.

    Args:
      func: The activation function.
      dfunc: The derivative of the activation function.

    Returns:
      Returns a new activation function object.
    """

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
  def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=0.1, activation_function=sigmoid):
    """
    NeuralNetwork constructor: creates a neural network.

    Args:
      input_nodes: The number of input_nodes.
      hidden_nodes: The number of hidden_nodes.
      output_nodes: The number of output_nodes.
      learning_rate: The learning rate value (default 0.1).
      activation_function: The activation function (default sigmoid).

    Returns:
      Returns a new neural network object.
    """

    self.input_nodes = input_nodes
    self.hidden_nodes = hidden_nodes
    self.output_nodes = output_nodes

    self.weights_ih = Matrix(self.hidden_nodes, self.input_nodes).randomize()
    self.weights_ho = Matrix(self.output_nodes, self.hidden_nodes).randomize()

    self.bias_h = Matrix(self.hidden_nodes, 1).randomize()
    self.bias_o = Matrix(self.output_nodes, 1).randomize()

    self.learning_rate = learning_rate
    self.activation_function = activation_function


  def set_learning_rate(self, lr=0.1):
    """
    Sets the learning rate of the neural network.

    Args:
      lr: The learning rate value (default 0.1).

    Returns:
      Returns the resulting neural network.
    """

    self.learning_rate = lr
    return self


  def set_activation_function(self, func=sigmoid):
    """
    Sets the activation function of the neural network.

    Args:
      func: The activation function value (default sigmoid).

    Returns:
      Returns the resulting neural network.
    """

    self.activation_function = func
    return self


  def predict(self, input_list):
    """
    Calculates the outputs using the given inputs.

    Args:
      input_list: List of numbers containing the input values.

    Returns:
      Returns a list of numbers containing the outputs.
    """

    inputs = Matrix.from_list(input_list)
    hidden = Matrix.dot_product(self.weights_ih, inputs)
    hidden.add_matrix(self.bias_h)
    hidden.map(self.activation_function.func)

    outputs = Matrix.dot_product(self.weights_ho, hidden)
    outputs.add_matrix(self.bias_o)
    outputs.map(self.activation_function.func)

    return outputs.to_list()


  def train(self, input_list, target_list):
    """
    Trains the neural network using the given input_list and target_array.

    Args:
      input_list: List of numbers containing the input values.
      target_list: List of numbers containing the target values.

    Returns:
      Returns the resulting neural network.
    """

    inputs = Matrix.from_list(input_list)
    hidden = Matrix.dot_product(self.weights_ih, inputs)
    hidden.add_matrix(self.bias_h)
    hidden.map(self.activation_function.func)

    outputs = Matrix.dot_product(self.weights_ho, hidden)
    outputs.add_matrix(self.bias_o)
    outputs.map(self.activation_function.func)

    targets = Matrix.from_list(target_list)

    # Calculate the error
    output_errors = targets.sub_matrix(outputs, False)

    # Calculate gradient
    gradients = outputs.map(self.activation_function.dfunc, False)
    gradients.mul_matrix(output_errors)
    gradients.mul_scalar(self.learning_rate)

    # Calculate deltas
    hidden_t = Matrix.transpose(hidden)
    weights_ho_deltas = Matrix.dot_product(gradients, hidden_t)

    # Adjust the weights and the biases by the calculated deltas
    self.weights_ho.add_matrix(weights_ho_deltas)
    self.bias_o.add_matrix(gradients)


    # Calculate the hidden layer errors
    weights_ho_t = Matrix.transpose(self.weights_ho)
    hidden_errors = Matrix.dot_product(weights_ho_t, output_errors)

    # Calculate hidden gradient
    hidden_gradients = hidden.map(self.activation_function.dfunc, False)
    hidden_gradients.mul_matrix(hidden_errors)
    hidden_gradients.mul_scalar(self.learning_rate)

    # Calculate input -> hidden deltas
    inputs_t = Matrix.transpose(inputs)
    weights_ih_deltas = Matrix.dot_product(hidden_gradients, inputs_t)

    # Adjust the weights and the biases by the calculated deltas
    self.weights_ih.add_matrix(weights_ih_deltas)
    self.bias_h.add_matrix(hidden_gradients)

    return self


  def copy(self):
    """
    Builds a copy of the neural network.

    Returns:
      Returns a new neural network object.
    """

    nn = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes, self.learning_rate, self.activation_function)
    nn.weights_ih = self.weights_ih.copy()
    nn.weights_ho = self.weights_ho.copy()
    nn.bias_h = self.bias_h.copy()
    nn.bias_o = self.bias_o.copy()

    return nn


  def mutate(self, func=None, rate=0.1):
    """
    Mutates the neural network using the given function or the given rate (with gaussian distribution).

    Args:
      func: The mutation function (must include the rate of mutation).
      rate: The rate of mutation.

    Returns:
      Returns the resulting neural network.
    """

    if not func:
      func = lambda val: val + random.gauss(0, 0.1) if random.uniform(0, 1) < rate else val

    self.weights_ih.map(func)
    self.weights_ho.map(func)
    self.bias_h.map(func)
    self.bias_o.map(func)

    return self


  def serialize(self):      # TODO test and implement
    """
    Serializes the neural network.

    Returns:
      Returns the serialized neural network.
    """

    return json.dumps(self)


  @staticmethod
  def deserialize(data):    # TODO test and implement
    """
    Deserializes the given data.

    Args:
      data: The data to deserialize (stringified json or python dict)

    Returns:
      Returns the resulting neural network.
    """

    if type(data) == str:
      data = json.loads(data)

    #nn = NeuralNetwork(data["input_nodes"], data["hidden_nodes"], data["output_nodes"])

