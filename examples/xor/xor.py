import random
from src.neuralnetwork import NeuralNetwork   # ! this doesn't work

training_data = [
  {
    "inputs": [0, 0],
    "outputs": [0]
  },
  {
    "inputs": [0, 1],
    "outputs": [1]
  },
  {
    "inputs": [1, 0],
    "outputs": [1]
  },
  {
    "inputs": [1, 1],
    "outputs": [0]
  }
]


nn = NeuralNetwork(2, 4, 1)

for i in range(50000):
  data = random.choice(training_data)
  nn.train(data["inputs"], data["outputs"])


print("[0, 0] ->", nn.predict([0, 0]))
print("[0, 1] ->", nn.predict([0, 1]))
print("[1, 0] ->", nn.predict([1, 0]))
print("[1, 1] ->", nn.predict([1, 1]))
