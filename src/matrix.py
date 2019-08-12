"""
matrix.py
Author: Giacomo Rosin
(based on Daniel Shiffman Toy-Neural-Network-JS, https://github.com/CodingTrain/Toy-Neural-Network-JS)

Matrix library
"""

import random, json


class Matrix(object):
  def __init__(self, rows, cols):
    """
    Matrix constructor: create a (rows x cols) matrix, initialized to 0.

    Args:
      rows: The number of rows of the matrix.
      cols: The number of columns of the matrix.

    Returns:
      Returns a new matrix object.
    """

    self.rows = rows
    self.cols = cols
    self.data = [[0 for _ in range(cols)] for _ in range(rows)]


  def copy(self):
    """
    Builds a copy of the matrix.

    Returns:
      Returns a new matrix object.
    """

    m = Matrix(self.rows, self.cols)

    for i in range(self.rows):
      for j in range(self.cols):
        m.data[i][j] = self.data[i][j]

    return m


  @staticmethod
  def from_2d_list(rows, cols, l):
    """
    Creates a (rows x cols) matrix, initializing the values through list of list of numbers.

    Args:
      rows: The number of rows of the matrix.
      cols: The number of columns of the matrix.
      l: The list of list of numbers, used to initialize the values of the matrix.

    Returns:
      Returns a new matrix object.
    """

    m = Matrix(rows, cols)
    m.data = l

    return m


  @staticmethod
  def from_flat_list(rows, cols, l):
    """
    Creates a (rows x cols) matrix, initializing the values through a list of numbers.

    Args:
      rows: The number of rows of the matrix.
      cols: The number of columns of the matrix.
      l: The list of numbers, used to initialize the values of the matrix.

    Returns:
      Returns a new matrix object.

    Raises:
      ValueError: if rows * cols != len(l)
    """

    if rows * cols != len(l):
      raise ValueError("Incompatible list of list")

    data = [[l[cols * r + c] for c in range(cols)] for r in range(rows)]
    m = Matrix(rows, cols)
    m.data = data

    return m


  @staticmethod
  def from_list(l):
    """
    Builds a new column matrix (len(l) x 1) that contains the elements of the given list.

    Args:
      rows: The number of rows of the matrix.
      cols: The number of columns of the matrix.
      l: The list of numbers, used to initialize the values of the matrix.

    Returns:
      Returns a new matrix object.
    """

    return Matrix.from_flat_list(len(l), 1, l)


  def to_list(self):
    """
    Builds a new list of numbers by flattening the matrix.

    Returns:
      Returns a list of numbers.
    """

    return [self.data[i // self.cols][i % self.cols] for i in range(self.rows * self.cols)]


  def randomize(self, _min=-1.0, _max=1.0):
    """
    Randomizes the values contained in the matrix, giving a random value between min and max.

    Args:
      _min: The minimum possible number (default -1.0).
      _max: The maximum possible number (default 1.0).

    Returns:
      Returns the modified matrix.
    """

    for r in range(self.rows):
      for c in range(self.cols):
        self.data[r][c] = random.uniform(_min, _max)

    return self


  def add_matrix(self, m, in_place=True):
    """
    Sums a matrix to this one.

    Args:
      m: The matrix to add.
      in_place:
        - True: modifies the original matrix (default)
        - False: builds a new matrix

    Returns:
      Returns the resulting matrix.

    Raises:
      ValueError: if a.rows != b.rows or a.cols != b.cols
    """

    a = self
    b = m
    if a.rows != b.rows or a.cols != b.cols:
      raise ValueError("Matrixes not compatible")

    return a.mapi(lambda el, r, c: el + b.data[r][c], in_place)


  def add_scalar(self, n, in_place=True):
    """
    Sums a scalar to this matrix.

    Args:
      m: The scalar value to add.
      in_place:
        - True: modifies the original matrix (default)
        - False: builds a new matrix

    Returns:
      Returns the resulting matrix.
    """

    return self.map(lambda el: el + n, in_place)


  def sub_matrix(self, m, in_place=True):
    """
    Subtracts a matrix to this one.

    Args:
      m: The matrix to subtract.
      in_place:
        - True: modifies the original matrix (default)
        - False: builds a new matrix

    Returns:
      Returns the resulting matrix.

    Raises:
      ValueError: if a.rows != b.rows or a.cols != b.cols
    """

    a = self
    b = m
    if a.rows != b.rows or a.cols != b.cols:
      raise ValueError("Matrixes not compatible")

    return a.mapi(lambda el, r, c: el - b.data[r][c], in_place)


  def sub_scalar(self, n, in_place=True):
    """
    Subtracts a scalar to this matrix.

    Args:
      m: The scalar value to subtract.
      in_place:
        - True: modifies the original matrix (default)
        - False: builds a new matrix

    Returns:
      Returns the resulting matrix.
    """

    return self.map(lambda el: el - n, in_place)


  def mul_matrix(self, m, in_place=True):
    """
    Multiplies a matrix with this one (Hadamard Product).

    Args:
      m: The matrix to multiply.
      in_place:
        - True: modifies the original matrix (default)
        - False: builds a new matrix

    Returns:
      Returns the resulting matrix.

    Raises:
      ValueError: if a.rows != b.rows or a.cols != b.cols
    """

    a = self
    b = m
    if a.rows != b.rows or a.cols != b.cols:
      raise ValueError("Matrixes not compatible")

    return a.mapi(lambda el, r, c: el * b.data[r][c], in_place)


  def mul_scalar(self, n, in_place=True):
    """
    Multiplies this matrix with a scalar.

    Args:
      m: The multiplier scalar value.
      in_place:
        - True: modifies the original matrix (default)
        - False: builds a new matrix

    Returns:
      Returns the resulting matrix.
    """

    return self.map(lambda el: el * n, in_place)


  def div_matrix(self, m, in_place=True):
    """
    Divides this matrix with another one. (Hadamard "Division")

    Args:
      m: The divisor matrix.
      in_place:
        - True: modifies the original matrix (default)
        - False: builds a new matrix

    Returns:
      Returns the resulting matrix.

    Raises:
      ValueError: if a.rows != b.rows or a.cols != b.cols
    """

    a = self
    b = m
    if a.rows != b.rows or a.cols != b.cols:
      raise ValueError("Matrixes not compatible")

    return a.mapi(lambda el, r, c: el / b.data[r][c], in_place)


  def div_scalar(self, n, in_place=True):
    """
    Divides this matrix with a scalar.

    Args:
      m: The divisor scalar value.
      in_place:
        - True: modifies the original matrix (default)
        - False: builds a new matrix

    Returns:
      Returns the resulting matrix.
    """

    return self.map(lambda el: el / n, in_place)


  @staticmethod
  def dot_product(a, b):
    """
    Builds a new matrix obtained by the multiplication of the two given matrixes (dot product).

    Args:
      a: The first matrix.
      b: The second matrix.

    Returns:
      Returns the resulting matrix.

    Raises:
      ValueError: if a.cols != b.rows
    """

    if a.cols != b.rows:
      raise ValueError("Matrixes not compatible")

    new_data = []
    for i in range(a.rows):
      new_row = []
      for j in range(b.cols):
        _sum = 0
        for k in range(a.cols):
          _sum += a.data[i][k] * b.data[k][j]
        new_row.append(_sum)
      new_data.append(new_row)

    m = Matrix(a.rows, b.cols)
    m.data = new_data

    return m


  @staticmethod
  def transpose(m):     # TODO maybe also in_place
    """
    Builds a new matrix obtained transposing the given one.

    Args:
      m: The input matrix.

    Returns:
      Returns the resulting transposed matrix.

    Raises:
      ValueError: if a.cols != b.rows
    """

    return Matrix(m.cols, m.rows).mapi(lambda _, r, c: m.data[c][r])


  def map(self, fn, in_place=True):
    """
    Applies the given function to all the elements of the matrix.

    Args:
      fn: The function to apply (number -> number).
      in_place:
        - True: modifies the original matrix (default)
        - False: builds a new matrix

    Returns:
      Returns the resulting matrix.
    """

    new_data = self.data.copy()
    for r in range(self.rows):
      for c in range(self.cols):
        new_data[r][c] = fn(self.data[r][c])

    if in_place:
      m = Matrix(self.rows, self.cols)
      m.data = new_data
      return m
    else:
      self.data = new_data
      return self


  def mapi(self, fn, in_place=True):
    """
    Applies the given function to all the elements of the matrix (with row and column index).

    Args:
      fn: The function to apply (number, row_index, col_index -> number).
      in_place:
        - True: modifies the original matrix (default)
        - False: builds a new matrix

    Returns:
      Returns the resulting matrix.
    """

    new_data = self.data.copy()
    for r in range(self.rows):
      for c in range(self.cols):
        new_data[r][c] = fn(self.data[r][c], r, c)

    if in_place:
      m = Matrix(self.rows, self.cols)
      m.data = new_data
      return m
    else:
      self.data = new_data
      return self


  def print(self):
    """
    Prints the matrix.

    Returns:
      Returns the input matrix.
    """

    print(f"({self.rows}, {self.cols})")
    print(self.data)

    return self


  def serialize(self):
    """
    Serializes the matrix.

    Returns:
      Returns the serialized matrix.
    """

    return json.dumps(self.__dict__)


  @staticmethod
  def deserialize(data):
    """
    Deserializes the given data.

    Args:
      data: The data to deserialize (stringified json or python dict)

    Returns:
      Returns the resulting matrix.
    """

    if type(data) == str:
      data = json.loads(data)

    m = Matrix(data.rows, data.cols)
    m.data = data.data

    return m
