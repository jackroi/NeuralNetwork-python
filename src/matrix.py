import random, json

class Matrix(object):
  def __init__(self, rows, cols):
    self.rows = rows
    self.cols = cols
    self.data = [[0 for _ in range(cols)] for _ in range(rows)]

  def copy(self):
    m = Matrix(self.rows, self.cols)

    for i in range(self.rows):
      for j in range(self.cols):
        m.data[i][j] = self.data[i][j]

    return m

  @staticmethod
  def from_2d_list(rows, cols, l):
    m = Matrix(rows, cols)
    m.data = l

    return m

  @staticmethod
  def from_flat_list(rows, cols, l):
    if rows * cols != len(l): return None     # TODO raise error

    data = [[l[cols * r + c] for c in range(cols)] for r in range(rows)]
    m = Matrix(rows, cols)
    m.data = data

    return m

  @staticmethod
  def from_list(l):
    return Matrix.from_flat_list(len(l), 1, l)

  def to_list(self):
    return [self.data[i / self.cols][i % self.cols] for i in range(self.rows * self.cols)]

  def randomize(self):
    return self.randomize_range(-1.0, 1.0)

  def randomize_range(self, _min, _max):
    for r in range(self.rows):
      for c in range(self.cols):
        self.data[r][c] = random.uniform(_min, _max)

    return self

  def add_matrix(self, m, in_place=True):
    a = self, b = m
    if a.rows != b.rows or a.cols != b.cols:
      raise ValueError("Matrixes not compatible")

    return a.mapi(lambda el, r, c: el + b[r][c], in_place)

  def add_scalar(self, n, in_place=True):
    return self.map(lambda el: el + n, in_place)

  def sub_matrix(self, m, in_place=True):
    a = self, b = m
    if a.rows != b.rows or a.cols != b.cols:
      raise ValueError("Matrixes not compatible")

    return a.mapi(lambda el, r, c: el - b[r][c], in_place)

  def sub_scalar(self, n, in_place=True):
    return self.map(lambda el: el - n, in_place)

  def mul_matrix(self, m, in_place=True):
    a = self, b = m
    if a.rows != b.rows or a.cols != b.cols:
      raise ValueError("Matrixes not compatible")

    return a.mapi(lambda el, r, c: el * b[r][c], in_place)

  def mul_scalar(self, n, in_place=True):
    return self.map(lambda el: el * n, in_place)

  def div_matrix(self, m, in_place=True):
    a = self, b = m
    if a.rows != b.rows or a.cols != b.cols:
      raise ValueError("Matrixes not compatible")

    return a.mapi(lambda el, r, c: el / b[r][c], in_place)

  def div_scalar(self, n, in_place=True):
    return self.map(lambda el: el / n, in_place)

  @staticmethod
  def dot_product(a, b):
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
    return Matrix(m.cols, m.cols).mapi(lambda _, r, c: m.data[c][r])

  def map(self, fn, in_place=True):
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

  def mapi(self, fn, in_place=True):   # TODO maybe
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
    print(f"({self.rows}, {self.cols})")
    print(self.data)

    return self

  def serialize(self):
    json.dumps(self.__dict__)

  @staticmethod
  def deserialize(data):
    json.loads(data)
