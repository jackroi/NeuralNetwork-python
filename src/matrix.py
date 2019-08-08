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
  def from_list(l):
    pass

  @staticmethod
  def subtract(a, b):
    pass

  def to_list(self):
    pass
