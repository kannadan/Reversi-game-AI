from reversi.GameState import GameState


class Matrix():

    def __init__(self, list, rows, columns):
        if list != None:
            self.rows = len(list)
            self.columns = len(list[0])
            self.array = list
            self.dimensions = "%sx%s" %(self.rows, self.columns)
        else:
            self.rows = rows
            self.columns = columns
            self.array = []
            for y in xrange(rows):
                self.array.append([])
                for x in xrange(columns):
                    self.array[y].append(0)
            self.dimensions = "%sx%s" % (self.rows, self.columns)

    def transpose(self):
        newMatrix = Matrix(None, self.columns, self.rows)
        for y in xrange(self.rows):
            for x in xrange(self.columns):
                newMatrix.array[x][y] = self.array[y][x]
        return newMatrix

    def printMatrix(self):
        for i in xrange(self.rows):
            print self.array[i]

def addMatrix(m1, m2):
    try:
        rows1 = m1.rows
        columns1 = m1.columns
    except AttributeError:
        print "Not a matrix object"
        return
    rows2 = m2.rows
    columns2 = m2.columns
    if rows1 != rows2 or columns1 != columns2:
        print "cannot add these matrixes"
        return
    newMatrix = Matrix(None, rows1, columns1)
    for y in xrange(rows1):
        for x in xrange(columns1):
            newMatrix.array[y][x] = m1.array[y][x] + m2.array[y][x]
    return newMatrix

def multiplyMatrix(m1,m2):
    try:
        rows2 = m2.rows
        columns1 = m1.columns
    except AttributeError:
        print "Not a matrix object"
        return
    if columns1 != rows2:
        print "can't multiply these matrixes"
    rows1 = m1.rows
    columns2 = m2.columns
    newMatrix = Matrix(None, rows1, columns2)
    for y in xrange(columns2):
        for x in xrange(rows1):
            total = 0
            for k in range(columns1):
                total += m1.array[x][k]*m2.array[k][y]
            newMatrix.array[x][y] = total
    return newMatrix

def scalarMultiply(scalar, mat1):
    try:
        mat1.rows
    except AttributeError:
        print "Not a matrix object"
        return
    newMatrix = Matrix(None, mat1.rows, mat1.columns)
    for y in xrange(mat1.rows):
        for x in xrange(mat1.columns):
            newMatrix.array[y][x] = mat1.array[y][x]*scalar
    return newMatrix

def multiplyElements(mat1,mat2):
    try:
        rows1 = mat1.rows
        columns1 = mat1.columns
    except AttributeError:
        print "Not a matrix object"
        return
    rows2 = mat2.rows
    columns2 = mat2.columns
    if rows1 != rows2 or columns1 != columns2:
        print "cannot multiply these matrixes"
        return
    newMatrix = Matrix(None, rows1, columns1)
    for y in xrange(mat1.rows):
        for x in xrange(mat1.columns):
            newMatrix.array[y][x] = mat1.array[y][x] * mat2.array[y][x]
    return newMatrix

def subtract(m1,m2):
    try:
        rows1 = m1.rows
        columns1 = m1.columns
    except AttributeError:
        print "Not a matrix object"
        return
    rows2 = m2.rows
    columns2 = m2.columns
    if rows1 != rows2 or columns1 != columns2:
        print "cannot subtract these matrixes"
        return
    newMatrix = Matrix(None, rows1, columns1)
    for y in xrange(rows1):
        for x in xrange(columns1):
            newMatrix.array[y][x] = m1.array[y][x] - m2.array[y][x]
    return newMatrix




