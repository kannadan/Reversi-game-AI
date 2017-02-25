from reversi.GameState import GameState
from random import randint
import Matrix

"""
Did this stability function on it's own file because I needed to test it a lot and didn't want to play the game for
every test.
"""

horz = [[1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1]]

vert = [[1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1]]

diago1 = [[1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]]

diago2 = [[1, 1, 1, 1, 1, 1, 1, 1],
         [1, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 1],
         [1, 1, 1, 1, 1, 1, 1, 1]]


diag1 = [(5, 0), (4, 0), (3, 0), (2, 0), (1, 0), (0, 0), (0, 1),
         (0, 2), (0, 3), (0, 4), (0, 5)]

diag2 = [(0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 7),
         (2, 7), (3, 7), (4, 7), (5, 7)]

results = [3,4,5,6,7,8,7,6,5,4,3]


def turntoMatrix(state):
    """
    turn the gamestate into a matrix
    """
    array = []
    for y in xrange(8):
        row = []
        for x in xrange(8):
            row.append(state.getMarkAt(x, y))
        array.append(row)
    array = Matrix.Matrix(array, None, None)
    return array


def addFour(horiz, vert, diag1, diag2):
    stable = Matrix.addMatrix(horiz, vert)
    stable = Matrix.addMatrix(stable, diag1)
    stable = Matrix.addMatrix(stable, diag2)
    return stable


def removeFromlist(list1, list2):
    for unit in list2:
        list1.remove(unit)
    return list1


def turnToList(Stable):
    """ Creates a list of coordinates that have stable blocks on them"""
    list = []
    for y in xrange(8):
        for x in xrange(8):
            if Stable.array[y][x] == 4:
                list.append((y,x))
    return list


def expand(all, new,horizontal, vertical, diagonal1, diagonal2, array):
    """ If there are stable blocks, this expands them to their neighbors if they can be made stable too"""
    study = new         # Coordinates that need to be checked
    for y,x in study:   # go through all 8 neighbors and update their corresponding stability matrix.
        if x+1 < 8:
            if array.array[y][x] == array.array[y][x+1]:
                horizontal.array[y][x+1] = 1
        if x-1 >= 0:
            if array.array[y][x] == array.array[y][x-1]:
                horizontal.array[y][x-1] = 1

        if y+1 < 8:
            if array.array[y][x] == array.array[y+1][x]:
                vertical.array[y+1][x] = 1
        if y-1 >= 0:
            if array.array[y][x] == array.array[y-1][x]:
                vertical.array[y-1][x] = 1

        if x-1 >= 0 and y-1 >= 0:
            if array.array[y][x] == array.array[y-1][x - 1]:
                diagonal1.array[y-1][x - 1] = 1
        if x + 1 < 8 and y+1 < 8:
            if array.array[y][x] == array.array[y+1][x + 1]:
                diagonal1.array[y+1][x + 1] = 1

        if x + 1 < 8 and y-1 >=0:
            if array.array[y][x] == array.array[y-1][x + 1]:
                diagonal2.array[y-1][x + 1] = 1
        if x - 1 >= 0 and y+1 < 8:
            if array.array[y][x] == array.array[y+1][x - 1]:
                diagonal2.array[y+1][x - 1] = 1
    """
    after finding new stables, we now need to check if they too upgrade their neighbors.
    """
    stable = addFour(horizontal,vertical,diagonal1,diagonal2)
    coordinates = turnToList(stable)
    new = removeFromlist(coordinates[:], all)
    if new != []:
        final = expand(coordinates, new, horizontal,vertical,diagonal1,diagonal2, array)
    else:
        final = stable
    return final


def Stability(state, turn):

    """ This is a stability function that finds all positions on the board that are stable(cannot be turned). Returns
        a matrix where position is 1 for stable and 0 for non stable positions. Position is stable if one of the following
        requirements are met in each of the four directions(horizontal, vertical and the two diagonal)
        1: The row is full. No more marks can be added.
        2: Direction is out of bounds meaning of the board.
        3: Direction has a stable ally.
        """

    array = turntoMatrix(state)
    markArray = Matrix.Matrix(None, array.rows, array.columns)
    horizontal = Matrix.Matrix(horz, None, None)
    vertical = Matrix.Matrix(vert, None, None)
    diagonal1 = Matrix.Matrix(diago1, None, None)
    diagonal2 = Matrix.Matrix(diago2, None, None)
    for y in xrange(8):             # Creates a Matrix that has 1 for every mark, friend or foe and -1 for empty places
        for x in xrange(8):
            if array.array[y][x] >= 0:
                markArray.array[y][x] = 1
            else:
                markArray.array[y][x] = -1
                """Checks for rule number 2"""
                horizontal.array[y][x] = 0
                vertical.array[y][x] = 0
                diagonal1.array[y][x] = 0
                diagonal2.array[y][x] = 0
    marksv = markArray.transpose()
    for y in xrange(8):              # Checks for rule number 1.
        if sum(markArray.array[y]) == 8:    # Horizontal
            for x in xrange(8):
                horizontal.array[y][x] = 1
        if sum(marksv.array[y]) == 8:           # Vertical
            for x in xrange(8):
                vertical.array[x][y] = 1
    pointer = 0
    for pair in diag1:      # Diagonal stability. Pit more difficult to calculate. Still rule 1
        y, x = pair
        total = 0
        while 1:
            if x > 7 or y > 7:
                break
            total += markArray.array[y][x]
            y += 1
            x += 1
        if total == results[pointer]:
            y, x = pair
            while 1:
                if x > 7 or y > 7:
                    break
                diagonal1.array[y][x] = 1
                y += 1
                x += 1
        pointer += 1
    pointer = 0
    for pair in diag2:
        y, x = pair
        total = 0
        while 1:
            if x < 0 or y > 7:
                break
            total += markArray.array[y][x]
            y += 1
            x -= 1
        if total == results[pointer]:
            y, x = pair
            while 1:
                if x < 0 or y > 7:
                    break
                diagonal2.array[y][x] = 1
                y += 1
                x -= 1
        pointer += 1
    """
    Now rules 1 and 2 have been checked and we know what blocks are stable based on them. Now we check if these stable
    blocks can turn their unstable neighbors stable too.
    """
    stable = addFour(horizontal, vertical, diagonal1, diagonal2)
    study = turnToList(stable)
    stable = expand([], study, horizontal, vertical, diagonal1, diagonal2, array)
    total = 0
    mystable = 0
    opstable = 0
    for y in xrange(8):             # finally, count the stable blocks
        for x in xrange(8):
            if stable.array[y][x] == 4:
                if array.array[y][x] == turn:
                    stable.array[y][x] = 1
                    mystable += 1.0
                elif array.array[y][x] == 1 - turn:
                    stable.array[y][x] = -1
                    opstable += 1.0
            else:
                stable.array[y][x] = 0
    if mystable + opstable != 0:
        total = 100*((mystable - opstable)/(mystable + opstable))
    return total

if __name__ == "__main__":
    state = GameState()
    val = randint(30, 40)
    turn = 0
    for i in xrange(val):
        moves = state.getPossibleMoves(turn)
        move = moves[randint(0, len(moves)-1)]
        state = state.getNewInstance(move.x, move.y, turn)
        turn = 1- turn
    Stability(state, turn)
    print state.toString()



