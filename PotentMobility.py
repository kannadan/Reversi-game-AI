from reversi.GameState import GameState
from random import randint
import Matrix

surrounding = {1:(0, -1), 2:(1, -1), 3:(1, 0), 4:(1, 1), 5:(0, 1), 6:(-1, 1), 7:(-1, 0), 8:(-1, -1)}

def pmobility(state, turn):
    area = turntoMatrix(state)
    total = 0
    mymob = 0
    opmob = 0
    for y in xrange(8):
        for x in xrange(8):
            if area.array[y][x] >= 0:
                value = check(x, y, area, turn)
                if area.array[y][x] == turn and value == 1:
                    mymob += 1.0
                elif area.array[y][x] == 1 - turn and value == 1:
                    opmob += 1.0
                else:
                    pass
    if mymob + opmob != 0:
        total = 100*((opmob - mymob)/(mymob + opmob))
    return total


def turntoMatrix(state):
    array = []
    for y in xrange(8):
        row = []
        for x in xrange(8):
            row.append(state.getMarkAt(x, y))
        array.append(row)
    array = Matrix.Matrix(array, None, None)
    return array

def check(x, y, mat, turn):
    point = 1
    while point < 9:
        newx, newy = surrounding[point]
        newx += x
        newy += y
        if newx >= 0 and newx < 8:
            if newy >= 0 and newy < 8:
                if mat.array[newy][newx] == -1:
                    return 1
                else:
                    pass
        point += 1
    return 0

if __name__ == "__main__":
    state = GameState()
    val = randint(35, 55)
    turn = 0
    for i in xrange(val):
        moves = state.getPossibleMoves(turn)
        move = moves[randint(0, len(moves)-1)]
        state = state.getNewInstance(move.x, move.y, turn)
        turn = 1- turn
    print state.toString()
    print pmobility(state, turn)