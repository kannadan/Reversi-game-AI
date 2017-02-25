import pickle
import threading
from operator import attrgetter
from time import sleep, time
from reversi.Game import Game
from reversi.GameState import GameState
from reversi.ReversiAlgorithm import ReversiAlgorithm
from random import random
from math import exp
import pickle

import Matrix
neurons = 30
numInputs = 65
iterrations = 50000
episodes = 0
LearningRate = 0.7



class HAL(ReversiAlgorithm):
    # Constants

    realLimit = 10
    leaf = 0
    value1 = 2.7226430333
    value2 = 0.661154392126
    value3 = 36.4583564357
    value4 = 16.2896361325
    value5 = 13.2952623028
    value6 = 1.89078571869
    value7 = 12.5173871412
    value8 = 5.0

    DEPTH_LIMIT = 1  # starter value
    lose_not = []
    # Variables
    left = 64
    initialized = False
    running = False
    controller = None
    initialState = None
    max_depth = 0           # used to stop search if game is at end and search reaches bottom
    myIndex = 0
    max_player = 1          #tells node if it is max(1) or min(0) node
    selectedMove = None
    visualizeFlag = False


    #sides ar coordinates of the sides of play area, positions ate coordinates for corners. This is needed for evaluation heuristics
    sides = [(0,2),(0,5),(2,0),(5,0),(7,2),(7,5),(2,7),(5,7),(0,3),(0,4),(3,0),(4,0),(7,3),(7,4),(3,7),(4,7)]
    checkOrder = [(0,0),(0,1),(1,0),(1,1),(0,2),(2,0),(1,2),(2,1),
                 (2,2),(3,0),(0,3),(3,1),(1,3),(3,2),(2,3),(3,3),
                 (7,0),(6,0),(7,1),(6,1),(5,0),(7,2),(5,1),(6,2),
                 (5,2),(4,0),(7,3),(4,1),(6,3),(4,2),(5,3),(4,3),
                 (0,7),(0,6),(1,7),(1,6),(0,5),(2,7),(1,5),(2,6),
                 (2,5),(0,4),(3,7),(1,4),(3,6),(2,4),(3,5),(3,4),
                 (7,7),(6,7),(7,6),(6,6),(5,7),(7,5),(5,6),(6,5),
                 (5,5),(4,7),(7,4),(4,6),(6,4),(4,5),(5,4),(4,4)]
    positions = [(0, 0), (0, 7), (7, 0), (7, 7)]
    corner_c = {1:[(0,1), (1,0)], 2:[(0,6), (1,7)], 3:[(6,0), (7, 1)], 4:[(6,7), (7,6)] }
    corner_x = {1:(1,1), 2:(1,6), 3:(6,1), 4:(6,6)}


    class Mark():
        def __init__(self, state, x, y):
            self.side = state.getMarkAt(x, y)
            coordinates = (x, y)

    class State():

        def __init__(self):
            self.myMarks = []
            self.marks = {}
            self.empty = []



        def add_mark(self, mark, x, y,):
            self.marks[x,y] = mark

        def getMyMarks(self, mySide, positions):
            for i in self.marks:
                if self.marks[i].side == mySide and positions.count(i) == 0:
                    self.myMarks.append(i)
            return self.myMarks

        def getEmpty(self):
            for i in self.marks:
                if self.marks[i].side == -1:
                    self.empty.append(i)
            return self.empty




    class Node():# made my own node class so I could add alpha, beta, minmax and score attributes

        def __init__(self, state, move, max_player):
            self.state = state
            self.move = move
            self.max = max_player
            self.children = []
            self.alpha = float("-inf")
            self.beta = float("inf")
            self.score = None
            self.turn = None
            self.depth = None
            self.limit = None

        def addChild(self, new_node):
            self.children.append(new_node)

        def getOptimalChild(self):
            try:
                max(self.children)
            except ValueError:
                return None
            if self.max == 1:
                node = min(self.children, key=attrgetter("score"))
            else:
                node = max(self.children, key=attrgetter("score"))
            return node

        def getMove(self):
            return self.move

    class neuralLayer():
        def __init__(self, neurons, inputs):
            self.synaptic_weights = []
            for i in range(neurons):
                weights = []
                for r in range(inputs):
                    weights.append(2 * random() - 1)
                self.synaptic_weights.append(weights)
            self.synaptic_weights = Matrix.Matrix(self.synaptic_weights, None, None).transpose()

    class neuralNetwork():
        def __init__(self, Layer1, Layer2):
            self.layer1 = Layer1
            self.layer2 = Layer2
            self.error = [0]

        def sigmoid(self, x):
            return 1 / (1 + exp(-x))

        def sigmoidDerivative(self, x):
            return exp(x) / ((exp(x) + 1) ** 2)

        def train(self, trainingInputs, trainingOutputs, iterations):
            n = 0
            for i in xrange(iterations):
                self.forward(trainingInputs)
                self.backwards(trainingOutputs, trainingInputs)
                self.adjust(self.layer1, self.hiddenChange)
                self.adjust(self.layer2, self.outChanges)
                n += 1
                if n % 100 == 0:
                    summa = 0
                    for r in xrange(len(self.error.array)):
                        summa += self.error.array[r][0]
                    print "avarage error is %s%%" % (summa / float(len(self.error.array)))

        def forward(self, inputs):
            self.hiddensum = Matrix.multiplyMatrix(inputs, self.layer1.synaptic_weights)
            self.hiddenresult = self.transformSigmoid(self.hiddensum)
            self.outputSum = Matrix.multiplyMatrix(self.hiddenresult, self.layer2.synaptic_weights)
            self.outputResult = self.transformSigmoid(self.outputSum)

        def backwards(self, output, inputs):
            self.error = Matrix.subtract(output, self.outputResult)
            deltaOut = Matrix.multiplyElements(self.transformSigmoidDerivative(self.outputSum), self.error)
            self.outChanges = Matrix.scalarMultiply(LearningRate,
                                                    Matrix.multiplyMatrix(self.hiddenresult.transpose(), deltaOut))
            deltaHidden = Matrix.multiplyElements(
                Matrix.multiplyMatrix(deltaOut, self.layer2.synaptic_weights.transpose()), \
                self.transformSigmoidDerivative(self.hiddensum))
            self.hiddenChange = Matrix.scalarMultiply(LearningRate,
                                                      Matrix.multiplyMatrix(inputs.transpose(), deltaHidden))

        def adjust(self, Layer, change):
            newWeights = Matrix.addMatrix(Layer.synaptic_weights, change)
            Layer.synaptic_weights = newWeights

        def transformSigmoid(self, matrix):
            newMatrix = Matrix.Matrix(None, matrix.rows, matrix.columns)
            for y in xrange(matrix.rows):
                for x in xrange(matrix.columns):
                    newMatrix.array[y][x] = self.sigmoid(matrix.array[y][x])
            return newMatrix

        def transformSigmoidDerivative(self, matrix):
            newMatrix = Matrix.Matrix(None, matrix.rows, matrix.columns)
            for y in xrange(matrix.rows):
                for x in xrange(matrix.columns):
                    newMatrix.array[y][x] = self.sigmoidDerivative(matrix.array[y][x])
            return newMatrix

        def print_weights(self):
            print "Layer 1: %s neurons with %s inputs" % (neurons, numInputs)
            print self.layer1.synaptic_weights.array
            print "Layer 2: %s neurons with %s inputs" % (1, neurons)
            print self.layer2.synaptic_weights.array
            print "error with weights was %s" % (sum(self.error) / len(self.error))
            print "episodes done %s" % (episodes)


    def __init__(self):
        threading.Thread.__init__(self)
        self.myIndex = 0
        self.lock = threading.Lock()

    def requestMove(self, requester):
        self.running = False
        requester.doMove(self.selectedMove)

    @property
    def name(self):
        return "HAL"

    def cleanup(self):
        return

    def init(self, game, state, playerIndex, turnLength):
        self.lock.acquire()
        self.initialState = state
        self.myIndex = playerIndex
        self.oponent = 1 - self.myIndex
        self.left = 64 - (self.initialState.getMarkCount(self.myIndex) + self.initialState.getMarkCount(self.oponent))
        if self.left == 60 or self.left == 59:
            global brain		#I know but I wanted to test the network at this point
            layer1 = self.neuralLayer(neurons, numInputs)
            layer2 = self.neuralLayer(1, neurons)
            file = open("weights.p", "rb")
            layer1.synaptic_weights = pickle.load(file)
            layer2.synaptic_weights =pickle.load(file)
            brain = self.neuralNetwork(layer1, layer2)
        self.lenght = turnLength
        self.ending = None

        #sets the time algorithm has to do the search. capped at 10 seconds
        if self.lenght > 10:
            self.lenght = 9.95
        else:
            self.lenght -= 0.05
        self.initialState = state
        self.beging = time()#   start time of algorithm
        self.controller = game
        self.initialized = True



        if self.left < 8:
            self.realLimit = 10
        self.lock.release()


    def sortMoves(self, node):
        children = node.children
        children.sort(key=lambda child: child.score, reverse=True)
        moves = []
        for i in children:
            moves.append(i.getMove())
        return moves

    def run(self):
        print "Starting algorithmNN2"
        self.lock.acquire()
        input = [self.turnToInput(self.initialState, self.myIndex)]
        input = Matrix.Matrix(input, None, None)
        brain.forward(input)
        total = brain.outputResult.array[0][0]
        self.initialized = False
        self.running = True
        self.selectedMove = None
        self.moves = []
        self.search_tree(self.initialState)
        print "done", self.leaf, total
        self.controller.doMove(self.selectedMove)

        self.lock.release()

    def search_tree(self, root):
            while self.DEPTH_LIMIT<= self.realLimit:# repeats search tree and increases search depth every cycle
                self.leaf = 0
                root2 = self.Node(root, None, 0) #starter node
                root2.turn = self.myIndex
                root2.depth = 0
                root2.limit = self.DEPTH_LIMIT
                if self.DEPTH_LIMIT == 1:
                    self.moves = root.getPossibleMoves(self.myIndex)
                self.get_move(root2, self.moves)

                #breaks loop if time is running out
                if time()-self.beging > self.lenght:
                    break
                print "many moves", self.leaf

                #breaks loop if max_depth hasn't changed in two iterations. -> search at the bottom of game three
                if self.ending !=self.max_depth:
                    self.ending = self.max_depth
                else:
                    break
                self.moves = self.sortMoves(root2)
                if self.moves == []:# in case of no possible moves
                    best_move = None
                else:
                    best_move = self.moves[0]
                self.selectedMove = best_move
                self.DEPTH_LIMIT += 1

    def stability(self, node): #still unfinished
        value = 0
        state = node.state
        field = self.State()
        for i in self.checkOrder:
            mark = self.Mark(state, i[0], i[1])
            field.add_mark(mark, i[0], i[1])
        for i in self.positions:
            side = state.getMarkAt(i[0], i[1])
            if side == self.myIndex:
                field.marks[i[0], i[1]].stable = 1
            else:
                pass
        my_marks = field.getMyMarks(self.myIndex, self.positions)

        for i in my_marks:
            sides = {}
            coordinates = (i[0], i[1] - 1)
            block = 0
            while 1: #variables empty, enemy, wall, stable
                try:
                    field.marks[coordinates]
                except KeyError:
                    if block == 0:
                        sides["N"] = 1
                        break
                    else:
                        sides["N"] = 2
                        break
                disc = field.marks[coordinates]
                if disc.side == self.myIndex:
                    pass
                elif disc.side == self.oponent:
                    block = 1
                elif disc.side == -1:
                    if block == 0:
                        sides["N"] = 0
                        break
                    else:
                        sides["N"] = -1
                        break
                else:
                    pass
                coordinates = (coordinates[0], coordinates[1] - 1)
            coordinates = (i[0] + 1, i[1] - 1)
            block = 0
            while 1:
                try:
                    field.marks[coordinates]
                except KeyError:
                    if block == 0:
                        sides["N-E"] = 1
                        break
                    else:
                        sides["N-E"] = 2
                        break
                disc = field.marks[coordinates]
                if disc.side == self.myIndex:
                    pass
                elif disc.side == self.oponent:
                    block = 1
                elif disc.side == -1:
                    if block == 0:
                        sides["N-E"] = 0
                        break
                    else:
                        sides["N-E"] = -1
                        break
                else:
                    pass
                coordinates = (coordinates[0] + 1, coordinates[1] - 1)
            coordinates = (i[0] + 1, i[1])
            block = 0
            while 1:
                try:
                    field.marks[coordinates]
                except KeyError:
                    if block == 0:
                        sides["E"] = 1
                        break
                    else:
                        sides["E"] = 2
                        break
                disc = field.marks[coordinates]
                if disc.side == self.myIndex:
                    pass
                elif disc.side == self.oponent:
                    block = 1
                elif disc.side == -1:
                    if block == 0:
                        sides["E"] = 0
                        break
                    else:
                        sides["E"] = -1
                        break
                else:
                    pass
                coordinates = (coordinates[0]+1, coordinates[1])
            coordinates = (i[0] + 1, i[1] + 1)
            block = 0
            while 1:
                coordinates = (coordinates[0]+1, coordinates[1]+1)
                try:
                    field.marks[coordinates]
                except KeyError:
                    if block == 0:
                        sides["E-S"] = 1
                        break
                    else:
                        sides["E-S"] = 2
                        break
                disc = field.marks[coordinates]
                if disc.side == self.myIndex:
                    pass
                elif disc.side == self.oponent:
                    block = 1
                elif disc.side == -1:
                    if block == 0:
                        sides["E-S"] = 0
                        break
                    else:
                        sides["E-S"] = -1
                else:
                    pass
                coordinates = (coordinates[0] + 1, coordinates[1] + 1)
            coordinates = (i[0], i[1] + 1)
            block = 0
            while 1:
                try:
                    field.marks[coordinates]
                except KeyError:
                    if block == 0:
                        sides["S"] = 1
                        break
                    else:
                        sides["S"] = 2
                        break
                disc = field.marks[coordinates]
                if disc.side == self.myIndex:
                    pass
                elif disc.side == self.oponent:
                    block = 1
                elif disc.side == -1:
                    if block == 0:
                        sides["S"] = 0
                        break
                    else:
                        sides["S"] = -1
                        break
                else:
                    pass
                coordinates = (coordinates[0], coordinates[1] + 1)
            coordinates = (i[0] - 1, i[1] + 1)
            block = 0
            while 1:
                try:
                    field.marks[coordinates]
                except KeyError:
                    if block == 0:
                        sides["S-W"] = 1
                        break
                    else:
                        sides["S-W"] = 2
                        break
                disc = field.marks[coordinates]
                if disc.side == self.myIndex:
                    pass
                elif disc.side == self.oponent:
                    block = 1
                elif disc.side == -1:
                    if block == 0:
                        sides["S-W"] = 0
                        break
                    else:
                        sides["S-W"] = -1
                        break
                else:
                    pass
                coordinates = (coordinates[0] - 1, coordinates[1] + 1)
            coordinates = (i[0] - 1, i[1])
            block = 0
            while 1:
                try:
                    field.marks[coordinates]
                except KeyError:
                    if block == 0:
                        sides["W"] = 1
                        break
                    else:
                        sides["W"] = 2
                        break
                disc = field.marks[coordinates]
                if disc.side == self.myIndex:
                    pass
                elif disc.side == self.oponent:
                    block = 1
                elif disc.side == -1:
                    if block == 0:
                        sides["W"] = 0
                        break
                    else:
                        sides["W"] = -1
                        break
                else:
                    pass
                coordinates = (coordinates[0] - 1, coordinates[1])
            coordinates = (i[0] - 1, i[1] - 1)
            block = 0
            while 1:
                try:
                    field.marks[coordinates]
                except KeyError:
                    if block == 0:
                        sides["W-N"] = 1
                        break
                    else:
                        sides["W-N"] = 2
                        break
                disc = field.marks[coordinates]
                if disc.side == self.myIndex:
                    pass
                elif disc.side == self.oponent:
                    block = 1
                elif disc.side == -1:
                    if block == 0:
                        sides["W-N"] = 0
                        break
                    else:
                        sides["W-N"] = -1
                        break
                else:
                    pass
                coordinates = (coordinates[0] - 1, coordinates[1] - 1)
            neighbors_stable = 0
            coordinates = (i[0], i[1])
            if field.marks[i].stable == 1:
                neighbors_stable = 1
            else:
                try:
                    disc1 = field.marks[(coordinates[0], coordinates[1]-1)].stable
                except KeyError:
                    disc1 = 0
                try:
                    disc2 = field.marks[(coordinates[0] + 1, coordinates[1]-1)].stable
                except KeyError:
                    disc2 = 0
                try:
                    disc3 = field.marks[(coordinates[0] + 1, coordinates[1])].stable
                except KeyError:
                    disc3 = 0
                try:
                    disc4 = field.marks[(coordinates[0] + 1, coordinates[1]+1)].stable
                except KeyError:
                    disc4 = 0
                try:
                    disc5 = field.marks[(coordinates[0], coordinates[1]+1)].stable
                except KeyError:
                    disc5 = 0
                try:
                    disc6 = field.marks[(coordinates[0] - 1, coordinates[1]+1)].stable
                except KeyError:
                    disc6 = 0
                try:
                    disc7 = field.marks[(coordinates[0] - 1, coordinates[1])].stable
                except KeyError:
                    disc7 = 0
                try:
                    disc8 = field.marks[(coordinates[0] - 1, coordinates[1]-1)].stable
                except KeyError:
                    disc8 = 0
                if (disc1 or disc2 or disc3 or disc4 or disc5 or disc6 or disc7 or disc8) == 1:
                    neighbors_stable = 1

            if sides["N"] == 1 or sides["S"] == 1:
                North_South = 1
            elif sides["N"] == 2 and sides["S"] == 2:
                North_South = 1
            elif sides["N"] == sides["S"]:
                North_South = 0
            else:
                North_South = -1

            if sides["N-E"] == 1 or sides["S-W"] == 1:
                NE_SW = 1
            elif sides["N-E"] == 2 and sides["S-W"] == 2:
                NE_SW = 1
            elif sides["N-E"] == sides["S-W"]:
                NE_SW = 0
            else:
                NE_SW = -1

            if sides["E"] == 1 or sides["W"] == 1:
                East_West = 1
            elif sides["E"] == 2 and sides["W"] == 2:
                East_West = 1
            elif sides["E"] == sides["W"]:
                East_West = 0
            else:
                East_West = -1

            if sides["E-S"] == 1 or sides["W-N"] == 1:
                ES_WN = 1
            elif sides["E-S"] == 2 and sides["W-N"] == 2:
                ES_WN = 1
            elif sides["E-S"] == sides["W-N"]:
                ES_WN = 0
            else:
                ES_WN = -1

            if (North_South and NE_SW and East_West and ES_WN) == 1 and neighbors_stable == 1:
                value += self.value7
                field.marks[i[0], i[1]].stable = 1
            elif (North_South or NE_SW or East_West or ES_WN) == -1:
                value -= self.value6
            else:
                pass
        return value

    def c_corner(self, node):
        cornerC = []
        order = 1
        for i in self.positions:
            if node.state.getMarkAt(i[0], i[1]) == -1:
                c1, c2 = self.corner_c[order]
                c1 = node.state.getMarkAt(c1[0], c1[1])
                c2 = node.state.getMarkAt(c2[0], c2[1])
                if c1 == self.myIndex:
                    cornerC.append(-1.0)
                elif c1 == self.oponent:
                    cornerC.append(1.0)
                if c2 == self.myIndex:
                    cornerC.append(-1.0)
                elif c2 == self.oponent:
                    cornerC.append(1.0)
                order += 1
            else:
                order += 1
                pass
        return float(sum(cornerC))

    def x_corner(self, node):
        cornerX = []
        order = 1
        for i in self.positions:
            if node.state.getMarkAt(i[0], i[1]) == -1:
                x = self.corner_x[order]
                x = node.state.getMarkAt(x[0], x[1])
                if x == self.myIndex:
                    cornerX.append(-1.0)
                elif x == self.oponent:
                    cornerX.append(1.0)
                order += 1
            else:
                pass
                order += 1
        return float(sum(cornerX))

    def e_corner(self, node):
        corners = []
        order = 1
        for i in self.positions:
            value = node.state.getMarkAt(i[0], i[1])
            if value == self.myIndex:
                value = 1.0
                corners.append(value)
            elif value == self.oponent:
                value = -1.0
                corners.append(value)
        corners = float(sum(corners))
        return corners

    def edgeWedge(self, node):
        side = node.turn
        state = node.state
        my_wedge = 0
        op_wedge = 0
        edge = {1: (0, 0),
                2: (0, 7),
                3: (0, 0),
                4: (7, 0),
                5: (0, 0),
                6: (0, 7)}
        side1 = 0
        side2 = 0
        side3 = 0
        side4 = 0
        diag1 = 0
        diag2 = 0
        for i in range(6):
            i += 1
            pointer = edge[i]
            mark = state.getMarkAt(pointer[0], pointer[1])
            upmark = -1
            downmark = -1
            leftmark = -1
            rightmark = -1
            mid = None
            if mark == -1:
                pass
            elif mark == 0:
                upmark = 0
                leftmark = 0
            else:
                upmark = 1
                leftmark = 1
            if i == 1:
                while 1:
                    pointer = (pointer[0] + 1, pointer[1])
                    if pointer[0] > 7:
                        pointer = edge[i]
                        break
                    mark = state.getMarkAt(pointer[0], pointer[1])
                    if leftmark == mark and leftmark == -1:
                        pass
                    elif leftmark == -1:
                        leftmark = mark
                    elif leftmark != -1:
                        if mark == leftmark and mid == None:
                            pass
                        elif mark != leftmark and mid == None:
                            mid = mark
                        elif mark == -1 and mid == mark:
                            leftmark = mark
                            mid = None
                        elif leftmark == mark and mid != None:
                            if side == self.myIndex:
                                score = 1
                            else:
                                score = -1
                            side1 = score
                            pointer = edge[i]
                            break
            if i == 2:
                while 1:
                    pointer = (pointer[0] + 1, pointer[1])
                    if pointer[0] > 7:
                        pointer = edge[i]
                        break
                    mark = state.getMarkAt(pointer[0], pointer[1])
                    if leftmark == mark and leftmark == -1:
                        pass
                    elif leftmark == -1:
                        leftmark = mark
                    elif leftmark != -1:
                        if mark == leftmark and mid == None:
                            pass
                        elif mark != leftmark and mid == None:
                            mid = mark
                        elif mark == -1 and mid == mark:
                            leftmark = mark
                            mid = None
                        elif leftmark == mark and mid != None:
                            if side == self.myIndex:
                                score = 1
                            else:
                                score = -1
                            side2 = score
                            pointer = edge[i]
                            break
            if i == 3:
                while 1:
                    pointer = (pointer[0], pointer[1] + 1)
                    if pointer[1] > 7:
                        pointer = edge[i]
                        break
                    mark = state.getMarkAt(pointer[0], pointer[1])
                    if upmark == mark and upmark == -1:
                        pass
                    elif upmark == -1:
                        upmark = mark
                    elif upmark != -1:
                        if mark == upmark and mid == None:
                            pass
                        elif mark != upmark and mid == None:
                            mid = mark
                        elif mark == -1 and mid == mark:
                            upmark = mark
                            mid = None
                        elif upmark == mark and mid != None:
                            if side == self.myIndex:
                                score = 1
                            else:
                                score = -1
                            side3 = score
                            pointer = edge[i]
                            break

            if i == 4:
                while 1:
                    pointer = (pointer[0], pointer[1] + 1)
                    if pointer[1] > 7:
                        pointer = edge[i]
                        break
                    mark = state.getMarkAt(pointer[0], pointer[1])
                    if upmark == mark and upmark == -1:
                        pass
                    elif upmark == -1:
                        upmark = mark
                    elif upmark != -1:
                        if mark == upmark and mid == None:
                            pass
                        elif mark != upmark and mid == None:
                            mid = mark
                        elif mark == -1 and mid == mark:
                            upmark = mark
                            mid = None
                        elif upmark == mark and mid != None:
                            if side == self.myIndex:
                                score = 1
                            else:
                                score = -1
                            side4 = score
                            pointer = edge[i]
                            break
            if i == 5:
                while 1:
                    pointer = (pointer[0] + 1, pointer[1] + 1)
                    if pointer[1] > 7:
                        pointer = edge[i]
                        break
                    mark = state.getMarkAt(pointer[0], pointer[1])
                    if upmark == mark and upmark == -1:
                        pass
                    elif upmark == -1:
                        upmark = mark
                    elif upmark != -1:
                        if mark == upmark and mid == None:
                            pass
                        elif mark != upmark and mid == None:
                            mid = mark
                        elif mark == -1 and mid == mark:
                            upmark = mark
                            mid = None
                        elif upmark == mark and mid != None:
                            if side == self.myIndex:
                                score = 1
                            else:
                                score = -1
                            diag1 = score
                            pointer = edge[i]
                            break
            if i == 6:
                while 1:
                    pointer = (pointer[0] + 1, pointer[1] - 1)
                    if pointer[0] > 7:
                        pointer = edge[i]
                        break
                    mark = state.getMarkAt(pointer[0], pointer[1])
                    if upmark == mark and upmark == -1:
                        pass
                    elif upmark == -1:
                        upmark = mark
                    elif upmark != -1:
                        if mark == upmark and mid == None:
                            pass
                        elif mark != upmark and mid == None:
                            mid = mark
                        elif mark == -1 and mid == mark:
                            upmark = mark
                            mid = None
                        elif upmark == mark and mid != None:
                            if side == self.myIndex:
                                score = 1
                            else:
                                score = -1
                            diag2 = score
                            pointer = edge[i]
                            break
        return side1, side2, side3, side4, diag1, diag2

    def check(self,list, spot):
        connected = []
        North = (spot[0], spot[1]-1)
        NE = (spot[0]+1,spot[1]-1)
        East = (spot[0]+1, spot[1])
        ES = (spot[0]+1, spot[1]+1)
        South = (spot[0], spot[1]+1)
        SW = (spot[0]-1, spot[1]+1)
        West = (spot[0]-1, spot[1])
        WN = (spot[0]-1, spot[1]-1)
        if list.count(North) != 0:
            connected.append(North)
        if list.count(NE) != 0:
            connected.append(NE)
        if list.count(East) != 0:
            connected.append(East)
        if list.count(ES) != 0:
            connected.append(ES)
        if list.count(South) != 0:
            connected.append(South)
        if list.count(SW) != 0:
            connected.append(SW)
        if list.count(West) != 0:
            connected.append(West)
        if list.count(WN) != 0:
            connected.append(WN)
        return connected

    def Frontier(self, node):
        regions, field = self.partition(node)
        mySide = self.myIndex
        frontiers = {}
        index = 1
        checked = []
        for region in regions:
            myMarks = 0
            opMarks = 0
            for mark in region:
                temp = self.check(region, mark)
                for i in temp:
                    if checked.count(i) == 0:
                        if field.marks[i].side == mySide:
                            myMarks += 1
                        else:
                            opMarks += 1
                        checked.append(i)
                    else:
                        pass
            frontiers[index] = (myMarks, opMarks)
            index += 1
        return frontiers

    def turnToInput(self, state, turn):
        list = []
        for y in xrange(8):
            for x in xrange(8):
                list.append(state.getMarkAt(x, y))
        list.append(turn)
        return list

    def evaluate(self, node):#heuristics for the leaf node

        state = node.state
        input = [self.turnToInput(state, node.turn)]
        input = Matrix.Matrix(input, None, None)
        brain.forward(input)
        total = brain.outputResult.array[0][0]
        return total

    def evaluate_brute(self, node): #heuristic that only counts number of discs
        discs = node.state.getMarkCount(self.myIndex) - node.state.getMarkCount(self.oponent)
        return discs

    def get_move(self, root, moves):# function that deepens the search
        if time() - self.beging > self.lenght:#ends search and returns to higher node if time is running out
            return
        if root.depth > self.max_depth:#keep track of the depth of the search
            self.max_depth = root.depth
        if root.getMove() !=None:
            move = root.getMove()
            x = move.x
            y = move.y
            moves.sort(key= lambda move: ((move.x-x)**2+ (move.y - y)**2)**0.5)
        if root.limit > root.depth and len(moves) != 0: #if not leaf node
            for move in moves:#goes through all the child nodes
                new_state = root.state.getNewInstance(move.x, move.y, root.turn)#child state
                new_node = self.Node(new_state, move, 1-root.max)#child node

                #passes the alpha beta values to child
                new_node.alpha = root.alpha
                new_node.beta = root.beta
                new_node.turn = 1 - root.turn
                new_node.depth = root.depth + 1
                new_node.limit = root.limit

                root.addChild(new_node)#adds child to parents list of children
                new_moves = new_node.state.getPossibleMoves(new_node.turn)
                self.get_move(new_node, new_moves)# and deeper it goes. This is depth first search btw.

                #again. if lower level ended because time ran out, this level must go no further.
                if time() - self.beging > self.lenght:
                    return
                self.updateAB(root, new_node)
                if self.cut(root):
                    break
            best_move = root.getOptimalChild()
            root.score = best_move.score

        else: #we have hit rock bottom, must evaluate the board
            self.leaf += 1
            root.score = self.evaluate(root)

    def updateAB(self, root, nodeNEW, ):
        if root.max == 1 and root.beta > nodeNEW.score:
            root.beta = nodeNEW.score
        elif root.max == 0 and root.alpha < nodeNEW.score:
            root.alpha = nodeNEW.score
        else:
            return

    def cut(self, root):
        if root.alpha > root.beta:
            return True
        else:
            return False

if __name__ == '__main__':
    algo = HAL()
    state = GameState()
    algo.init(None, state, Game.currentPlayer, Game.currentTime)
    algo.visualizeFlag = True
    algo.run()
