from reversi.GameState import GameState
from reversi.Move import Move
from reversi.Game import Game
from random import randint
from reversi.ReversiAlgorithm import ReversiAlgorithm
from reversi.VisualizeGraph import VisualizeGraph
from reversi.VisualizeGameTable import VisualizeGameTable
import threading
from time import sleep, time
from operator import attrgetter
from Stability import Stability
from PotentMobility import pmobility



class HAL(ReversiAlgorithm):
    # Constants
    realLimit = 10
    leaf = 0

    #weights were chosen by runing a script that tested different combinations and evolved them.
    value1 = 18.865288467		#weight for possible moves
    value2 = 9.68306931955		#weight for pieces on the board
    value3 = 25.955808697		#weight for corner positions
    value4 = 13.2155671562		#weight for places next to corners
    value5 = 11.3421350792		#weight for spot diagonal to corner
    value8 = 3.08733515119		#weight for stability
    value9 = 1.88023949615		#weight for edge positions
    value10 = 3.95309223329              # Weight for possible mobility

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
    positions = [(0, 0), (0, 7), (7, 0), (7, 7)]
    corner_c = {1:[(0,1), (1,0)], 2:[(0,6), (1,7)], 3:[(6,0), (7, 1)], 4:[(6,7), (7,6)] }
    corner_x = {1:(1,1), 2:(1,6), 3:(6,1), 4:(6,6)}

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
            """
            Return the best move that the node can make.
            """
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

    def __init__(self):
        threading.Thread.__init__(self)
        self.myIndex = 0

    def requestMove(self, requester):
        self.running = False
        requester.doMove(self.selectedMove)

    @property
    def name(self):
        return "HAL"

    def cleanup(self):
        return

    def init(self, game, state, playerIndex, turnLength):
        """
        Initializes the algorithm.
        """
        self.lenght = turnLength # Time you have to make the move.
        self.ending = None  # Variable that ends search if the game as at the end

        #sets the time algorithm has to do the search. capped at 10 seconds
        if self.lenght > 10:
            self.lenght = 9.95
        else:
            self.lenght -= 0.05

        self.initialState = state   # Current gamestate
        self.myIndex = playerIndex  #Which marks are mine
        self.beging = time()#   start time of algorithm
        self.oponent = 1-self.myIndex
        self.controller = game
        self.initialized = True

        #keeps track of remaining turns
        self.left = 64 - (self.initialState.getMarkCount(self.myIndex) + self.initialState.getMarkCount(self.oponent))

        if self.left < 8:
            self.realLimit = 10


    def sortMoves(self, node):
        """
        Sorts the moves node has based on their score. This because the child that has highest score most likely is the
        best move even after many iterations. If you start search from the best move, there should be a lot of pruning,
        that should save time.
        """
        children = node.children
        children.sort(key=lambda child: child.score, reverse=True)
        moves = []
        for i in children:
            moves.append(i.getMove())
        return moves

    def run(self):
        """
        Run the algorithm to find the best move
        """
        print "Starting algorithm"
        while not self.initialized:
            sleep(1)
        mob = pmobility(self.initialState, self.myIndex)
        print Stability(self.initialState, self.myIndex)
        print mob
        mymoves = float(self.initialState.getPossibleMoveCount(self.myIndex))
        opmoves = float(self.initialState.getPossibleMoveCount(self.oponent))
        options = 0
        if mymoves + opmoves != 0:
            options = 100 * ((mymoves - opmoves) / (mymoves + opmoves))
        print options
        self.initialized = False
        self.running = True
        self.selectedMove = None
        self.moves = self.initialState.getPossibleMoves(self.myIndex)
        self.search_tree(self.initialState)
        print"done"
        self.controller.doMove(self.selectedMove)

    def search_tree(self, root):
        """
        Initializes the search three and updates the selected move.
        """
        while self.DEPTH_LIMIT <= self.realLimit: # repeats search tree and increases search depth every cycle
            self.leaf = 0
            root2 = self.Node(root, None, 0) # starter node
            root2.turn = self.myIndex
            root2.depth = 0
            root2.limit = self.DEPTH_LIMIT
            self.get_move(root2, self.moves) # does the search to depth defined by DEPTH_LIMIT

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
            if self.moves == []:    # In case of no possible moves
                best_move = None
                return
            else:
                best_move = self.moves[0]
            self.selectedMove = best_move
            self.DEPTH_LIMIT += 1


    def c_corner(self, node):
        """
        Part of the evaluation function. Checks the marks next to corners because if your mark is at these spots,
        opponent can jump to corner which would be bad
        """
        mycornerC = []
        opcornerC = []
        order = 1
        for i in self.positions:
            if node.state.getMarkAt(i[0], i[1]) == -1:
                c1, c2 = self.corner_c[order]   # Gives coordinates for the check
                c1 = node.state.getMarkAt(c1[0], c1[1])
                c2 = node.state.getMarkAt(c2[0], c2[1])
                if c1 == self.myIndex:
                    mycornerC.append(1.0)
                elif c1 == self.oponent:
                    opcornerC.append(1.0)
                if c2 == self.myIndex:
                    mycornerC.append(1.0)
                elif c2 == self.oponent:
                    opcornerC.append(1.0)
                order += 1
            else:
                order += 1
                pass
        mycornerC = sum(mycornerC)
        opcornerC = sum(opcornerC)
        total = 0
        if mycornerC + opcornerC != 0:
            total = 100*((opcornerC - mycornerC)/(opcornerC + mycornerC))
        return total


    def x_corner(self, node):
        """
        Checks the mark diagonal to corners for the evaluation function. This is very bad place for your mark because
         it's a easy jumping point for opponent to get to a corner.
        """
        mycornerX = []
        opcornerX = []
        order = 1
        for i in self.positions:
            if node.state.getMarkAt(i[0], i[1]) == -1:
                x = self.corner_x[order]
                x = node.state.getMarkAt(x[0], x[1])
                if x == self.myIndex:
                    mycornerX.append(1.0)
                elif x == self.oponent:
                    opcornerX.append(1.0)
                order += 1
            else:
                order += 1
                pass
        mycornerX = sum(mycornerX)
        opcornerX = sum(opcornerX)
        total = 0
        if mycornerX + opcornerX != 0:
            total = 100*((opcornerX-mycornerX)/(opcornerX + mycornerX))
        return total

    def e_corner(self, node):
        """
        Checks corners for evaluation functions. Corners are probably the strongest positions on the board and must be
        prioritised.
        """
        mycorners = []
        opcorners = []
        for i in self.positions:
            value = node.state.getMarkAt(i[0], i[1])
            if value == self.myIndex:
                mycorners.append(1.0)
            elif value == self.oponent:
                opcorners.append(1.0)
        mycorners = sum(mycorners)
        opcorners = sum(opcorners)
        total = 0
        if mycorners + opcorners != 0:
            total = 100*((mycorners - opcorners)/(mycorners + opcorners))
        return total

    def edgeWedge(self, node):
        """
        For the evaluation functions. Checks sides of the board and diagonal axes for wedge positions. This means that
        if you have mark at the side of the board wedged between two opponents marks, it cannot be turned and it is in
        a good position to turn the corner. Same works for diagonal axes but they are not unturnable. Good ambush though.
        """
        side = node.turn
        state = node.state
        my_wedge = 0
        op_wedge = 0
        edge = {1:(0,0),
                2:(0,7),
                3:(0,0),
                4:(7,0),
                5:(0,0),
                6:(0,7)}
        for i in range(6):
            i += 1
            pointer = edge[i]
            mark = state.getMarkAt(pointer[0], pointer[1])
            upmark = -1
            downmark = -1
            leftmark = -1
            rightmark= -1
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
                    pointer = (pointer[0]+1, pointer[1])
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
                                my_wedge += 1.0
                            else:
                                op_wedge += 1.0
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
                                my_wedge += 1.0
                            else:
                                op_wedge += 1.0
                            pointer = edge[i]
                            break
            if i == 3:
                while 1:
                    pointer = (pointer[0], pointer[1]+1)
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
                                my_wedge += 1.0
                            else:
                                op_wedge += 1.0
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
                                my_wedge += 1.0
                            else:
                                op_wedge += 1.0
                            pointer = edge[i]
                            break
            if i == 5:
                while 1:
                    pointer = (pointer[0]+1, pointer[1] + 1)
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
                                my_wedge += 1.0
                            else:
                                op_wedge += 1.0
                            pointer = edge[i]
                            break
            if i == 6:
                while 1:
                    pointer = (pointer[0]+1, pointer[1] - 1)
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
                                my_wedge += 1.0
                            else:
                                op_wedge += 1.0
                            pointer = edge[i]
                            break
        total = 0
        if my_wedge + op_wedge != 0:
            total = 100*((my_wedge - op_wedge)/(my_wedge + op_wedge))
        return total

    def evaluate(self, node):#heuristics for the leaf node
        """
        Heuristic function that calculates the strength of the gamestate. Higher the total value, the more likely this
        state is a winner.
        """
        mymarks = float(node.state.getMarkCount(self.myIndex))
        opmarks = float(node.state.getMarkCount(self.oponent))
        mymoves = float(node.state.getPossibleMoveCount(self.myIndex))
        opmoves = float(node.state.getPossibleMoveCount(self.oponent))
        options = 0
        discs = 0
        if mymoves + opmoves != 0:
            options = self.value1*100*((mymoves-opmoves)/(mymoves + opmoves))
        if mymarks + opmarks != 0:
            discs = self.value2 * 100*((mymarks - opmarks)/(mymarks + opmarks))
        corner_value1 = self.value3*self.e_corner(node)
        corner_value2 = self.value4*self.x_corner(node)
        corner_value3 = self.value5*self.c_corner(node)
        stability = Stability(node.state, self.myIndex)*self.value9
        wedge = self.edgeWedge(node)*self.value8
        pmob = pmobility(node.state, self.myIndex)*self.value10
        total = corner_value1 + corner_value2 + corner_value3 + stability + discs + options + wedge + pmob # add points together
        return total

    def evaluate_brute(self, node): # Heuristic that only counts number of discs. Good at the end game.
        discs = node.state.getMarkCount(self.myIndex) - node.state.getMarkCount(self.oponent)
        return discs

    def get_move(self, root, moves):
        """
        Deepens the search tree. Every time get_move is called, new node is created beneath its parent node.
        """
        if time() - self.beging > self.lenght:  # Ends search and returns to higher node if time is running out
            return
        if root.depth > self.max_depth:     # Keep track of the depth of the search
            self.max_depth = root.depth
        if root.getMove() is not None:
            move = root.getMove()
            x = move.x
            y = move.y
            moves.sort(key= lambda move: ((move.x-x)**2+ (move.y - y)**2)**0.5)
        if root.limit > root.depth and len(moves) != 0:     # If not leaf node and there is possible moves
            for move in moves:  # Goes through all the child nodes
                new_state = root.state.getNewInstance(move.x, move.y, root.turn)    # Child state
                new_node = self.Node(new_state, move, 1-root.max)   # Child node

                # Passes the alpha beta values to child. Tells it it's turn meaning if its min or max.
                new_node.alpha = root.alpha
                new_node.beta = root.beta
                new_node.turn = 1 - root.turn
                new_node.depth = root.depth + 1
                new_node.limit = root.limit

                root.addChild(new_node) # Adds child to parents list of children
                new_moves = new_node.state.getPossibleMoves(new_node.turn)
                self.get_move(new_node, new_moves)  # And deeper it goes. This is depth first search btw.

                # Again. if lower level ended because time ran out, this level must go no further.
                if time() - self.beging > self.lenght:
                    return
                self.updateAB(root, new_node) # Modifies alpha beta values based on the new score.
                if self.cut(root):  # Prunes tree if alpha or beta value is appropriate
                    break
            # After all moves have been scored, picks best move from children.
            best_move = root.getOptimalChild()
            root.score = best_move.score

        else: # We have hit rock bottom, must evaluate the board
            self.leaf += 1
            if self.left > 7:
                root.score = self.evaluate(root)
            else:
                root.score = self.evaluate_brute(root)

    def updateAB(self, root, nodeNEW):
        """
        Upgrades alpha beta values of the parent node.
        """
        if root.max == 1 and root.beta > nodeNEW.score:
            root.beta = nodeNEW.score
        elif root.max == 0 and root.alpha < nodeNEW.score:
            root.alpha = nodeNEW.score
        else:
            return

    def cut(self, root):
        """
        Prunes if alpha value is greater than beta
        """
        if root.alpha > root.beta:
            return True
        else:
            return False

if __name__ == '__main__':
    algo = HAL()
    state = GameState()
    move = Game.lastMove
    algo.init(None, state, Game.currentPlayer, Game.currentTime)
    algo.visualizeFlag = True
    algo.run()

