import numpy as np
import torch
import torch.nn.functional as F


class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def playtosave(self,max_iter=np.inf):
        n_iter = 0
        direction = self.step()
        onetime = self.game.board
        onetime = np.append(onetime, direction)
        saveboard = onetime
        self.game.move(direction)
        n_iter += 1
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            onetime = self.game.board
            onetime=np.append(onetime,direction)
            saveboard=np.vstack((saveboard,onetime))
            self.game.move(direction)
            n_iter += 1

        return saveboard

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction


class MYAgent(Agent):

    '''虽然采取了分层策略，但在使用时一般每层模型相同'''

    def __init__(self, lowmodel, highmodel, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        self.lowmodel = lowmodel
        self.highmodel = highmodel

    def step(self):
        new_board = self.game.board
        new_board[new_board == 0] = 1
        new_board = np.log2(new_board)
        new_board = np.reshape(new_board, (1, 4, 4))
        new_board1 = np.rot90(new_board, 1, (1, 2))
        new_board2 = np.rot90(new_board, 2, (1, 2))
        new_board3 = np.rot90(new_board, 3, (1, 2))

        if self.game.score <= 1024:
            new_board = F.one_hot(torch.LongTensor(new_board), 11).permute(0, 3, 1, 2).float()
            new_board1 = F.one_hot(torch.LongTensor(new_board1.astype(np.float)), 11).permute(0, 3, 1, 2).float()
            new_board2 = F.one_hot(torch.LongTensor(new_board2.astype(np.float)), 11).permute(0, 3, 1, 2).float()
            new_board3 = F.one_hot(torch.LongTensor(new_board3.astype(np.float)), 11).permute(0, 3, 1, 2).float()

            output = self.lowmodel(new_board)
            output1 = self.lowmodel(new_board1)
            output2 = self.lowmodel(new_board2)
            output3 = self.lowmodel(new_board3)

            output = torch.cat((output[:, 0], output[:, 1], output[:, 2], output[:, 3]), 0)
            output1 = torch.cat((output1[:, 1], output1[:, 2], output1[:, 3], output1[:, 0]), 0)
            output2 = torch.cat((output2[:, 2], output2[:, 3], output2[:, 0], output2[:, 1]), 0)
            output3 = torch.cat((output3[:, 3], output3[:, 0], output3[:, 1], output3[:, 2]), 0)

            output = output + output1 + output2 + output3

            direction = int(torch.max(output, 0)[1])
        else:
            if self.game.score < -1:
                new_board = F.one_hot(torch.LongTensor(new_board), 11).permute(0, 3, 1, 2).float()
                new_board1 = F.one_hot(torch.LongTensor(new_board1.astype(np.float)), 11).permute(0, 3, 1, 2).float()
                new_board2 = F.one_hot(torch.LongTensor(new_board2.astype(np.float)), 11).permute(0, 3, 1, 2).float()
                new_board3 = F.one_hot(torch.LongTensor(new_board3.astype(np.float)), 11).permute(0, 3, 1, 2).float()

                output = self.highmodel(new_board)
                output1 = self.highmodel(new_board1)
                output2 = self.highmodel(new_board2)
                output3 = self.highmodel(new_board3)

                output = torch.cat((output[:, 0], output[:, 1], output[:, 2], output[:, 3]), 0)
                output1 = torch.cat((output1[:, 1], output1[:, 2], output1[:, 3], output1[:, 0]), 0)
                output2 = torch.cat((output2[:, 2], output2[:, 3], output2[:, 0], output2[:, 1]), 0)
                output3 = torch.cat((output3[:, 3], output3[:, 0], output3[:, 1], output3[:, 2]), 0)

                output = output + output1 + output2 + output3

                direction = int(torch.max(output, 0)[1])
            else:
                direction = np.random.randint(0, 4)

        return direction
