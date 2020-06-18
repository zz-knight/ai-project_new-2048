import numpy as np
from game2048.game import Game
from game2048.displays import Display
from game2048.agents import ExpectiMaxAgent as TestAgent
import pandas as pd

'''在云端运行要使用linux下编译的expectimax'''

def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=Display(), **kwargs)
    one = agent.playtosave()
    return one

'''
GAME_SIZE = 4
SCORE_TO_WIN = 2048
N_TESTS = 2

game = Game(GAME_SIZE, SCORE_TO_WIN)
agent = TestAgent(game, display=Display())

for _ in range(N_TESTS):
    oneround = single_run(GAME_SIZE, SCORE_TO_WIN,AgentClass=TestAgent)
    onemax = oneround.max(axis=1)
    clue = np.argwhere(onemax==128)[0][0]
    onelow = oneround[0:clue,:]
    onehigh = oneround[clue:np.size(oneround,0),:]
    data1 = pd.DataFrame(onelow)  # header:原第一行的索引，index:原第一列的索引
    data1.to_csv('./data1.csv',index=False, header=False, mode='a+')
    data2 = pd.DataFrame(onehigh)  # header:原第一行的索引，index:原第一列的索引
    data2.to_csv('./data2.csv', index=False, header=False, mode='a+')
    '''


GAME_SIZE = 4
SCORE_TO_WIN = 2048
N_TESTS = 6000
group = 2

game = Game(GAME_SIZE, SCORE_TO_WIN)
agent = TestAgent(game, display=Display())

for i in range(int(N_TESTS / group)):
    oneround = single_run(GAME_SIZE, SCORE_TO_WIN, AgentClass=TestAgent)
    sumround = oneround
    for _ in range(group-1):
        oneround = single_run(GAME_SIZE, SCORE_TO_WIN,AgentClass=TestAgent)
        sumround = np.vstack((sumround, oneround))

    data = pd.DataFrame(sumround)  # header:原第一行的索引，index:原第一列的索引
    data.to_csv('./data3.csv', index=False, header=False, mode='a+')


