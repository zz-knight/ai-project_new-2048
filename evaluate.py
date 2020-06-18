from game2048.game import Game
from game2048.displays import Display
import  torch


def single_run( lowmodel, highmodel, size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass( lowmodel, highmodel, game, display=Display(), **kwargs)
    agent.play(verbose=True)
    return game.score

if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 10

    '''====================
    Use your own agent here.'''
    from game2048.agents import MYAgent as TestAgent
    '''===================='''

    from game2048.module import high_net

    lowmodel = high_net()
    lowmodel.load_state_dict(torch.load("game2048/para/high/epoch_7.pkl", map_location='cpu'))
    lowmodel.eval()

    highmodel = high_net()
    highmodel.load_state_dict(torch.load("game2048/para/high/epoch_7.pkl", map_location='cpu'))
    highmodel.eval()

    scores = []

    for _ in range(N_TESTS):
        score = single_run(lowmodel, highmodel, GAME_SIZE, SCORE_TO_WIN,
                           AgentClass=TestAgent)
        scores.append(score)


    print(scores)
    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))

