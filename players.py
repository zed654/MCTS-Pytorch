import numpy as np
import os
import torch

from gameEngine import GameEngine, TicTacToeMoves
Ms = TicTacToeMoves

class Player:
    def __init__(self):
        raise NotImplementedError

    def random_move(self, game, gen: np.random.Generator):
        del game, gen
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

class RandomPlayer(Player):
    def __init__(self, gameEngine: GameEngine, id=''):
        self.ge = gameEngine
        self.id = id

    def random_move(self, game, gen: np.random.Generator):
        moves = self.ge.legalMoves(game)
        return moves[gen.choice(len(moves))]

    def name(self):
        return f'Random{self.id}'

class NNPlayer(Player):
    def __init__(self, nnet, mcts, id=''):
        self.nnet = nnet
        self.mcts = mcts
        self.ge = mcts.ge
        self.id = id

    def gen_data(self, n_games=25_000, max_len=-1, progress=None, device=None):
        return self.mcts.gen_data(self.nnet, n_games=n_games, max_len=max_len, progress=progress, device=device)

    def random_move(self, game, gen: np.random.Generator):
        pi = self.mcts.policy(game, self.nnet, gen)

        moves = self.ge.legalMoves(game)
        p_pi = [pi[move] for move in moves]

        return moves[np.argmax(p_pi)]

    def best_move(self, game):
        pi = self.mcts.policy(game, self.nnet, None)

        moves = self.ge.legalMoves(game)
        p_pi = [pi[move] for move in moves]

        return moves[np.argmax(p_pi)]
        # return moves[gen.choice(len(moves), p=p_pi)]

    def name(self):
        return f"NN{self.id}"

def BestPlayer(mcts, config, gameEngine: GameEngine, id='') -> Player:
    id = f'Best{id}'
    model_type = "muzero" if config['use_muzero'] else "alphazero"
    model_file = f'./models/{config["game"]}_{model_type}_best.pt'
    
    # 모델 파일이 없으면 RandomPlayer 반환
    if not os.path.isfile(model_file):
        return RandomPlayer(gameEngine, id=id)
    
    # MuZero 모드인지 AlphaZero 모드인지에 따라 적절한 모델 선택
    from models import MuZeroNN, BaseNN
    if config['use_muzero']:
        nnet = MuZeroNN(hidden_size=32)
    else:
        nnet = BaseNN()
    
    # 모델 로드 시도
    try:
        nnet.load_state_dict(torch.load(model_file))
        return NNPlayer(nnet, mcts, id=id)
    except Exception as e:
        print(f"모델 로드 중 오류: {e}")
        print(f"기존 모델 파일({model_file})의 형식이 현재 선택된 모델 타입({model_type})과 일치하지 않습니다.")
        print("랜덤 플레이어를 대신 사용합니다.")
        return RandomPlayer(gameEngine, id=id)
