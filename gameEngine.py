import torch

class GameEngine:
    # returns the starting position
    def startingPosition(self):
        raise NotImplementedError

    # returns a hash of the given position
    def hash(self, state):
        del state
        raise NotImplementedError

    # returns a list of all possible moves
    def allMoves(self):
        raise NotImplementedError

    # returns True if the game is over
    def gameOver(self, state):
        del state
        raise NotImplementedError

    # returns 1 if the active player wins, -1 if the non-active player wins, 0 otherwise
    def outcome(self, state):
        del state
        raise NotImplementedError

    # returns the list of legal moves, the order must always be the same
    def legalMoves(self, state) -> list:
        del state
        raise NotImplementedError

    # plays move inplace
    def makeMove(self, state, move):
        del state, move
        raise NotImplementedError

    def copy(self, state):
        del state
        raise NotImplementedError

    def undoMove(self, state):
        del state
        raise NotImplementedError

    # Returns an encoded version of the last game state to be used as model input
    def encodeState(self, state, device=None):
        del state, device
        raise NotImplementedError

    # Returns an encoded version of the last game state that includes the outputs to predict
    def encodeStateAndOutput(self, state, policy, evaluation, device=None):
        del state, policy, evaluation, device
        raise NotImplementedError

    # Pretty prints the game
    def print(self, state, event=("?", "?"), players='', reversed=False):
        del state, event, players, reversed
        raise NotImplementedError


class InvalidBoardState(Exception):
    def __init__(self, state):
        super().__init__(state)


class InvalidMove(Exception):
    pass


TicTacToeMoves = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

class TicTacToe(GameEngine):
    def startingPosition(self):
        # 틱택토 게임 시작 상태 반환
        # 3x3 행렬, 모든 원소가 0
        return [torch.zeros((3,3), dtype=torch.int)]

    def hash(self, game):
        # 게임 상태의 해시 값 계산
        # 게임 상태를 튜플로 변환하여 해시 값 계산
        # game[-1] 은 마지막 게임 상태를 의미함.
        # numpy() 통해, 3x3 을 array 로 변환
        # flatten() 통해, array 를 1차원 리스트로 변환 (3x3 -> 9)
        # tuple() 통해, 1차원 리스트를 튜플로 변환
        # hash() 통해, 튜플을 해시 값으로 변환 (해시 값은 튜플의 값들이 같으면 같은 값을 반환함)
        return hash(tuple(game[-1].numpy().flatten())) 

    def allMoves(self):
        # 모든 가능한 수 반환
        return TicTacToeMoves

    def gameOver(self, game):
        # 게임 종료 여부 확인
        # 게임 결과가 0이 아니거나, 게임 상태에 0이 없으면 게임 종료
        return self.outcome(game) != 0 or 0 not in game[-1]

    def outcome(self, game):
        # 게임 결과 확인
        state = game[-1]
        winner = -2 # -2는 아무도 이기지 않은 초기값 / 1은 승리 / -1은 패배 / 0은 무승부
        for i in range(3):
            if state[i][0] != 0 and state[i][0] == state[i][1] == state[i][2]:
                winner = max(winner, state[i][0])
            if state[0][i] != 0 and state[0][i] == state[1][i] == state[2][i]:
                winner = max(winner, state[0][i])
        if state[0][0] != 0 and state[0][0] == state[1][1] == state[2][2]:
            winner = max(winner, state[0][0])
        if state[0][2] != 0 and state[0][2] == state[1][1] == state[2][0]:
            winner = max(winner, state[0][2])
        if winner == 1:
            raise InvalidBoardState(game)
        return -1 if winner == -1 else 0

    def legalMoves(self, game):
        # 가능한 수 반환
        moves = []
        for i in range(3):
            for j in range(3):
                if game[-1][i][j] == 0:
                    moves.append((i, j))
        return moves

    def makeMove(self, game, move):
        # 수 실행
        state = game[-1].clone()
        i, j = move
        if state[i][j] != 0:
            raise InvalidMove
        state[i][j] = 1
        game.append(-state)
        return game

    def copy(self, game):
        # 게임 상태 복사
        return [state.clone() for state in game]

    def undoMove(self, game):
        # 수 취소
        game.pop()
        return game

    def encodeState(self, game, device=None):
        # 게임 상태 인코딩
        return game[-1]

    def encodeStateAndOutput(self, game, policy, evaluation, device=None):
        # 게임 상태와 정책/가치 인코딩
        return (
            self.encodeState(game).to(device),
            torch.tensor([policy.get(a, 0.) for a in TicTacToeMoves], dtype=torch.float, device=device),
            torch.tensor(evaluation, dtype=torch.float, device=device)
        )

    def print(self, game, event=("?", "?"), players=None, reversed=False):
        # 게임 상태 출력
        output = [[' ']*4*len(game) for _ in range(3)]
        for i, state in enumerate(game):
            if reversed:
                state = -state
            state = ["X·O"[j+1] for j in ((-1) ** (i)) * state.flatten()]
            for j in range(3):
                for k in range(3):
                    output[j][4*i+k] = state[3*j+k]
        if players is not None:
            print(players)
        for l in output:
            print(''.join(l))
        print()





