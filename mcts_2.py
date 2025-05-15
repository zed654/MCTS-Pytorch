import copy
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pickle as pkl
import rich.progress as rp
import torch
from typing import Any

from gameEngine import GameEngine, TicTacToe, TicTacToe_MuZero
from models import BaseNN, MuZeroNN
from utils import get_freer_gpu
from players import Player, NNPlayer, RandomPlayer, BestPlayer

# 실험에 사용할 하이퍼파라미터 설정
if True:
    config = {
        'game': 'ttt', # 게임 종류: 틱택토
        'n_sim': 100, # MCTS 시뮬레이션 횟수
        'n_gen_games': 90, # self-play로 생성할 게임 수
        'n_train_iter': 100, # 신경망 학습 반복 횟수
        'batch_size': 1024,
        'n_games_eval': 45, # 평가용 게임 수
        'n_iter': 10, # 전체 반복 횟수
        'n_jobs': 90, # 병렬 처리 작업 수
        'use_muzero': True, # MuZero 알고리즘 사용 여부
        'n_unroll_steps': 3, # 몬테카를로 트리 탐색 시 탐색 깊이
    }
else:
    config = {
        'game': 'ttt', # 게임 종류: 틱택토
        'n_sim': 4, # MCTS 시뮬레이션 횟수(빠른 실험용)
        'n_gen_games': 50, # self-play로 생성할 게임 수
        'n_train_iter': 100, # 신경망 학습 반복 횟수
        'batch_size': 1024,
        'n_games_eval': 100, # 평가용 게임 수
        'n_iter': 51, # 전체 반복 횟수
        'n_jobs': 50, # 병렬 처리 작업 수
        'use_muzero': True, # MuZero 알고리즘 사용 여부
        'n_unroll_steps': 3, # 몬테카를로 트리 탐색 시 탐색 깊이
    }

# MCTS(몬테카를로 트리 탐색) 클래스
class MCTS:
    def __init__(self, gameEngine: GameEngine, c=1.0, n_sim=config['n_sim'], single_thread=False, device=None, is_muzero=config['use_muzero']):
        super().__init__()

        self.ge = gameEngine  # 게임 엔진(틱택토 등)
        self.device = "cpu" if device is None else device
        self.is_muzero = is_muzero  # MuZero 사용 여부

        self.c = c  # PUCT 상수
        self.n_sim = n_sim  # 시뮬레이션 횟수
        self.alpha = 0.03  # Dirichlet 노이즈 파라미터
        self.eps_exploration = 0.25  # 탐험 비율
        self.num_sampling_moves = 30  # 확률적으로 수를 고르는 턴 수

        self.single_thread = single_thread
        if self.single_thread:
            self.model = dict()
        else:
            manager = multiprocessing.Manager()
            self.model = manager.dict()

    # PUCT 점수 계산 (탐험/이용 균형)
    def puct_score(self, sqrt_N, n, p, q):
        return q + self.c * p * sqrt_N / (1 + n)

    # 현재 상태에서 가장 좋은 수 선택
    def bestMove(self, moves, N, P, Q):
        max_u, best_a_i = -float("inf"), np.random.choice(len(moves))
        sqrt_N = np.sqrt(1+np.sum(N))
        s = []
        for a_i, (n, p, q) in enumerate(zip(N, P, Q)):
            u = self.puct_score(sqrt_N, n, p, q)
            s.append(u)
            if u > max_u:
                max_u = u
                best_a_i = a_i
        return best_a_i, moves[best_a_i]

    # 노드 확장: 신경망으로 정책/가치 예측, 트리에 추가
    def expand(self, nnet, N, P, Q, s, hs=None, zero=False, latent_state=None):
        if hs is None: # 처음에는 hs 가 None 임. 따라서, hash 로 정의해줌. hash(tuple(game[-1].numpy().flatten())) 로 구성되어있음
            hs = self.ge.hash(s)
            
        # MuZero 모드일 때는 추상 상태 사용
        if self.is_muzero and latent_state is None:
            latent_state = self.ge.representation(s[-1])
            
        if zero or hs not in self.model: # zero 가 true 면 predict() 를 강제로 넣어줌. 처음에는 False 임. self.model 은 딕셔너리임. self.model[hs] 의 키 값이 없으면 실행되는것
            if self.is_muzero:
                # MuZero: 추상 상태에서 정책/가치 예측
                self.model[hs] = nnet.predict(latent_state, device=self.device)
            else:
                # AlphaZero: 실제 게임 상태에서 정책/가치 예측
                self.model[hs] = nnet.predict(self.ge.encodeState(s, device=self.device))
        model_v, model_P = self.model[hs] # hs(state의 hash 임. 즉 게임판 위치). model_v 는 1개가 나오고, model_P 는 9개(3x3 보드판)가 나옴.

        moves = self.ge.legalMoves(s) # 게임판에서 점유하지 않은 공간 정보 반환
        P[hs] = []
        for move in moves: # BaseNN 으로 예측한 model_P(Policy) 값을 P[hs] 에 업데이트 하기 위한 작업. 
            i = -1
            if config['game'] == 'ttt':
                for j in range(len(self.ge.allMoves())):
                    if self.ge.allMoves()[j] == move:
                        i = j
                        break
            else:
                raise NotImplementedError
            assert i != -1
            P[hs].append(model_P[i]) # 위에서 계산된 model_P[i] 값에 해당하는 hs 찾아서 P[hs] 에 추가
        Q[hs] = [0.] * len(moves) # 모두 0으로 초기화
        N[hs] = [0] * len(moves) # 모두 0으로 초기화

        return model_v

    # MCTS 트리 탐색 및 백업
    def search(self, s, nnet, N, P, Q):
        actions = []
        hs = self.ge.hash(s) # 루트노드 가져오는 과정임 (현재(마지막) s[루트노드 게임판들] 의 hash 값을 가져옴)
        
        # MuZero 모드일 때는 추상 상태 표현으로 변환
        latent_state = None
        if self.is_muzero:
            latent_state = self.ge.representation(s[-1])

        # 게임이 끝나지 않았고, 이미 확장된 노드라면 계속 탐색
        while not self.ge.gameOver(s) and hs in N:
            moves = self.ge.legalMoves(s) # s 에서 빈칸 찾아서 반환
            # move_i 는 moves 중에 몇 번째인지 인덱스 반환이고, 인덱스로 선택된 moves 를 move에 저장하는 것
            move_i, move = self.bestMove(moves, N[hs], P[hs], Q[hs]) # PCTU Score 계산해서 가장 좋은 수 선택

            actions.append((hs, move_i))
            
            # MuZero 모드에서는 동역학 모델 사용
            if self.is_muzero and latent_state is not None:
                # 동역학 모델로 다음 상태와 보상 예측
                latent_state_new, predicted_reward = self.ge.dynamics(latent_state, move)
                latent_state = latent_state_new
            
            # 실제 게임 상태도 업데이트
            s = self.ge.makeMove(s, move) # 결정된 수 두는 곳
            hs = self.ge.hash(s) # 위에서 선택이 이전 판과 동일하면, 이전에 선택한 hs 값이 나올거임. 이게 핵심. (새로운 hs 를 만들어줌)

        # 게임이 끝났으면 결과 반환, 아니면 노드 확장
        if self.ge.gameOver(s):
            v = self.ge.outcome(s)
        else: # 위에서 hs 가 새로 만들어져서 expand() 의 매개변수로 넣어줬으므로, 결과값에서 hs 값이 유지됨
            v = self.expand(nnet, N, P, Q, s, hs=hs, zero=True, latent_state=latent_state) # P, v 에 모델(BaseNN) 결과값 넣고, N, Q 초기화상태

        # 백업(역방향으로 결과 전파)
        for hs, move_i in actions[::-1]: # 탐색 경로를 역순으로 순회
            v = -v # 현재는 플레이어 입장이므로, 이전은 상대이므로 부호 반전
            Q[hs][move_i] = (Q[hs][move_i] * N[hs][move_i] + v) / (N[hs][move_i] + 1)
            N[hs][move_i] += 1

        return v

    # 현재 상태에서 여러 번 시뮬레이션을 돌려 정책(확률 분포) 생성
    def policy(self, s, nnet, gen: np.random.Generator | None, progress=None):
        hs = self.ge.hash(s)
        N = {} # N[s][a]: 특정 상태에서 특정 수를 선택한 횟수 (방문 횟수)
        P = {} # P[s][a]: 신경망이 평가한 확률 (사전 확률)
        Q = {} # Q[s][a]: 평균 가치 (행동 가치)
        
        # MuZero 모드일 때만 추상 상태 표현 변환
        latent_state = None
        if self.is_muzero:
            latent_state = self.ge.representation(s[-1])

        # Dirichlet 노이즈로 탐험성 부여
        # 매번 같은 방식으로 수를 고르면, 데이터 편향 발생함. dirichlet 분포 노이즈 넣는거임.
        if gen is None:
            dir_noise = np.ones(len(self.ge.legalMoves(s)))
        else:
            dir_noise = gen.dirichlet([self.alpha] * len(self.ge.legalMoves(s))) # 탐험을 촉진하기 위한 노이즈 추가

        # 모델 초기화 및 확장
        self.expand(nnet, N, P, Q, s, hs=hs, latent_state=latent_state) # 모델 예측 값을 가져옴. (BaseNN) / model_v 결과값은 1개 나오는데, 버려짐!!! (학습에 쓰지 않기 때문) -> 이 구조는 P, Q, N 딕셔너리 구조를 초기화하려는 것
        for i, noise in enumerate(dir_noise): # expand() 에서 생된 각 보드판(hs) 의 각각의 자리(3x3)에 P 값에 
            P[hs][i] = P[hs][i] * (1 - self.eps_exploration) + noise * self.eps_exploration

        
        # 아래는 시뮬레이션/탐험 단계 (search() 함수 n_sim[4회] 호출; MCTS 트리 탐색)
        # 여기서 생성되는 시뮬레이션들은 정책값(P)에 dirichlet 노이즈 넣지 않음 -> 실제 모델의 정책(P)에 따라 전개되어야만 함
        task = None
        if progress is not None:
            task = progress[0].add_task(f"[cyan]{progress[1]}: [magenta]Generating Policy", total=self.n_sim)
        for _ in range(self.n_sim):
            self.search(self.ge.copy(s), nnet, N, P, Q)
            if progress is not None:
                progress[0].update(task, advance=1)

        total = np.sum(N[hs])
        moves = self.ge.legalMoves(s)
        return {
            a: N[hs][a_i] / total
            for a_i, a in enumerate(moves)
        }

    # 여러 게임을 생성하여 학습 데이터(상태, 정책, 가치) 생성
    def gen_data(self, nnet, n_games=25_000, max_len=-1, progress=None, device=None, K=3):
        task = None
        if progress is not None:
            task = progress[0].add_task(f"[cyan]{progress[1]}: [green]Generating Data", total=n_games, arg1_n="#positions", arg1=0)

        def gen_game(gen: np.random.Generator):
            if not self.is_muzero:
                # AlphaZero-style game generation
                examples = []
                game = self.ge.startingPosition()
                while not self.ge.gameOver(game):
                    pi = self.policy(game, nnet, gen)
                    moves = self.ge.legalMoves(game)
                    p_pi = [pi[move] for move in moves]
                    move = moves[gen.choice(len(moves), p=p_pi)]
                    examples.append([game.copy(), pi])
                    game = self.ge.makeMove(game, move)
                result = self.ge.outcome(game) * (-1) ** len(examples)
                for example in examples:
                    example.append(result)
                    result = -result
                return examples
            else:
                # MuZero K-step rollout data generation
                examples_per_game = []
                game = self.ge.startingPosition()
                latent_game = self.ge.representation(game[-1])
                prev_state = game.copy()
                prev_latent = latent_game
                states_seq = []
                policies_seq = []
                values_seq = []
                actions_seq = []
                rewards_seq = []
                next_states_seq = []
                # For K-step: collect sequences per sample
                # Play through the game, collect all transitions
                while not self.ge.gameOver(game) and (max_len == -1 or len(states_seq) < max_len):
                    pi = self.policy(game, nnet, gen)
                    moves = self.ge.legalMoves(game)
                    p_pi = [pi[move] for move in moves]
                    if len(states_seq) < self.num_sampling_moves:
                        move_idx = gen.choice(len(moves), p=p_pi)
                    else:
                        move_idx = np.argmax(p_pi)
                    move = moves[move_idx]
                    game_next = self.ge.makeMove(game, move)
                    reward = 0.0
                    if self.ge.gameOver(game_next):
                        reward = self.ge.outcome(game_next)
                    elif hasattr(self.ge, 'reward'):
                        reward = self.ge.reward(game, move, game_next)
                    latent_game_new, _ = self.ge.dynamics(latent_game, move)
                    # Store per-step
                    states_seq.append(prev_state.copy())
                    policies_seq.append(pi)
                    values_seq.append(None)  # Placeholder, to fill after game ends
                    actions_seq.append(move)
                    rewards_seq.append(reward)
                    next_states_seq.append(game_next.copy())
                    latent_game = latent_game_new
                    prev_state = game_next.copy()
                    game = game_next
                # Now, for each position, compute value from end
                result = self.ge.outcome(game) * (-1) ** len(states_seq)
                value_seq = []
                for _ in range(len(states_seq)):
                    value_seq.append(result)
                    result = -result
                # For K-step, build rollout sequences: for each position, build up to K steps
                data_samples = []
                n = len(states_seq)
                for start in range(n):
                    # For each start index, build a rollout of up to K steps
                    k_actions = []
                    k_rewards = []
                    for k in range(K):
                        idx = start + k
                        if idx < n:
                            # Convert action to index
                            action = actions_seq[idx]
                            if isinstance(action, int):
                                action_idx = action
                            else:
                                try:
                                    action_idx = self.ge.allMoves().index(action)
                                except Exception:
                                    action_idx = int(action)
                            k_actions.append(action_idx)
                            k_rewards.append(rewards_seq[idx])
                        else:
                            # Padding: repeat last action/reward (or 0)
                            k_actions.append(0)
                            k_rewards.append(0.0)
                    # Compose sample: (state, policy, value, actions[K], rewards[K], next_state)
                    data_samples.append([
                        states_seq[start],
                        policies_seq[start],
                        torch.tensor([value_seq[start]], dtype=torch.float32),
                        k_actions,
                        k_rewards,
                        next_states_seq[start]
                    ])
                return data_samples

        # 병렬로 여러 게임 생성
        # examples 에 시뮬레이션된 게임들이 모여짐
        # 총 n_jobs(50)개의 게임판이 생기고, 각 게임판에서 게임이 끝날 때까지 시뮬레이션(현재는 4회; config 의 n_sim 값) 돌림
        examples: Any = Parallel(
            n_jobs=config['n_jobs'],
            batch_size=1, # type: ignore
            return_as='generator'
        )(
            delayed(gen_game)(np.random.default_rng(np.random.randint(int(1e10))))
            for _ in range(n_games)
        )
        data = []
        for gameData in examples:
            if self.is_muzero:
                # Detect if K-step data (list of lists)
                # Each gameData is a list of samples: (state, policy, value, actions[K], rewards[K], next_state)
                for sample in gameData:
                    state, policy, value, actions, rewards, next_state = sample
                    state_enc, policy_enc, value_enc = self.ge.encodeStateAndOutput(state, policy, value, device=self.device if device is None else device)
                    next_state_enc, _, _ = self.ge.encodeStateAndOutput(next_state, policy, value, device=self.device if device is None else device)
                    # actions: list of K
                    actions_tensor = torch.tensor(actions, dtype=torch.long, device=state_enc.device)
                    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=state_enc.device)
                    data.append((state_enc, policy_enc, value_enc, actions_tensor, rewards_tensor, next_state_enc))
            else:
                for state, policy, value, _, _, _ in gameData:
                    data.append(self.ge.encodeStateAndOutput(state, policy, value, device=self.device if device is None else device))
            if progress is not None:
                progress[0].update(task, advance=1, arg1=len(data))
        return data

# Runs a game between model1 and model2 and returns the result
def play_game(game, player1: Player, player2: Player, gameEngine: GameEngine, gen: np.random.Generator):
    mult = 1
    while not gameEngine.gameOver(game):
        move = player1.random_move(game, gen)
        game = gameEngine.makeMove(game, move)
        player1, player2 = player2, player1
        mult = -mult

    return mult * gameEngine.outcome(game), game

def results_to_score(results):
    if isinstance(results, list) and len(results) == 3:
        return f"{(results[0]+results[1]/2) / sum(results):.5f}"
    return "N/A"

def pretty_results(results):
    return f'{results[0]}/{results[1]}/{results[2]} {results_to_score(results)}'

# Runs n games between player1 and player2 and returns the results
def pit(player1: Player, opponents: dict[str, Player], gameEngine: GameEngine, n_games=400, progress=None, n_display=5, step=-1):
    tasks = {}
    if progress is not None:
        tasks = {
            task_name: progress[0].add_task(
                f"[cyan]{progress[1]}:[/ cyan] Playing vs {task_name}",
                total=n_games,
                arg1_n="results",
                arg1="0/0/0 N/A"
            )
            for task_name in opponents.keys()
        }

    def playgames(i, task_name: str, player2: Player, gen):
        if i % 2 == 0:
            x, game = play_game(gameEngine.startingPosition(), player2, player1, gameEngine, gen)
            x = -x
            game = (False, game)
        else:
            x, game = play_game(gameEngine.startingPosition(), player1, player2, gameEngine, gen)
            game = (True, game)
        return task_name, player2.name(), x, game

    rresults: Any = Parallel(
        n_jobs=config['n_jobs'],
        batch_size=1, # type: ignore
        return_as='generator'
    )(
        delayed(playgames)(i, task_name, player2, np.random.default_rng(np.random.randint(int(1e10))))
        for i in range(n_games)
        for task_name, player2 in opponents.items()
    )
    results = {
        task_name: [0, 0, 0]
        for task_name in opponents.keys()
    }
    displayed = {
        task_name: [0, 0]
        for task_name in opponents.keys()
    }
    for task_name, player2_name, x, (normalOrder, game) in rresults:
        if displayed[task_name][normalOrder] < n_display and x == -1:
            players = f'{player1.name()}/{player2_name}' if normalOrder else f'{player2_name}/{player1.name()}'
            gameEngine.print(game, ("MCTS Training", step), players, reversed=not normalOrder)
            displayed[task_name][normalOrder] += 1
        results[task_name][1-x] += 1
        if progress is not None:
            progress[0].update(tasks[task_name], advance=1, arg1=pretty_results(results[task_name]))
    return results

def finalnet(gameEngine, iterations=200, n_games_eval=400, device=[None, None]):
    # MuZero 모드인지 AlphaZero 모드인지에 따라 적절한 모델 선택
    if config['use_muzero']:
        nnet = MuZeroNN(hidden_size=32).to(device[1])
        mcts_instance = MCTS(gameEngine, device=device[1], is_muzero=True)
    else:
        nnet = config['model']().to(device[1])
        mcts_instance = MCTS(gameEngine, device=device[1])
        
    oldPlayer = NNPlayer(nnet, mcts_instance, id="init") # 학습 과정에서 이전 모델을 저장하는 용도로 사용
    randomPlayer = RandomPlayer(gameEngine)
    bestPlayer = BestPlayer(mcts_instance, config, gameEngine)

    dpi = 96
    steps = []
    plt.figure(figsize=(1920/dpi, 1080/dpi), dpi=dpi)
    # with Progress 는 커맨드라인에 Bar 만드는 시각화 툴임. 뒤에 progress.add_task() 까지. 그 뒤에 값들 넣어서 progress.update() 하는 식으로 사용함.
    with rp.Progress(
        *rp.Progress.get_default_columns(),
        rp.TimeElapsedColumn(),
        rp.MofNCompleteColumn(),
        rp.TextColumn("{task.fields[arg1_n]}: {task.fields[arg1]}"),
        refresh_per_second=1
    ) as progress:
        task = progress.add_task("[cyan]Stepping", total=iterations, arg1_n="perf", arg1="N/A")
        for i in range(iterations):
            data = oldPlayer.gen_data(n_games=config['n_gen_games'], max_len=512, progress=(progress, i), device=device[0])

            with open(f'data/games/{config["game"]}-it{i:03}-{nnet.id()}.out', 'wb') as f:
                pkl.dump({
                    'data': data
                }, f)

            # 기존 모델 복사 (추후 학습 후 업데이트 위함)
            new_nnet = copy.deepcopy(nnet).to(device[0])

            # BaseNN 의 fit() 함수는 데이터를 받아서 학습을 진행함.
            # 생성하나 게임 (n_gen_games) 만큼의 결과를 모아서 한 번에 학습 진행 (fit 함수가 그 역할 함)
            losses = new_nnet.fit(
                data,
                n_iter=config['n_train_iter'],
                batch_size=config['batch_size'],
                progress=(progress, i),
                K=config['n_unroll_steps']
            )

            # MCTS 인스턴스 생성 (MuZero 모드 여부에 따라)
            # mcts_instance 는 학습 과정에서 이전 모델의 설정정보를 용도로 사용됨. 따라서 재사용하면 안됨.
            if config['use_muzero']:
                new_mcts = MCTS(gameEngine, device=device[1], is_muzero=True)
            else:
                new_mcts = MCTS(gameEngine, device=device[1])
                
            # 학습된 모델을 가져옴.
            newPlayer = NNPlayer(new_nnet.to(device[1]), new_mcts, id=f'Step{i:03}')

            # 학습된 모델과 랜덤 모델, 기존 모델, 최고 모델을 비교하여 결과 출력
            results = pit(
                newPlayer,
                {
                    'random': randomPlayer,
                    'self-play': oldPlayer,
                    'best': bestPlayer,
                },
                gameEngine,
                n_games=n_games_eval,
                progress=(progress, i),
                step=i+1
            )
            print(results)

            # 학습된 모델을 기존 모델로 업데이트
            nnet = new_nnet
            oldPlayer = newPlayer

            if i % 5 == 0:
                # MuZero 모델일 때는 losses가 (value_loss, policy_loss) 튜플 형태로 반환됨
                if config['use_muzero']:
                    plt.plot([v_loss + p_loss for v_loss, p_loss in losses], label=f'Iteration {i}')
                else:
                    plt.plot([loss.sum() for loss in losses], label=f'Iteration {i}')
            
            # MuZero 모델일 때는 마지막 손실값 처리 방식 변경
            if config['use_muzero']:
                steps.append((results, losses[-1]))
            else:
                steps.append((results, losses[-1]))
            progress.update(task, advance=1, arg1=f'{results_to_score(results["random"])}(random) {results_to_score(results["self-play"])}(self) {results_to_score(results["best"])}(best)')

    plt.legend()
    plt.savefig(f'plots/loss/{config["game"]}-{nnet.id()}.svg')

    fig, ax1 = plt.subplots()
    fig.set_size_inches((1920/dpi, 1080/dpi))
    fig.set_dpi(dpi)
    ax1.set_xlabel('step')

    color = 'tab:red'
    ax1.set_ylabel('score', color=color)
    results = {
        tp: [step[tp] for step, _ in steps]
        for tp in steps[0][0]
    }
    ls = {
        'random': '--',
        'self-play': ':',
        'best': '-'
    }
    ax1.plot([0.5] * len(results['best']), ':', color='grey')
    for tp, scores in results.items():
        ax1.plot([float(results_to_score(result)) for result in scores], ls[tp], color=color, label=tp)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0., 1.])
    plt.legend()

    ax2 = ax1.twinx()  # instantiate a second ax that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('loss', color=color)
    
    # MuZero 모델과 일반 모델의 losses 형식이 다름에 따라 처리 방식 분기
    if config['use_muzero']:
        # MuZero 모델은 (value_loss, policy_loss) 튜플 형태
        ax2.plot([l1 for _, (l1, _) in steps], ':', color=color, label='value loss')
        ax2.plot([l2 for _, (_, l2) in steps], '--', color=color, label='policy loss')
        ax2.plot([l1+l2 for _, (l1, l2) in steps], color=color, label='total loss')
    else:
        # AlphaZero 모델은 numpy 배열 형태 [mse_loss, cross_entropy_loss]
        ax2.plot([l1 for _, (l1, _) in steps], ':', color=color, label='mse')
        ax2.plot([l2 for _, (_, l2) in steps], '--', color=color, label='cross entropy')
        ax2.plot([l1+l2 for _, (l1, l2) in steps], color=color, label='loss')
    
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend()
    ax1.legend(loc='upper left')
    ax2.legend(loc='lower left')
    plt.savefig(f'plots/steps/{config["game"]}-{nnet.id()}.svg')

    return nnet

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    device, device2 = "cpu", "cpu"
    if torch.cuda.is_available():
        gpus = get_freer_gpu(2)
        if len(gpus) >= 2:
            a, b = gpus
        else:
            a, b = gpus[0], gpus[0]
        device = f"cuda:{a}" # Training Device
        # device2 = f"cuda:{b}" # MCTS Device

    if config['game'] == 'ttt':
        if config['use_muzero']:
            # MuZero 모드에서는 MuZeroNN 인스턴스를 생성하고 게임 엔진에 전달
            muzero_nn = MuZeroNN(hidden_size=32).to(device2)
            gameEngine = TicTacToe_MuZero(nn=muzero_nn)
            # MuZero 모드에서는 모델 타입 설정 불필요 (MuZeroNN 사용)
        else:
            gameEngine = TicTacToe()
            config['model'] = BaseNN
    else:
        raise NotImplementedError
    
    nnet = finalnet(gameEngine, iterations=config['n_iter'], n_games_eval=config['n_games_eval'], device=[device, device2])
    
    # 모델 저장
    model_type = "muzero" if config['use_muzero'] else "alphazero"
    torch.save(nnet.state_dict(), f'./models/{config["game"]}_{model_type}_best.pt')
