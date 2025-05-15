import sys
import time
import torch
import numpy as np
from gameEngine import TicTacToe, TicTacToe_MuZero, InvalidBoardState
from models import BaseNN, MuZeroNN
from mcts_2 import MCTS, NNPlayer

# play_against_ai.py에서 필요한 함수들 가져오기
def display_board(board):
    symbols = {0: ' ', 1: 'X', -1: 'O'}
    print("\n-------------")
    for i in range(3):
        print("|", end=" ")
        for j in range(3):
            cell = board[i*3+j]
            print(f"{symbols[cell]}", end=" | ")
        print("\n-------------")

def convert_move_to_coord(move):
    # 이미 (row, col) 형태로 되어 있음
    return f"({move[0]}, {move[1]})"

def play_game():
    """
    사용자가 AI와 틱택토 게임을 진행합니다.
    """
    print("==== 틱택토 게임 AI 대결 ====")
    
    # AI 모델 선택
    print("\n사용할 AI 모델을 선택하세요:")
    print("1. AlphaZero (정책+가치 네트워크)")
    print("2. MuZero (표현+동역학+예측 네트워크)")
    
    while True:
        try:
            model_choice = int(input("모델 번호를 입력하세요 (1 또는 2): "))
            if model_choice in [1, 2]:
                break
            print("1 또는 2를 입력해주세요.")
        except ValueError:
            print("유효한 숫자를 입력해주세요.")
    
    use_muzero = (model_choice == 2)
    
    # 게임 엔진 생성
    if use_muzero:
        # MuZero는 TicTacToe_MuZero 게임 엔진 필요
        muzero_model = MuZeroNN(hidden_size=32)
        gameEngine = TicTacToe_MuZero(nn=muzero_model)
        model_path = './models/ttt_muzero_best.pt'
        model_name = "MuZero"
    else:
        # AlphaZero는 일반 TicTacToe 게임 엔진 사용
        gameEngine = TicTacToe()
        model_path = './models/ttt_alphazero_best.pt'
        model_name = "AlphaZero"
    
    # 모델 불러오기
    if use_muzero:
        model = MuZeroNN(hidden_size=32)
    else:
        model = BaseNN()
        
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"✓ {model_name} 모델 불러오기 성공")
    except FileNotFoundError:
        print(f"✗ {model_path} 모델 파일을 찾을 수 없습니다.")
        print("학습된 모델이 없다면 먼저 mcts_2.py를 실행하여 모델을 학습시켜주세요.")
        return
    model.to("cpu")
    model.eval()  # 평가 모드로 설정
    
    # AI 플레이어 생성 (MuZero 모드 설정)
    mcts_instance = MCTS(gameEngine, n_sim=100, device="cpu", is_muzero=use_muzero)
    ai_player = NNPlayer(model, mcts_instance, id=f"{model_name} AI")
    
    # 선공/후공 선택
    print("\n선공/후공을 선택하세요:")
    print("1. 사용자 선공 (X)")
    print("2. AI 선공 (X)")
    
    while True:
        try:
            turn_choice = int(input("선택 (1 또는 2): "))
            if turn_choice in [1, 2]:
                break
            print("1 또는 2를 입력해주세요.")
        except ValueError:
            print("유효한 숫자를 입력해주세요.")
    
    user_first = (turn_choice == 1)
    
    # 게임 초기화
    game = gameEngine.startingPosition()
    current_player = 1  # X 먼저 시작
    
    # 게임 시작
    print(f"\n=== 게임 시작! {model_name} AI와의 대결 ===")
    print("사용자: O" if not user_first else "사용자: X")
    print(f"{model_name} AI: X" if not user_first else f"{model_name} AI: O")
    
    # 게임이 끝날 때까지 진행
    turn = 0
    while True:
        turn += 1
        
        # 게임이 끝났는지 확인
        is_game_over = gameEngine.gameOver(game)
        if is_game_over:
            break
            
        print(f"\n턴 {turn} (플레이어: {'X' if current_player == 1 else 'O'})")
        gameEngine.print(game, ("게임", turn), f"사용자/{model_name} AI")
        
        is_user_turn = (user_first and current_player == 1) or (not user_first and current_player == -1)
        
        if is_user_turn:  # 사용자 플레이어 턴
            legal_moves = gameEngine.legalMoves(game)
            
            # 가능한 수 표시
            print("가능한 수:")
            for i, move in enumerate(legal_moves):
                print(f"{i+1}. {convert_move_to_coord(move)}")
            
            # 사용자 입력 받기
            while True:
                try:
                    choice = int(input("두실 위치의 번호를 선택하세요: ")) - 1
                    if 0 <= choice < len(legal_moves):
                        move = legal_moves[choice]
                        break
                    else:
                        print(f"1부터 {len(legal_moves)}까지의 숫자를 입력해주세요.")
                except ValueError:
                    print("유효한 숫자를 입력해주세요.")
            
            print(f"사용자가 {convert_move_to_coord(move)}에 {'X' if current_player == 1 else 'O'}를 두었습니다.")
        else:  # AI 턴
            print(f"{model_name} AI가 생각 중...")
            move = ai_player.best_move(game)
            print(f"{model_name} AI가 {convert_move_to_coord(move)}에 {'X' if current_player == 1 else 'O'}를 두었습니다.")
        
        game = gameEngine.makeMove(game, move)
        current_player = -current_player
    
    # 게임 결과 출력
    print("\n게임 결과:")
    gameEngine.print(game, ("결과", 0), f"사용자/{model_name} AI")
    
    # X(1)가 이기면 InvalidBoardState 예외가 발생하므로, 이를 처리
    try:
        outcome = gameEngine.outcome(game)
        # -1이면 O(후공)이 이긴 것, 0이면 무승부
        if outcome == 0:
            print("게임 결과: 무승부입니다!")
        else:  # outcome == -1
            winner_is_user = (not user_first)  # O가 이겼으면 후공이 승리
            winner = "사용자" if winner_is_user else f"{model_name} AI"
            print(f"게임 결과: {winner}가 이겼습니다!")
            print_winning_lines(game[-1])
    except InvalidBoardState:
        # X(1)가 이긴 경우
        winner_is_user = user_first  # X가 이겼으면 선공이 승리
        winner = "사용자" if winner_is_user else f"{model_name} AI"
        print(f"게임 결과: {winner}가 이겼습니다!")
        print_winning_lines(game[-1])
    
    print("\n게임 종료!")
    
    # 다시 플레이 여부
    while True:
        play_again = input("\n다시 플레이하시겠습니까? (y/n): ").lower()
        if play_again in ['y', 'n']:
            break
        print("y 또는 n을 입력해주세요.")
    
    if play_again == 'y':
        play_game()

# 승리 라인 출력 함수
def print_winning_lines(board):
    # 보드를 3x3 형태로 변환
    if isinstance(board, torch.Tensor):
        b = board.numpy().reshape(3, 3)
    else:
        b = board.reshape(3, 3)
    
    print("\n승리 조건:")
    
    # 가로줄 체크
    for i in range(3):
        if abs(sum(b[i, :])) == 3:
            print(f"가로 {i+1}번째 줄 완성!")
            
    # 세로줄 체크
    for i in range(3):
        if abs(sum(b[:, i])) == 3:
            print(f"세로 {i+1}번째 줄 완성!")
            
    # 대각선 체크
    if abs(b[0, 0] + b[1, 1] + b[2, 2]) == 3:
        print("좌상단-우하단 대각선 완성!")
    if abs(b[0, 2] + b[1, 1] + b[2, 0]) == 3:
        print("우상단-좌하단 대각선 완성!")

if __name__ == "__main__":
    play_game() 