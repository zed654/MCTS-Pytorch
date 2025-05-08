import sys
import time
import torch
import numpy as np
from gameEngine import TicTacToe
from models import BaseNN
from mcts import MCTS, NNPlayer

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

def test_game_functionality():
    """
    틱택토 게임 기능을 자동으로 테스트합니다.
    """
    print("==== 틱택토 게임 기능 테스트 ====")
    
    # 게임 엔진 생성
    gameEngine = TicTacToe()
    
    # 모델 불러오기
    model = BaseNN()
    try:
        model.load_state_dict(torch.load('./models/tttbest.pt'))
        print("✓ 모델 불러오기 성공")
    except FileNotFoundError:
        print("✗ 모델 파일을 찾을 수 없습니다.")
        return
    model.to("cpu")
    
    # AI 플레이어 생성
    ai_player = NNPlayer(model, MCTS(gameEngine), id="AI")
    
    # 게임 초기화
    game = gameEngine.startingPosition()
    print("✓ 게임 초기화 성공")
    
    # 게임판 출력 테스트
    print("\n게임판 출력 테스트:")
    gameEngine.print(game, ("테스트", 0), "테스트/AI")
    
    # AI 의사결정 테스트
    print("\nAI 의사결정 테스트:")
    ai_move = ai_player.best_move(game)
    print(f"AI가 선택한 수: {ai_move} (좌표: {convert_move_to_coord(ai_move)})")
    
    # 게임 진행 테스트
    print("\n게임 진행 테스트:")
    game = gameEngine.makeMove(game, ai_move)
    gameEngine.print(game, ("테스트", 0), "테스트/AI")
    
    # 게임 판정 테스트
    print("\n게임 판정 테스트:")
    is_game_over = gameEngine.gameOver(game)
    print(f"게임 종료 여부: {is_game_over}")
    
    # 간단한 게임 시뮬레이션
    print("\n간단한 게임 시뮬레이션:")
    game = gameEngine.startingPosition()
    current_player = 1  # X 먼저 시작
    
    # 게임이 끝날 때까지 진행
    turn = 0
    while not gameEngine.gameOver(game):
        turn += 1
            
        print(f"\n턴 {turn} (플레이어: {'X' if current_player == 1 else 'O'})")
        gameEngine.print(game, ("테스트", turn), "테스트/AI")
        
        if (turn-1) % 2 == 0:  # 사용자 플레이어 턴
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
            
            print(f"플레이어가 {convert_move_to_coord(move)}에 {'X' if current_player == 1 else 'O'}를 두었습니다.")
        else:  # AI 턴
            print("AI가 생각 중...")
            move = ai_player.best_move(game)
            print(f"AI가 {convert_move_to_coord(move)}에 {'X' if current_player == 1 else 'O'}를 두었습니다.")
        
        game = gameEngine.makeMove(game, move)
        current_player = -current_player
    
    # 게임 결과 출력
    print("\n게임 결과:")
    gameEngine.print(game, ("결과", 0), "플레이어/AI")
    
    if gameEngine.gameOver(game):
        outcome = gameEngine.outcome(game)
        if outcome == 0:
            if 0 not in game[-1]:  # 모든 칸이 채워졌는지 확인
                print("게임 결과: 무승부입니다!")
            else:
                print("게임이 아직 끝나지 않았습니다.")
        else:
            # outcome은 마지막에 둔 플레이어 기준이므로 현재 플레이어와 반대
            winner = "AI" if current_player == 1 else "플레이어"
            print(f"게임 결과: {winner}가 이겼습니다!")
    else:
        print("게임이 아직 끝나지 않았습니다.")
    
    print("\n테스트 완료!")
    return True

if __name__ == "__main__":
    test_result = test_game_functionality()
    if test_result:
        print("✓ 모든 테스트가 성공적으로 완료되었습니다.")
    else:
        print("✗ 테스트 중 문제가 발생했습니다.") 