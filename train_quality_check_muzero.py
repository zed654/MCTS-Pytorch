import torch
import numpy as np
from gameEngine import TicTacToe_MuZero
from models import MuZeroNN

def test_muzero_components(model_path='./models/ttt_muzero_best.pt'):
    """
    MuZero 모델의 각 구성요소(Representation, Dynamics, Prediction)가 
    잘 학습되었는지 테스트하는 함수
    """
    print("==== MuZero 모델 테스트 ====")
    
    # 모델 불러오기
    try:
        model = MuZeroNN(hidden_size=32)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"✓ 모델 '{model_path}' 불러오기 성공")
    except FileNotFoundError:
        print(f"✗ 모델 파일 '{model_path}'을 찾을 수 없습니다.")
        print("  mcts_2.py를 먼저 실행하여 모델을 학습시키세요.")
        return
    
    # 게임 엔진 생성
    game_engine = TicTacToe_MuZero(nn=model)
    
    # 테스트 케이스들
    test_cases = [
        {'name': '빈 보드', 'moves': []},
        {'name': '중앙에 X', 'moves': [(1, 1)]},
        {'name': 'X와 O 두 개', 'moves': [(1, 1), (0, 0)]},
        {'name': '대각선 두 개', 'moves': [(0, 0), (1, 0), (1, 1), (2, 0), (2, 2)]},
    ]
    
    total_mse = 0.0
    test_count = 0
    
    # 각 테스트 케이스에 대해 실행
    for test_case in test_cases:
        print(f"\n=== 테스트: {test_case['name']} ===")
        
        # 게임 초기화
        game = game_engine.startingPosition()
        
        # 기존 이동 적용
        for move in test_case['moves']:
            game = game_engine.makeMove(game, move)
        
        # 현재 게임 상태 출력
        game_engine.print(game, ("테스트", 0), "테스트 게임")
        
        # 1. representation_nn 테스트
        state = game[-1]
        latent = model.representation_nn(state)
        print(f"→ 추상 상태 크기: {latent.shape}")
        
        # 가능한 모든 이동에 대해 dynamics_nn 테스트
        legal_moves = game_engine.legalMoves(game)
        if legal_moves:
            print(f"\n→ 가능한 이동 {len(legal_moves)}개에 대해 테스트:")
            
            for move in legal_moves:
                # 2. dynamics_nn으로 다음 상태 예측
                # move가 (i,j) 형태면 인덱스로 변환
                if isinstance(move, tuple) and len(move) == 2:
                    action_idx = move[0] * 3 + move[1]
                else:
                    action_idx = move
                
                next_latent, reward = model.dynamics_nn(latent, action_idx)
                
                # 3. 실제로 수를 두고 실제 상태와 비교
                next_game = game_engine.makeMove(game.copy(), move)
                actual_latent = model.representation_nn(next_game[-1])
                
                # 예측된 latent와 실제 latent 비교
                mse = torch.nn.functional.mse_loss(next_latent, actual_latent).item()
                total_mse += mse
                test_count += 1
                
                # prediction_nn 테스트 (가치와 정책 예측)
                value, policy = model.prediction_nn(latent)
                policy_probs = torch.nn.functional.softmax(policy, dim=-1)
                
                # 결과 출력
                move_name = f"({move[0]}, {move[1]})"
                print(f"  이동 {move_name}:")
                print(f"    - 보상 예측: {reward.item():.4f}")
                print(f"    - 가치 예측: {value.item():.4f}")
                print(f"    - 정책 확률: {policy_probs[action_idx].item():.4f}")
                print(f"    - 예측 오차(MSE): {mse:.6f}")
                
                # 모델 정확도 등급 부여
                if mse < 0.05:
                    grade = "매우 좋음 ✓✓"
                elif mse < 0.1:
                    grade = "좋음 ✓"
                elif mse < 0.3:
                    grade = "보통 ⚠"
                else:
                    grade = "나쁨 ✗"
                print(f"    - 예측 품질: {grade}")
        else:
            print("→ 가능한 이동이 없습니다. 게임이 종료되었습니다.")
    
    # 종합 결과
    if test_count > 0:
        avg_mse = total_mse / test_count
        print("\n==== 종합 결과 ====")
        print(f"평균 예측 오차(MSE): {avg_mse:.6f}")
        
        # 전체 모델 성능 평가
        if avg_mse < 0.05:
            print("MuZero 모델 품질: 매우 좋음 ✓✓")
            print("이 모델은 dynamics_nn과 representation_nn이 매우 잘 학습되었습니다!")
        elif avg_mse < 0.1:
            print("MuZero 모델 품질: 좋음 ✓")
            print("이 모델은 dynamics_nn과 representation_nn이 잘 학습되었습니다.")
        elif avg_mse < 0.3:
            print("MuZero 모델 품질: 보통 ⚠")
            print("이 모델은 더 많은 학습이 필요할 수 있습니다.")
        else:
            print("MuZero 모델 품질: 나쁨 ✗")
            print("이 모델은 dynamics_nn과 representation_nn이 제대로 학습되지 않았습니다.")
    
    return avg_mse if test_count > 0 else None

if __name__ == "__main__":
    # 기본 모델 경로로 테스트 실행
    test_muzero_components()
    
    # 다른 모델을 테스트하려면 경로 지정
    # test_muzero_components('./models/다른_모델.pt') 