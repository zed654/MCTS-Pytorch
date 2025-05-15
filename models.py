from scipy.special import softmax
import time
import torch
import torch.utils.data
import torch.utils.data.dataset
import torch.autograd
import numpy as np

from gameEngine import TicTacToeMoves
Ms = TicTacToeMoves

class BaseNN(torch.nn.Module):
    def __init__(self, n_layers=3, hidden_size=32, n_iter=None, batch_size=None, lr=None):
        super().__init__()
        self.uid = time.strftime("%Y-%m-%d:%Hh%M")

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.n_iter = 100 if n_iter is None else n_iter
        self.batch_size = 1024 if batch_size is None else batch_size
        self.lr = 0.001 if lr is None else lr

        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(9, hidden_size)
        ] + [
            torch.nn.Linear(hidden_size, hidden_size)
            for _ in range(n_layers - 1)
        ])
        self.evaluation_output = torch.nn.Linear(hidden_size, 1)
        self.policy_output = torch.nn.Linear(hidden_size, 9)

    def forward(self, x):
        # 신경망의 순전파
        # 입력 -> 정책(policy)과 가치(value) 출력
        x = x.float()
        x = x.reshape(*x.shape[:-2], -1)
        for layer in self.layers:
            x = torch.nn.functional.leaky_relu(layer(x))
        return self.evaluation_output(x).squeeze(dim=-1), self.policy_output(x)

    def predict(self, x):
        # forward를 사용해서 예측
        # state -> (value, policy) 반환
        v, P = self(x) # self(x) 는 forward 함수를 호출하는 것임.
        v = v.item() # 예측된 가치 값을 스칼라로 변환
        P = softmax(P.squeeze().detach().numpy()) # 예측된 정책 값을 소프트맥스 함수를 사용해 확률로 변환
        return v, P

    def fit(self, data, n_iter=None, batch_size=None, lr=None, progress=None):
        # 학습 데이터로 신경망 학습하는 함수
        
        # 파라미터 값이 주어지면 클래스 변수 업데이트, 아니면 기본값 사용
        if n_iter is not None:
            self.n_iter = n_iter  # 학습 반복 횟수 설정

        if batch_size is not None:
            self.batch_size = batch_size  # 배치 크기 설정

        if lr is not None:
            self.lr = lr  # 학습률 설정 (여기서 조정하면 loss 감소 속도 바꿀 수 있음)

        self.train()  # 신경망을 학습 모드로 설정 (드롭아웃, 배치정규화 등 활성화)

        # 학습 데이터 준비: 상태(V), 정책(TP), 가치(TV)로 분리
        V, TP, TV = zip(*data)  # data는 (상태, 정책, 가치) 튜플의 리스트
        V = torch.stack(V)  # 상태 텐서 - 게임판 상태
        TP = torch.stack(TP)  # 정책 텐서 - 어떤 수를 둘지에 대한 확률 분포
        TV = torch.stack(TV)  # 가치 텐서 - 게임 결과 (-1, 0, 1)
        
        # PyTorch 데이터셋과 데이터로더 생성 (미니배치 학습 위함)
        dataset = torch.utils.data.dataset.TensorDataset(V, TP, TV)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)  # 데이터 섞어서 학습

        # Adam 옵티마이저 생성 - 학습률(lr)이 여기서 설정됨
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)  # 학습률 낮추면 더 천천히 학습됨

        # 손실 함수 정의 - MSE(평균제곱오차)와 Cross Entropy(교차 엔트로피) 사용
        def criterion(v, p, tv, tp):
            return (
                torch.nn.functional.mse_loss(v, tv),  # 가치 예측 손실 (회귀)
                torch.nn.functional.cross_entropy(p, tp)  # 정책 예측 손실 (분류)
            )

        # 진행 상황 표시용 설정
        task = None
        if progress is not None:
            task = progress[0].add_task("Training", total=n_iter, arg1_n="loss", arg1="N/A")
        losses = []  # 각 반복마다의 손실값 저장할 리스트
        
        # 학습 반복 시작
        for _ in range(self.n_iter):
            cl = np.zeros(2)  # 현재 반복의 손실값 [가치 손실, 정책 손실]
            
            # 미니배치 학습
            for s, tp, tv in loader:
                optimizer.zero_grad()  # 그래디언트 초기화
                v, p = self(s)  # 신경망에 상태 입력하여 가치와 정책 예측
                loss = criterion(v, p, tv, tp)  # 손실 계산
                
                # 현재 배치의 손실값 누적
                for i in range(2):
                    cl[i] += loss[i].item() * len(s)
                
                loss = loss[0] + loss[1]  # 두 손실 합침 (여기서 가중치 조절 가능!)
                loss.backward()  # 역전파 - 그래디언트 계산
                optimizer.step()  # 가중치 업데이트
            
            cl /= len(data)  # 평균 손실값 계산
            losses.append(cl)  # 손실값 저장
            
            # 진행 상황 업데이트
            if progress is not None:
                progress[0].update(task, advance=1, arg1=f'{cl.sum():.5f} = {cl[0]:.5f} + {cl[1]:.5f}')

        self.eval()  # 학습 완료 후 평가 모드로 전환

        return losses  # 모든 반복의 손실값 반환

    def id(self):
        # 모델의 고유 식별자 반환
        # 저장된 모델 구분용
        return f'BaseNN-{self.n_layers}-{self.hidden_size}-{self.n_iter}-{self.batch_size}-{self.lr}_{self.uid}'


# MuZero 신경망 모델 - 별도 클래스로 구현
class MuZeroNN(torch.nn.Module):
    def __init__(self, hidden_size=32, n_iter=None, batch_size=None, lr=None):
        super().__init__()
        self.uid = time.strftime("%Y-%m-%d:%Hh%M")
        
        self.hidden_size = hidden_size
        self.n_iter = 100 if n_iter is None else n_iter
        self.batch_size = 1024 if batch_size is None else batch_size
        self.lr = 0.001 if lr is None else lr
        
        # MuZero의 세 가지 핵심 모델
        self.representation_nn = ObsNN(hidden_size)  # 관측 -> 추상 상태 표현
        self.dynamics_nn = DynamicsNN(hidden_size)   # 추상 상태 + 액션 -> 다음 추상 상태 + 보상
        self.prediction_nn = PredictionNN(hidden_size)  # 추상 상태 -> 정책 + 가치
    
    def predict(self, x, device=None):
        # 추상 상태에서 정책과 가치 예측
        x = x.to(device) if device is not None else x
        v, P = self.prediction_nn(x)
        v = v.item()  # 예측된 가치 값을 스칼라로 변환
        P = softmax(P.squeeze().detach().numpy())  # 예측된 정책 값을 소프트맥스 함수를 사용해 확률로 변환
        return v, P
    
    def fit(self, data, n_iter=None, batch_size=None, lr=None, progress=None):
        # MuZero 학습 구현 (실제로는 더 복잡하지만 간단하게 구현)
        if n_iter is not None:
            self.n_iter = n_iter

        if batch_size is not None:
            self.batch_size = batch_size

        if lr is not None:
            self.lr = lr

        self.train()

        # backward compatibility: AlphaZero-style data
        is_muzero_data = False
        if len(data) > 0 and isinstance(data[0], tuple) and len(data[0]) == 6:
            is_muzero_data = True

        if not is_muzero_data:
            # fallback to old style (state, policy, value)
            states, policies, values = zip(*data)
            states = torch.stack(states)
            policies = torch.stack(policies)
            values = torch.stack(values)
            dataset = torch.utils.data.dataset.TensorDataset(states, policies, values)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        else:
            # MuZero: (state, policy, value, action, reward, next_state)
            states, policies, values, actions, rewards, next_states = zip(*data)
            states = torch.stack(states)
            policies = torch.stack(policies)
            values = torch.stack(values)
            actions = torch.tensor(actions, dtype=torch.long, device=states.device)
            rewards = torch.stack(rewards).squeeze(-1)
            next_states = torch.stack(next_states)
            dataset = torch.utils.data.dataset.TensorDataset(states, policies, values, actions, rewards, next_states)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 각 네트워크별 옵티마이저 생성 (개선점 1: 네트워크별 최적화 분리)
        rep_optimizer = torch.optim.Adam(self.representation_nn.parameters(), lr=self.lr)
        dyn_optimizer = torch.optim.Adam(self.dynamics_nn.parameters(), lr=self.lr)
        pred_optimizer = torch.optim.Adam(self.prediction_nn.parameters(), lr=self.lr)

        task = None
        if progress is not None:
            task = progress[0].add_task("Training MuZero", total=n_iter, arg1_n="loss", arg1="N/A")

        losses = []
        for iter_idx in range(self.n_iter):
            rep_loss_total = 0.0  # 추가: representation loss 추적
            dyn_loss_total = 0.0
            consistency_loss_total = 0.0
            reward_loss_total = 0.0
            value_loss_total = 0.0
            policy_loss_total = 0.0
            total_batches = 0

            for batch in loader:
                if not is_muzero_data:
                    state_batch, policy_batch, value_batch = batch
                    
                    # prediction loss
                    pred_optimizer.zero_grad()
                    latent_states = self.representation_nn(state_batch)
                    pred_values, pred_policies = self.prediction_nn(latent_states)
                    value_loss = torch.nn.functional.mse_loss(pred_values, value_batch)
                    policy_loss = torch.nn.functional.cross_entropy(pred_policies, torch.argmax(policy_batch, dim=1))
                    prediction_loss = value_loss + policy_loss
                    prediction_loss.backward()
                    pred_optimizer.step()
                    
                    value_loss_total += value_loss.item() * state_batch.size(0)
                    policy_loss_total += policy_loss.item() * state_batch.size(0)
                    total_batches += state_batch.size(0)
                    continue

                state_batch, policy_batch, value_batch, action_batch, reward_batch, next_state_batch = batch
                batch_size = state_batch.size(0)
                
                # 1. Representation Network 학습
                rep_optimizer.zero_grad()
                latent_states = self.representation_nn(state_batch)
                target_next_latents = self.representation_nn(next_state_batch)
                
                # 2. Dynamics Network 학습 (실제 액션 기반)
                dyn_optimizer.zero_grad()
                pred_next_latents = []
                pred_rewards = []
                
                for i in range(batch_size):
                    # 개선점 2: 실제 게임에서 사용된 액션으로 학습
                    pred_next_latent, pred_reward = self.dynamics_nn(latent_states[i].detach(), action_batch[i].item())
                    pred_next_latents.append(pred_next_latent)
                    pred_rewards.append(pred_reward)
                
                pred_next_latents = torch.stack(pred_next_latents)
                pred_rewards = torch.stack(pred_rewards)
                
                # 개선점 3: Consistency Loss - 표현 일관성 손실 추가
                consistency_loss = torch.nn.functional.mse_loss(pred_next_latents, target_next_latents.detach())
                
                # 개선점 4: 실제 보상 값으로 학습
                reward_loss = torch.nn.functional.mse_loss(pred_rewards, reward_batch)
                
                dynamics_loss = consistency_loss + reward_loss
                dynamics_loss.backward()
                dyn_optimizer.step()
                
                # 3. Prediction Network 학습
                pred_optimizer.zero_grad()
                pred_values, pred_policies = self.prediction_nn(latent_states.detach())
                value_loss = torch.nn.functional.mse_loss(pred_values, value_batch)
                policy_loss = torch.nn.functional.cross_entropy(pred_policies, torch.argmax(policy_batch, dim=1))
                prediction_loss = value_loss + policy_loss
                prediction_loss.backward()
                pred_optimizer.step()
                
                # 4. Representation Network도 consistency loss로 학습
                rep_optimizer.zero_grad()
                latent_states = self.representation_nn(state_batch)
                pred_next_latents = []
                
                for i in range(batch_size):
                    pred_next_latent, _ = self.dynamics_nn(latent_states[i], action_batch[i].item())
                    pred_next_latents.append(pred_next_latent)
                
                pred_next_latents = torch.stack(pred_next_latents)
                target_next_latents = self.representation_nn(next_state_batch)
                rep_loss = torch.nn.functional.mse_loss(pred_next_latents.detach(), target_next_latents)
                rep_loss.backward()
                rep_optimizer.step()
                
                # 손실 합산
                rep_loss_total += rep_loss.item() * batch_size
                dyn_loss_total += dynamics_loss.item() * batch_size
                consistency_loss_total += consistency_loss.item() * batch_size
                reward_loss_total += reward_loss.item() * batch_size
                value_loss_total += value_loss.item() * batch_size
                policy_loss_total += policy_loss.item() * batch_size
                total_batches += batch_size

            if total_batches == 0:
                mean_losses = (0, 0)
            else:
                mean_losses = (
                    value_loss_total / total_batches,
                    policy_loss_total / total_batches
                )
            losses.append(mean_losses)
            if progress is not None:
                loss_str = f'V:{mean_losses[0]:.5f} P:{mean_losses[1]:.5f}'
                if is_muzero_data:
                    rep_loss_mean = rep_loss_total / total_batches if total_batches > 0 else 0
                    dyn_loss_mean = dyn_loss_total / total_batches if total_batches > 0 else 0
                    reward_loss_mean = reward_loss_total / total_batches if total_batches > 0 else 0
                    loss_str += f' R:{rep_loss_mean:.5f} D:{dyn_loss_mean:.5f} Rw:{reward_loss_mean:.5f}'
                progress[0].update(task, advance=1, arg1=loss_str)

        self.eval()
        return losses
    
    def id(self):
        # 모델의 고유 식별자 반환
        return f'MuZeroNN-{self.hidden_size}-{self.n_iter}-{self.batch_size}-{self.lr}_{self.uid}'


class ObsNN(torch.nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        # 관측 -> 추상 상태 표현 변환
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(9, hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
        ])
    
    def forward(self, observation):
        # 게임 관측을 받아 추상 상태 표현 반환
        x = observation.float()
        x = x.reshape(*x.shape[:-2], -1)
        for layer in self.layers:
            x = layer(x)
        return x  # 추상 표현 반환


class DynamicsNN(torch.nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        # 추상 상태 + 액션 -> 다음 추상 상태 + 보상 예측
        self.action_embedding = torch.nn.Linear(9, hidden_size)  # 액션 임베딩
        
        # 추상 상태와 액션을 합쳐서 다음 상태 예측
        self.dynamics_layers = torch.nn.ModuleList([
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
        ])
        
        # 보상 예측 레이어
        self.reward_head = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, latent_state, action):
        # 추상 상태와 액션을 받아 다음 추상 상태와 보상 예측
        # latent_state 디바이스 가져오기
        device = latent_state.device
        
        # 액션이 텐서가 아니면 텐서로 변환
        if isinstance(action, int) or isinstance(action, np.integer):
            # 액션을 원-핫 인코딩 (latent_state와 같은 디바이스에 생성)
            action_one_hot = torch.zeros(9, dtype=torch.float, device=device)
            action_one_hot[action] = 1.0
        else:
            # 이미 원-핫 인코딩된 액션 - 올바른 디바이스로 이동
            action_one_hot = action if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.float)
            action_one_hot = action_one_hot.to(device)
        
        # 액션 임베딩
        action_emb = self.action_embedding(action_one_hot)
        
        # 추상 상태와 액션 임베딩 합치기
        # 차원이 맞는지 확인
        if len(latent_state.shape) != len(action_emb.shape):
            # 배치 차원 추가 또는 제거
            if len(latent_state.shape) > len(action_emb.shape):
                action_emb = action_emb.unsqueeze(0)
            else:
                latent_state = latent_state.unsqueeze(0)
        
        try:
            x = torch.cat([latent_state, action_emb], dim=-1)
        except RuntimeError as e:
            # 디버깅 정보
            print(f"Error in dynamics forward: {e}")
            print(f"latent_state shape: {latent_state.shape}, action_emb shape: {action_emb.shape}")
            # 응급 처치 - 차원 맞추기
            if latent_state.shape[-1] != action_emb.shape[-1]:
                action_emb = action_emb.repeat(1, latent_state.shape[-1] // action_emb.shape[-1])
            x = torch.cat([latent_state, action_emb], dim=-1)
        
        # 다음 추상 상태 예측
        for layer in self.dynamics_layers:
            x = layer(x)
        
        # 보상 예측
        reward = self.reward_head(x)
        
        return x, reward.squeeze()  # 다음 추상 상태, 예측된 보상 반환


class PredictionNN(torch.nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        # 추상 상태 -> 정책 및 가치 출력
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.LeakyReLU(),
        ])
        
        # 출력 헤드들
        self.policy_output = torch.nn.Linear(hidden_size, 9)  # 정책(액션 확률) 출력
        self.value_output = torch.nn.Linear(hidden_size, 1)   # 가치 출력
    
    def forward(self, latent_state):
        # 추상 상태를 받아 정책과 가치 예측
        x = latent_state
        for layer in self.layers:
            x = layer(x)
        
        # 정책과 가치 출력
        policy = self.policy_output(x)
        value = self.value_output(x).squeeze(dim=-1)
        
        return value, policy

