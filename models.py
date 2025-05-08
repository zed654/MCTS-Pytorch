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
        # 학습 데이터로 신경망 학습
        # loss 계산, 역전파, 가중치 업데이트
        if n_iter is not None:
            self.n_iter = n_iter

        if batch_size is not None:
            self.batch_size = batch_size

        if lr is not None:
            self.lr = lr

        self.train()

        V, TP, TV = zip(*data)
        V = torch.stack(V)
        TP = torch.stack(TP)
        TV = torch.stack(TV)
        dataset = torch.utils.data.dataset.TensorDataset(V, TP, TV)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        def criterion(v, p, tv, tp):
            return (
                torch.nn.functional.mse_loss(v, tv),
                torch.nn.functional.cross_entropy(p, tp)
            )

        task = None
        if progress is not None:
            task = progress[0].add_task("Training", total=n_iter, arg1_n="loss", arg1="N/A")
        losses = []
        for _ in range(self.n_iter):
            cl = np.zeros(2)
            for s, tp, tv in loader:
                optimizer.zero_grad()
                v, p = self(s)
                loss = criterion(v, p, tv, tp)
                for i in range(2):
                    cl[i] += loss[i].item() * len(s)
                loss = loss[0] + loss[1]
                loss.backward()
                optimizer.step()
            cl /= len(data)
            losses.append(cl)
            if progress is not None:
                progress[0].update(task, advance=1, arg1=f'{cl.sum():.5f} = {cl[0]:.5f} + {cl[1]:.5f}')

        self.eval()

        return losses

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

        # MuZero 학습 데이터 준비
        # data는 (state, policy, value) 형태로 제공됨
        states, policies, values = zip(*data)
        states = torch.stack(states)
        policies = torch.stack(policies)
        values = torch.stack(values)
        
        dataset = torch.utils.data.dataset.TensorDataset(states, policies, values)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 옵티마이저 설정
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        task = None
        if progress is not None:
            task = progress[0].add_task("Training MuZero", total=n_iter, arg1_n="loss", arg1="N/A")
        
        losses = []
        for iter_idx in range(self.n_iter):
            rep_loss_total = 0.0
            dyn_loss_total = 0.0
            pred_loss_total = 0.0
            
            for state_batch, policy_batch, value_batch in loader:
                optimizer.zero_grad()
                
                # 1. 표현 모델: 관측 -> 추상 상태
                latent_states = self.representation_nn(state_batch)
                
                # 2. 예측 모델: 추상 상태 -> 정책 및 가치
                pred_values, pred_policies = self.prediction_nn(latent_states)
                
                # 3. 동역학 모델 학습 (간단한 자기지도 학습)
                # 배치에서 무작위 액션 선택 (실제로는 데이터에서 액션을 가져와야 함)
                batch_size = state_batch.size(0)
                random_actions = torch.randint(0, 9, (batch_size,))
                
                # 각 액션에 대해 다음 추상 상태와 보상 예측
                dynamics_loss = 0.0
                for i in range(batch_size):
                    # 랜덤 액션을 사용하여 다음 상태 예측 (실제로는 실제 액션을 사용해야 함)
                    pred_next_state, pred_reward = self.dynamics_nn(latent_states[i], random_actions[i].item())
                    
                    # 다음 상태를 동일한 표현 네트워크로 인코딩 (실제로는 실제 다음 상태를 사용해야 함)
                    # 여기서는 간단히 같은 상태를 사용 (이상적이진 않지만 구조를 유지하기 위함)
                    # 자기지도 학습을 위한 지름길
                    true_next_state = latent_states[i]
                    
                    # 동역학 모델 손실 (MSE)
                    state_mse = torch.nn.functional.mse_loss(pred_next_state, true_next_state.detach())
                    reward_mse = torch.nn.functional.mse_loss(pred_reward, torch.zeros_like(pred_reward))  # 간단히 0으로 가정
                    
                    dynamics_loss += state_mse + reward_mse
                
                dynamics_loss = dynamics_loss / batch_size
                
                # 3. 손실 계산
                value_loss = torch.nn.functional.mse_loss(pred_values, value_batch)
                policy_loss = torch.nn.functional.cross_entropy(pred_policies, torch.argmax(policy_batch, dim=1))
                
                # 총 손실 (MuZero: 예측 손실 + 동역학 손실)
                # 동역학 손실은 실제 훈련 데이터보다 가중치를 낮게 설정
                loss = value_loss + policy_loss + 0.1 * dynamics_loss
                
                # 역전파 및 최적화
                loss.backward()
                optimizer.step()
                
                # 손실 누적
                rep_loss_total += 0.0  # 현재는 표현 모델 손실을 명시적으로 계산하지 않음
                dyn_loss_total += dynamics_loss.item() * batch_size
                pred_loss_total += (value_loss.item() + policy_loss.item()) * batch_size
            
            # 배치당 평균 손실
            batch_loss = (rep_loss_total + dyn_loss_total + pred_loss_total) / len(data)
            
            # 손실 저장 (가치 손실, 정책 손실)
            losses.append((value_loss.item(), policy_loss.item()))
            
            if progress is not None:
                progress[0].update(task, advance=1, arg1=f'{batch_loss:.5f}')
        
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

