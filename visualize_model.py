import torch
from torchviz import make_dot
from models import BaseNN

# 모델 인스턴스 생성
model = BaseNN()

# 더미 입력 생성 (배치 크기 1, 9개의 입력)
x = torch.randn(1, 9)

# 모델의 출력 계산
v, p = model(x)

# 계산 그래프 생성
dot = make_dot(v, params=dict(model.named_parameters()))

# 그래프 저장
dot.render("model_architecture/model_architecture", format="png", cleanup=True) 