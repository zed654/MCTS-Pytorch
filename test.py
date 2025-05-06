import torch

state = torch.tensor([
  [0, 1, 0],
  [0, -1, 0],
  [0, 0, 0]
])
print(hash(tuple(state.numpy().flatten()))) 