# FILE: src/pid_training/plant.py (GPU 지원 버전)
import torch

class FirstOrderPlant:
    def __init__(self, ku: float = 2.0, tau: float = 5.0, initial_state: float = 0.0, device="cpu"):
        self.device = device
        # 모든 텐서를 지정된 장치(device)로 보냅니다.
        self.ku = torch.tensor(ku, dtype=torch.float32).to(self.device)
        self.tau = torch.tensor(tau, dtype=torch.float32).to(self.device)
        self.state = torch.tensor(initial_state, dtype=torch.float32, requires_grad=False).to(self.device)

    def reset(self, initial_state: float = 0.0):
        self.state = torch.tensor(initial_state, dtype=torch.float32, requires_grad=False).to(self.device)

    def step(self, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        dydt = (-self.state + self.ku * u) / self.tau
        new_state = self.state + dt * dydt
        self.state = new_state.detach()
        return new_state