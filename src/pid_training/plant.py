# FILE: src/pid_training/plant.py (최종 해결 버전)
import torch

class FirstOrderPlant:
    """
    메모리 누수가 해결된 1차 지연 시스템 시뮬레이터입니다.
    """
    def __init__(self, ku: float = 2.0, tau: float = 5.0, initial_state: float = 0.0):
        self.ku = torch.tensor(ku, dtype=torch.float32)
        self.tau = torch.tensor(tau, dtype=torch.float32)
        # 내부 상태는 항상 계산 그래프와 분리된 순수 데이터여야 합니다.
        self.y = torch.tensor(initial_state, dtype=torch.float32)

    def reset(self):
        """시스템 상태를 초기값으로 리셋합니다."""
        self.y = torch.tensor(0.0)

    def step(self, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        오일러 방법을 사용하여 시스템 상태를 한 스텝 업데이트합니다.
        """
        # 1. 다음 스텝의 값을 계산합니다.
        #    이 `next_y` 텐서는 현재 계산 그래프에 연결되어 있습니다.
        next_y = self.y + (self.ku * u - self.y) / self.tau * dt
        
        # 2. 다음 반복에서 사용할 내부 상태(self.y)를 업데이트합니다.
        #    반드시 .detach()를 호출하여 계산 그래프로부터 분리해야 합니다.
        self.y = next_y.detach()
        
        # 3. loss 계산을 위해 그래프 연결이 유지된 텐서를 반환합니다.
        return next_y