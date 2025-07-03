# FILE: src/pid_training/pid_net.py (수정본)
import torch
import torch.nn as nn

class PIDNet(nn.Module):
    """
    학습 가능한 미분 기반 PID 제어기.
    Kp, Ki, Kd 게인 값을 학습 가능한 파라미터로 가집니다.
    """
    def __init__(self, kp: float, ki: float, kd: float):
        super().__init__()
        self.kp = nn.Parameter(torch.tensor(kp, dtype=torch.float32))
        self.ki = nn.Parameter(torch.tensor(ki, dtype=torch.float32))
        self.kd = nn.Parameter(torch.tensor(kd, dtype=torch.float32))
        
        # (수정된 부분) torch.zeros(1) 대신 torch.tensor(0.0)을 사용하여
        # 내부 상태 변수들을 스칼라로 초기화합니다.
        self.register_buffer('integral_term', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('prev_error', torch.tensor(0.0, dtype=torch.float32))

    def forward(self, current_value: torch.Tensor, target_value: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        PID 제어 출력을 계산합니다.

        Args:
            current_value (torch.Tensor): 현재 시스템 값.
            target_value (torch.Tensor): 목표 값.
            dt (torch.Tensor): 시간 간격 (delta time).

        Returns:
            torch.Tensor: 제어 출력 값.
        """
        error = target_value - current_value
        
        # 적분항 (Integral)
        self.integral_term = self.integral_term + error * dt
        
        # 미분항 (Derivative)
        derivative_term = (error - self.prev_error) / dt
        
        # 제어 출력 계산
        output = (self.kp * error) + \
                 (self.ki * self.integral_term) + \
                 (self.kd * derivative_term)
        
        # 다음 스텝을 위해 현재 에러를 저장 (그래디언트 흐름을 끊어야 함)
        self.prev_error = error.detach()
        
        return output

    def reset(self):
        """시뮬레이션 에피소드 시작 시 내부 상태를 초기화합니다."""
        self.integral_term.zero_()
        self.prev_error.zero_()