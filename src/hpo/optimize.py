# FILE: src/hpo/optimize.py (Double Free 해결 최종 버전)
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import ray
from ray import tune
from ray.tune import Trainable
from ray.tune.search.optuna import OptunaSearch
import gc
import os
import time

from src.pid_training.pid_net import PIDNet
from src.pid_training.plant import FirstOrderPlant

class PIDTrainable(Trainable):
    def setup(self, config):
        torch.autograd.set_detect_anomaly(True)
        self.config = config
        self.pid_net = PIDNet(kp=config["kp"], ki=config["ki"], kd=config["kd"])
        self.plant = FirstOrderPlant(ku=2.0, tau=5.0)
        self.optimizer = optim.Adam(self.pid_net.parameters(), lr=config["lr"])
        self.current_val = torch.tensor(0.0)
        self.target_val = torch.tensor(10.0)
        self.dt = torch.tensor(0.1)
        self.steps_per_epoch = 20

    def step(self):
        total_loss = 0.0
        self.pid_net.reset()
        self.plant.reset()

        for _ in range(self.steps_per_epoch):
            u = self.pid_net(self.current_val, self.target_val, self.dt)
            next_val = self.plant.step(u, self.dt)
            loss = (self.target_val - next_val) ** 2
            
            self.optimizer.zero_grad()

            # (★★★★★ 최종 해결책 ★★★★★)
            # backward()를 호출할 loss 텐서를 안전하게 복제(clone)하여 사용합니다.
            # 이렇게 하면 원본 loss 텐서와 다른 메모리 공간을 사용하게 되어
            # 메모리 이중 해제(double free) 오류를 방지할 수 있습니다.
            loss_clone = loss.clone()
            loss_clone.backward()

            self.optimizer.step()
            
            total_loss += loss.item()
            
            current_val_detached = next_val.detach().clone()
            self.current_val = current_val_detached

        gc.collect()
        return {"loss": total_loss / self.steps_per_epoch}

    def cleanup(self):
        del self.pid_net, self.plant, self.optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(1.0)

def run_hpo():
    try:
        ray.init(local_mode=True, ignore_reinit_error=True)
        search_space = {
            "lr": tune.loguniform(1e-4, 1e-1),
            "kp": tune.uniform(0.1, 5.0),
            "ki": tune.uniform(0.01, 2.0),
            "kd": tune.uniform(0.01, 2.0),
        }
        search_alg = OptunaSearch(metric="loss", mode="min")
        tuner = tune.Tuner(
            PIDTrainable,
            param_space=search_space,
            tune_config=tune.TuneConfig(search_alg=search_alg, num_samples=5),
        )
        print("HPO를 최종 안정화 버전으로 시작합니다...")
        results = tuner.fit()
        best_result = results.get_best_result(metric="loss", mode="min")
        print("\n=======================================================")
        print(" HPO 완료!")
        print(f"  최소 손실 (Loss): {best_result.metrics['loss']:.4f}")
        print("  최적 하이퍼파라미터:")
        for param, value in best_result.config.items():
            print(f"    - {param}: {value:.4f}")
        print("=======================================================")
    finally:
        ray.shutdown()

if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    run_hpo()