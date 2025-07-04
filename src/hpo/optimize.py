# FILE: src/hpo/optimize.py (본격 학습 버전)
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import ray
from ray import tune
from ray.tune import Trainable
from ray.air import RunConfig
from ray.tune.search.optuna import OptunaSearch
import gc
import os
import time

from src.pid_training.pid_net import PIDNet
from src.pid_training.plant import FirstOrderPlant

class PIDTrainable(Trainable):
    def setup(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n--- Starting Trial on device: {self.device} ---")
        self.pid_net = PIDNet(kp=config["kp"], ki=config["ki"], kd=config["kd"]).to(self.device)
        self.plant = FirstOrderPlant(ku=2.0, tau=5.0, device=self.device)
        self.optimizer = optim.Adam(self.pid_net.parameters(), lr=config["lr"])
        self.current_val = torch.tensor(0.0).to(self.device)
        self.target_val = torch.tensor(10.0).to(self.device)
        self.dt = torch.tensor(0.1).to(self.device)

    def step(self):
        total_loss = 0.0
        steps = 200
        self.pid_net.reset()
        self.plant.reset()
        for _ in range(steps):
            u = self.pid_net(self.current_val, self.target_val, self.dt)
            next_val = self.plant.step(u, self.dt)
            loss = (self.target_val - next_val) ** 2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            self.current_val = next_val.detach()
        return {"loss": total_loss / steps}

    def save_checkpoint(self, checkpoint_dir: str):
        return {"status": "ok"}

    def load_checkpoint(self, checkpoint: dict):
        pass
    
    def cleanup(self):
        del self.pid_net, self.plant, self.optimizer
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

def run_hpo():
    try:
        ray.init(ignore_reinit_error=True)
        search_space = { "lr": tune.loguniform(1e-4, 1e-1), "kp": tune.uniform(0.1, 5.0), "ki": tune.uniform(0.01, 2.0), "kd": tune.uniform(0.01, 2.0) }
        search_alg = OptunaSearch(metric="loss", mode="min")
        
        # (★★★★★ 핵심 수정 ★★★★★)
        # 각 실험이 step()을 10번 반복하도록 training_iteration을 10으로 늘립니다.
        run_config = RunConfig(
            name="PIDTrainable_Full_Training",
            stop={"training_iteration": 10},
        )
        
        trainable_with_resources = tune.with_resources(PIDTrainable, {"cpu": 1, "gpu": 1})
        
        tuner = tune.Tuner(
            trainable_with_resources,
            param_space=search_space,
            tune_config=tune.TuneConfig(search_alg=search_alg, num_samples=5),
            run_config=run_config,
        )
        print("HPO를 시작합니다 (각 Trial은 10회 반복 학습)...")
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