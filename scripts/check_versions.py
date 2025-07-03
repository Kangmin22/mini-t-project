# FILE: scripts/check_versions.py (수정본)
import torch
import ray
import transformers
import peft
import datasets

print("="*50)
print("✅ 라이브러리 버전 검증 (안정 버전 스택)")
print("="*50)
print(f"PyTorch version: {torch.__version__}")
print(f"Ray version: {ray.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"PEFT version: {peft.__version__}")
print(f"Datasets version: {datasets.__version__}")
print("\n모든 핵심 라이브러리가 성공적으로 임포트되었습니다.")
print("="*50)