CUDA_VISIBLE_DEVICES=3 python -u -O train.py  -n "Pong-baseline-pssm-100k"   -seed 0   -config_path "config_files/PWM.yaml"   -env_name "ALE/Pong-v5"   -device "cuda:0"
CUDA_VISIBLE_DEVICES=3 python -u -O eval.py  -env_name "ALE/Pong-v5" -run_name "Pong-baseline-pssm-100k"   -seed 0   -config_path "config_files/PWM.yaml"   -device "cuda:0"

CUDA_VISIBLE_DEVICES=2 python -u -O train.py  -n "Pong-ours-rwkv-100k"   -seed 0   -config_path "config_files/PWM.yaml"   -env_name "ALE/Pong-v5"   -device "cuda:0" --use_rwkv 
CUDA_VISIBLE_DEVICES=2 python -u -O eval.py  -env_name "ALE/Pong-v5" -run_name "Pong-ours-rwkv-100k"   -seed 0   -config_path "config_files/PWM.yaml"   -device "cuda:0"


