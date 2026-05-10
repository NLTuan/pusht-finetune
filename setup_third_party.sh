git clone https://github.com/huggingface/lerobot.git third_party/lerobot

cd third_party/lerobot
uv pip install -e ".[training, viz, smolvla, diffusion, pusht, pi]"

cd ../..