git clone https://github.com/huggingface/lerobot.git third_party/lerobot

cd third_party/lerobot
uv pip install --index-strategy unsafe-best-match -e ".[training, viz, smolvla, diffusion, pusht, pi]"

cd ../..