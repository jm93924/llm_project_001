# install -r requirements.txt 하기 전에 아래 코드 먼저.
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# SFTConfig 변수명 확인
python -c "from trl import SFTConfig; import inspect; print(inspect.signature(SFTConfig))"