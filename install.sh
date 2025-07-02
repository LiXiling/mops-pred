ENVNAME="new_env"

conda create --name $ENVNAME && \
eval "$(conda shell.bash hook)" && \
conda activate $ENVNAME && \

# INSTALL CONDA PACKAGES
conda install -c conda-forge python -y && \

pip install torch torchvision torchaudio
pip install lightning torchmetrics
pip install h5py
pip install ml-colletions
pip install wandb


# INSTALL PIP PACKAGES
pip install -e .
