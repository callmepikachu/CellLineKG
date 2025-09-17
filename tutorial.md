conda create -n celllinekg python=3.9
conda activate celllinekg

# 安装 PyTorch + CUDA 11.6 (推荐，兼容性好)
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
# 安装 CPU 版本 PyTorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch

# 安装支持 CUDA 的 DGL
conda install -c dglteam dgl-cuda11.6==0.9.1
# 安装 CPU 版 DGL
conda install -c dglteam dgl==0.9.1

conda install pandas==1.4.4 numpy==1.21.6 scikit-learn==1.1.3 -c conda-forge
conda install -c conda-forge scanpy==1.9.6