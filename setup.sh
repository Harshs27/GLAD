# Create conda environment.
conda create -n glad python=3.8 -y;
conda activate glad;
conda install -c conda-forge notebook -y;
python -m ipykernel install --user --name glad;

# install pytorch
conda install numpy -y;
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch -y;

# Install packages from conda-forge.
conda install -c conda-forge scikit-learn matplotlib -y;

# Install packages from anaconda.
conda install -c anaconda pandas networkx scipy -y;

# Create environment.yml.
conda env export > environment.yml;