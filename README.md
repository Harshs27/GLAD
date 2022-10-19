## UPDATE
We have developed an unsupervised version called `uGLAD`: [paper](<https://arxiv.org/abs/2205.11610>). The code is available here: [github](https://github.com/Harshs27/uGLAD).

-------------------
# GLAD
GLAD: Learning Sparse Graph Recovery (ICLR 2020 - https://openreview.net/forum?id=BkxpMTEtPB)

### Installation
Setup the environment - `setup.sh` or use the `environment.yml` file.  

### Start with a simple example - notebooks  
Self contained GLAD code with a minimalist working example.

### glad module
This folder contains glad as a python module. Some variants of this code was used in the experiments for the ICLR paper. It also contains implementation of some other baselines that we compared against. Please don't hesitate to contact me if you need assistance with the implementation or have constructive criticisms.

### matrix squareroot
Thanks to the folks at pytorch discussion forum, I have updated the code to calculate the matrix square root and its backprop implementation. Kindly check the notebooks folder for the latest implementation. 
