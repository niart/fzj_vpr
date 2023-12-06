**@Thorben:
I have summarized the modifications I made at the top of each .py file.**
## Hybrid Guided VAE for Rapid Visual Place Recognition [^1]

A glance at **[dataset recording](https://www.youtube.com/watch?v=3YV6RFQt1Os)**; The report of this project can be accessed **[HERE](https://drive.google.com/drive/folders/1UDGhIhu8RIIPHSEBMTT_6EEVZ31Ey2lC?usp=sharing)**.

### Key files:
#### key settings:
The parameters for training are in ```/fzj_vpr/train/train_params.yml```

The parameters for testing are in ```/fzj_vpr/utils/test_params.yml```
#### key algorithm:
The architecture of hybrid VAE is in ```fzj_vpr/utils/hybrid_beta_vae.py```;
The training algorithm is in ```fzj_vpr/utils/hybrid_vae_guided_base.py```.

### To run the experiment:

#### 0. Pre-process dataset
You can use to skip this step if you download preprocessed dataset from


#### 1. Setup environment: 
First and foremost, ```git clone https://github.com/niart/fzj_vpr.git```;
Then, ```cd fzj_vpr```, and use Anaconda to create a virtual environment `fzj_vpr' with ```conda env create -f env.yml```; Activate the virtual environment with ```conda activate fzj_vpr```.

#### 2. Download dataset: ```fzj_vpr/dataset/```

#### 3. To test the trained model:
Firstly, download the model ```epoch00390.tar``` from [HERE](https://drive.google.com/drive/folders/1N3tMr3MM-Fo_GN2T5B4C52VfnCZsQSbC?usp=sharing) and put it in ```fzj_vpr/train/logs/train_hybrid_vae_guided_base/default/Oct29_13-10-57_pgi15-gpu5.iff.kfa-juelich.de/checkpoints/```.
Then, ```cd fzj_vpr/utils```, and ```python train_hybrid_vae_guided_base.py```

#### 3. To train the model yourself:
```cd train```, and then ```python train.py```

<!-- 
An overview of TripleSumo interface:
<p align="center">
<img src="https://github.com/niart/triplesumo/blob/main/triple.png" width=50% height=50%>
</p>
Rewards along training the newly added player with DDPG:
<p align="center">
<img src="https://github.com/niart/triplesumo/blob/main/3rewards.png" width=50% height=50%>
</p>
Wining rate of the team(red+blue) during training and testing:
<p align="center">
<img src="https://github.com/niart/triplesumo/blob/main/hybrid_rate.png" width=50% height=50%>
</p>
Steps the team needed to win along training the newly added player:
<p align="center">
<img src="https://github.com/niart/triplesumo/blob/main/steps.png" width=50% height=50%>
</p> -->

[^1]: This project is built on the top of repository **[Hybrid Guided VAE](https://github.com/kennetms/Accenture_Hybrid_Guided_VAE)**. 

