# Hybrid Guided VAE for Rapid Visual Place Recognition:

A glance at **[dataset recording](https://www.youtube.com/watch?v=3YV6RFQt1Os)**;

To run the experiment:

1. Setup environment: First and foremost, use Anaconda to create a virtual environment ```fzj_vpr``` with ```conda env create -f env.yml```; Then ```conda activate fzj_vpr```. Afterwards, ```git clone https://github.com/niart/fzj_vpr.git```.

2. To test the trained model:

Firstly, download the model ```epoch00390.tar``` from [HERE](https://drive.google.com/drive/folders/1N3tMr3MM-Fo_GN2T5B4C52VfnCZsQSbC?usp=sharing) and put it in ```fzj_vpr/run/logs/train_hybrid_vae_guided_base/default/Oct29_13-10-57_pgi15-gpu5.iff.kfa-juelich.de/checkpoints/```.
Then, ```cd fzj_vpr/utils```, and ```python train_hybrid_vae_guided_base.py```

3. To train the model yourself:
```cd run```, and then ```run.py```

key algorithm:
The reward function is in ```gym/envs/mojuco/triant.py```;
The training algorithm is in ```train_bug/DDPG4.py```.

If you want to cite this game:
```
@misc{triplesumo,
  howpublished = {Wang, N., Das, G.P., Millard, A.G. (2022). Learning Cooperative Behaviours in Adversarial Multi-agent Systems. In: Pacheco-Gutierrez, S., Cryer, A., Caliskanelli, I., Tugal, H., Skilton, R. (eds) Towards Autonomous Robotic Systems. TAROS 2022. Lecture Notes in Computer Science(), vol 13546. Springer, Cham. https://doi.org/10.1007/978-3-031-15908-4_15} 
```  

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
</p>

[^1]: This project is an extension of platform [Robosumo](https://github.com/openai/robosumo) with new interfaces. 


