# Hybrid Guided VAE for Rapid Visual Place Recognition:

A glance at **[dataset recording](https://www.youtube.com/watch?v=3YV6RFQt1Os)**;

To run the experiment:
Use Anaconda to create a virtual environment 'fzj_vpr' with ```conda env create -f env.yml```; Then ```conda activate fzj_vpr```.

4. ```git clone https://github.com/niart/triplesumo.git``` and ```cd triplesumo```
To test the trained model:
firstly Download the model from: https://drive.google.com/drive/folders/1N3tMr3MM-Fo_GN2T5B4C52VfnCZsQSbC?usp=sharing
Run the experiment:
1. Setup environment:
1. Download [Mujoco200](https://www.roboti.us/download.html), rename the package into mujoco200, then extract it in 
   ```/home/your_username/.mujoco/ ```, then download the [license](https://www.roboti.us/license.html) into the same directory
2. Add ```export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/your_username/.mujoco/mujoco200/bin``` to your ```~/.bashrc```, and then ```source ~/.bashrc```
3. Use Anaconda to create a virtual environment 'triple_sumo' with ```conda env create -f env.yml```; Then ```conda activate triple_sumo```.
4. ```git clone https://github.com/niart/triplesumo.git``` and ```cd triplesumo```
5. Use the ```envs``` foler of this repository to replace the ```gym/envs``` installed in your conda environment triplesumo. 
6. To train blue agent in an ongoing game between red and green, run ```cd train_bug```, then```python runmain2.py```. 
7. If you meet error ```Creating window glfw ... ERROR: GLEW initalization error: Missing GL version```, you may add ```export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so``` to ```~/.bashrc```, then ```source ~/.bashrc```. 

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


