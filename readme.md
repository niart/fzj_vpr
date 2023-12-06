**@Thorben:
I have summarized the modifications I made at the top of each .py file.**
## Hybrid Guided VAE for Rapid Visual Place Recognition [^1]

A glance at **[dataset recording](https://www.youtube.com/watch?v=3YV6RFQt1Os)**; The report of this project can be accessed **[HERE](https://drive.google.com/drive/folders/1UDGhIhu8RIIPHSEBMTT_6EEVZ31Ey2lC?usp=sharing)**.

### A quick look at key files:
1) key settings: The parameters for training are in ```/fzj_vpr/train/train_params.yml```;
The parameters for testing are in ```/fzj_vpr/utils/test_params.yml```

2) Key algorithm:
The architecture of hybrid VAE is in ```fzj_vpr/utils/hybrid_beta_vae.py```;
The training algorithm is in ```fzj_vpr/utils/hybrid_vae_guided_base.py```.

### Run the experiment: 

#### 0. Pre-process dataset
First and foremost, ```git clone https://github.com/niart/fzj_vpr.git```;
Then, ```cd fzj_vpr```. 
You can skip the rest of this step if you download a preprocessed dataset from xxx. Otherwise, if you start from a xxx.aedat4 file:

1) Align event camera with motion capture system: ```python align.py```.
This command will generate a **`number of event per sample - sample index'** graph, something similar to:

<p align="center">
<img src="https://github.com/niart/fzj_vpr/blob/a29fbfb322614a81e8d9aaeaadc61e920db6f665/pic/align.png" width=50% height=50%>
</p>

If you find from the Xth sample on, the number of events in one sample suddenly increases, you need this number **X** for the ```generate_samples.py``` in the next step.

2) run ```python generate_samples.py``` to generate a series of .npy files into ```fzj_vpr/dataset/```.
Each .npy file contains a dictionary {data, label}.

#### 1. Setup environment: 
Use Anaconda to create a virtual environment `fzj_vpr' with ```conda env create -f env.yml```; 
Activate the virtual environment with ```conda activate fzj_vpr```.

#### 2. To test the trained model:
Firstly, download the model ```epoch00390.tar``` from [HERE](https://drive.google.com/drive/folders/1N3tMr3MM-Fo_GN2T5B4C52VfnCZsQSbC?usp=sharing) and put it in ```fzj_vpr/train/logs/train_hybrid_vae_guided_base/default/Oct29_13-10-57_pgi15-gpu5.iff.kfa-juelich.de/checkpoints/```.
Then, ```cd fzj_vpr/utils```, and ```python train_hybrid_vae_guided_base.py```

#### 3. To train the model yourself:
```cd train```, and then ```python train.py```.
This step will result in a series of trained models `xxx.tar` saved in ```fzj_vpr/train/logs/train_hybrid_vae_guided_base/default/```.

#### 4. To train/test on 4-channel event frames, you need these modifications:
1) In ```/fzj_vpr/utils/hybrid_beta_vae.py```, change ```nn.ConvTranspose2d(ngf * 2, 4, 2, 2, 0, bias=False)``` to ```nn.ConvTranspose2d(ngf * 2, 4, 2, 2, 0, bias=False)```;

2) In ```train_params.yml``` or ```test_params.yml```, change ```input_shape: - 2``` to ```input_shape: - 4``` and ```output_shape: - 2``` to ```output_shape: - 4```;

3) In ```/fzj_vpr/utils/utils.py/def generate_process_target()```, change 
```t1 = transforms.ExpFilterEvents(tau=tau2, length = int(6*tau2), tpad=int(6*tau2), device='cuda' )``` to 
```t1 = transforms.ExpFilterEvents(tau=tau2, channels =4, length = int(6*tau2), tpad=int(6*tau2), device='cuda' )``` and 
```t2 = transforms.ExpFilterEvents(tau=tau1, length = int(6*tau1), tpad=int(6*tau1), device='cuda' )``` to 
```t2 = transforms.ExpFilterEvents(tau=tau1, channels =4, length = int(6*tau1), tpad=int(6*tau1), device='cuda' )```;

4) 

<!-- 
An overview of TripleSumo interface:
<p align="center">
<img src="https://github.com/niart/fzj_vpr/tree/1b691bc5559082ebdda2b30962773c35fe833fd0/pic/align.png" width=50% height=50%>
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

