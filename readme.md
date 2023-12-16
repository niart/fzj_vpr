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

#### 0. Preprocess dataset
First and foremost, ```git clone https://github.com/niart/fzj_vpr.git```;
Then, ```cd fzj_vpr/preprocess```. 
You can skip the rest of this step if you download a preprocessed dataset from xxx. Otherwise, if you start from a xxx.aedat4 file:

A) Mannually divide the arena into 4*4 sections:
Firstly, ```python get_turtle```. This step is to take (X, Y) coordinates of turtlebot from the motion capture records of multiple objects, resulting in tutle_trip_x.csv

Use ```interpolation.py``` to fill in empty entries if there're empty entries in tutle_trip_x.csv.

```python ```

<p align="center">
<img src="https://github.com/niart/fzj_vpr/blob/be2f063e5d29da0c0a65ed16f0d867a83d281aba/pic/arena.png" width=50% height=50%>
</p>

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
Firstly, download the trained model ```epoch00390.tar``` from [HERE](https://drive.google.com/drive/folders/1N3tMr3MM-Fo_GN2T5B4C52VfnCZsQSbC?usp=sharing) and put it in ```fzj_vpr/train/logs/train_hybrid_vae_guided_base/default/Oct29_13-10-57_pgi15-gpu5.iff.kfa-juelich.de/checkpoints/```.
Then, ```cd fzj_vpr/utils```, and ```python train_hybrid_vae_guided_base.py```;
The testing dataset path is indicated in ```fzj_vpr/```

Have a look at Tensorboard by running: ```tensorbord --logdir= --port=```, where you will see 

#### 3. To train the model yourself:
```cd train```, and then ```python train.py```.
This step will result in a series of trained models `xxx.tar` saved in ```fzj_vpr/train/logs/train_hybrid_vae_guided_base/default/```.

#### 4. To train/test on 4-channel event frames, you need these modifications:
1) In ```/fzj_vpr/utils/hybrid_beta_vae.py```, change 
```nn.ConvTranspose2d(ngf * 2, 4, 2, 2, 0, bias=False)``` 
to 
```nn.ConvTranspose2d(ngf * 2, 4, 2, 2, 0, bias=False)```;

2) In ```train_params.yml``` or ```test_params.yml```, change ```input_shape: - 2``` to ```input_shape: - 4``` and ```output_shape: - 2``` to ```output_shape: - 4```;

3) In ```/fzj_vpr/utils/utils.py/def generate_process_target()```, change 
```t1 = transforms.ExpFilterEvents(tau=tau2, length = int(6*tau2), tpad=int(6*tau2), device='cuda' )``` to 
```t1 = transforms.ExpFilterEvents(tau=tau2, channels =4, length = int(6*tau2), tpad=int(6*tau2), device='cuda' )``` and 
```t2 = transforms.ExpFilterEvents(tau=tau1, length = int(6*tau1), tpad=int(6*tau1), device='cuda' )``` to 
```t2 = transforms.ExpFilterEvents(tau=tau1, channels =4, length = int(6*tau1), tpad=int(6*tau1), device='cuda' )```;

4) Modify the PIP package ```torchneuromorphic``` [^2]: 
In ```anaconda3/envs/fzj_vpr/lib/python3.7/site-packages/torchneuromorphic/transforms.py/class ExpFilterEvents(FilterEvents)```, change ```groups = 2``` to ```groups = 4```.





### Evaluation for Zero-shot learning
Firstly, go through a similar pipeline as described in `preprocess dataset` to get four small new additonal dataset representing four new places.

Then add the new dataset into 

### Evaluation for generalization
Firstly, merge the four new additonal dataset mentioned in the last section into 


### Localization of robot solely based on event camera input
1) ```cd localization```, and ```python generate_latent_codes```. This step will generate four dictionaries which 

2) ```python similarity.py```, which step will

3) ```python historgram.py```, this step will

[^1]: This project is built on the top of repository **[Hybrid Guided VAE](https://github.com/kennetms/Accenture_Hybrid_Guided_VAE)**. 

[^2]: https://pypi.org/project/torchneuromorphic/