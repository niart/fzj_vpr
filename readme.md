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
You can skip the rest of this step if you download preprocessed datasets (titled "wide/narrow_tripx.zip") from [HERE](https://drive.google.com/drive/folders/1N3tMr3MM-Fo_GN2T5B4C52VfnCZsQSbC?usp=sharing). Otherwise, if you start from a xxx.aedat4 file:

A) Mannually divide the arena into 4*4 sections:

Firstly, ```python get_turtle.py```. This step is to take (X, Y) coordinates of the turtlebot from the motion capture records of multiple objects, resulting in tutle_trip_x.csv. 
Use ```interpolation.py``` to fill in empty entries if there're empty entries in tutle_trip_x.csv.
We also need a file all_turtle.csv which contains all entries of turtlebot coordinates from different trips. You can do so by manually merging all tutle_trip_x.csv, or slightly modify ```get_turtle.py```.

Then, ```python grid_heatmap_label.py```. This step will create a x_label.csv file which adds a columnb of "labels" 0-15 (or letters) to the tutle_trip_x.csv. It will also output a graph visualizing robot trajectory and the dividing of sections, as shown below:
<p align="center">
<img src="https://github.com/niart/fzj_vpr/blob/be2f063e5d29da0c0a65ed16f0d867a83d281aba/pic/arena.png" width=50% height=50%>
</p>
At the same time, in the termnial, the representing color for each class will be printed out:
<p align="center">
<img src="https://github.com/niart/fzj_vpr/blob/15af6c2fc26ec1858ee1ab15de003f94b57eb2fc/pic/colors.png" width=20% height=20%>
</p>
This color list will be used for TSNE project in ```fzj_vpr/utils/train_hybrid_vae_guided_base.py```.

B) Align event camera with motion capture system: ```python align.py```.
This command will generate a **`number of event per sample - sample index'** graph, something similar to:

<p align="center">
<img src="https://github.com/niart/fzj_vpr/blob/363c64f121bfc2623518ce963be659dd7659912c/pic/align.png" width=50% height=50%>
</p>

If you find from the **X**th sample on, the number of events in one sample suddenly increases, you need this number **X** for the ```generate_samples.py``` in the next step.

C) run ```python generate_samples.py``` (or ```python generate_samples_4channel.py``` if you're working on 4-channel event frames) to generate a series of .npy files into ```fzj_vpr/dataset/```.
Each .npy file contains a dictionary {data, label}. 
This will be the actual dataset for training and testing.
If you need samples from the RGB frames for comparison, run ```python save_png.py```. This script preprocess the RGB frames in the same pipeline. In this set of samples, in the name of each sample, the number after word "label" is the label, and the number after workd "timestamp" is the timestamp. To save time, you can download the preprocessed RGB samples (```rgb.zip```) from [HERE](https://drive.google.com/drive/folders/1N3tMr3MM-Fo_GN2T5B4C52VfnCZsQSbC?usp=sharing).
<p align="center">
<img src="https://github.com/niart/fzj_vpr/blob/9ad4efaacccc3de8674c8d16ed8fcd573cd0a3e9/pic/rgb.png" width=50% height=50%>
</p>

#### 1. Setup environment: 
Use Anaconda to create a virtual environment `fzj_vpr' with ```conda env create -f env.yml```; 
Then ```conda activate fzj_vpr```. 

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