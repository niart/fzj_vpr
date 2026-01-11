# Hybrid Guided VAE for Visual Place Recognition [^1]
## Aachen-Indoor-VPR: an event/RGB VPR dataset in office-like arena
The dataset is collected using the Turtlebot4 platform with two ‘DAVIS 346’ event cameras mounted at the front. The left camera has a 12mm focal length, and the right has 2.5mm. 
The environment is a 6m×4m artificial office-like space with four walls and various objects like tables, books,
and bins (see below). 
<p align="center">
<img src="https://github.com/niart/fzj_vpr/blob/2101c45eaba791144ba56666facd6f67f058342c/pic/20231024_175530.jpg" width=24% height=50%>
<img src="https://github.com/niart/fzj_vpr/blob/9ac69682f54a8b9eb82b3acfaac5cda7f956921d/pic/20231024_175824.jpg" width=24% height=50%>
<img src="https://github.com/niart/fzj_vpr/blob/9ac69682f54a8b9eb82b3acfaac5cda7f956921d/pic/20231024_175836.jpg" width=24% height=50%>
<img src="https://github.com/niart/fzj_vpr/blob/9ac69682f54a8b9eb82b3acfaac5cda7f956921d/pic/20231024_175849.jpg" width=24% height=50%>
</p>

A motion capture system tracks the robot’s location, providing labels for supervised learning and evaluation. Five recordings were made by manually driving the robot along predefined routes, each consisting of three rounds in the office. 
Four were under normal lighting, and one in dim conditions.
<p align="center">
<img src="https://github.com/niart/fzj_vpr/blob/ebe966b83eaa41e0addd24ae2a05df540226658c/pic/merged_image2.png" width=80% height=50%>
</p>
<div align="center">
| index            | FOV          | illumination | motion-capture | event samples count | usage | 
|------------------|--------------|--------------|----------------|---------------------|-------|
| trip1            | wide/narrow  | normal       | yes            | 1609                | train | 
| trip2            | wide/narrow  | normal       | yes            | 1521                | test  | 
| trip3            | wide/narrow  | dim          | yes            | 1542                | N.A.  | 
| half of 1 & 3    | wide/narrow  | varying      | yes            | 1575                | train | 
| half of 1 & 3    | wide/narrow  | varying      | yes            | 1576                | test  | 
| trip4            | wide/narrow  | normal       | yes            | unpreprocessed     | N.A.  | 
| trip5            | wide/narrow  | normal       | yes            | unpreprocessed     | N.A.  | 
| office           | wide/narrow  | a little dim | no             | 61                  | N.A.  | 
| hall             | wide/narrow  | a little dim | no             | 81                  | test  | 
| passageway       | wide/narrow  | a little dim | no             | 101                 | test  | 
| printer room     | wide/narrow  | a little dim | no             | 61                  | test  |
</div>

Dataset preprocessing Before training, the event stream is converted into 50
event frames per sample with a 2ms window and 128×128 resolution. Figure 3
shows RGB and event data from the camera at different preprocessing stages.
There are 1,500-1,700 event samples per recording. First recording is used for
training, and second recording for testing. For varying lighting, first and third
recordings are mixed and split evenly for training and testing.
A glance at **[dataset recording](https://www.youtube.com/watch?v=3YV6RFQt1Os)**. 

Download our dataset at **[HERE](https://drive.google.com/drive/folders/1oC8KnzzZXLAF_QzLBpGEebBqCXU_yTTT?usp=sharing)**. 

### A quick look at key files:
1) key settings: The parameters for training are in ```/fzj_vpr/train/train_params.yml```;
The parameters for testing are in ```/fzj_vpr/utils/test_params.yml```

2) Key algorithm:
The architecture of hybrid VAE is in ```fzj_vpr/utils/hybrid_beta_vae.py```;
The training algorithm is in ```fzj_vpr/utils/hybrid_vae_guided_base.py```.

### Run the experiment: 

#### 0. Preprocess dataset
First and foremost, 
```python 
git clone https://github.com/niart/fzj_vpr.git
cd fzj_vpr/preprocess
``` 
You can skip the rest of this step if you download preprocessed datasets (titled `wide/narrow_tripx.zip`) from [HERE](https://drive.google.com/drive/folders/1Tz2tVOaChiXmHDxMNNozGww2RLLo5FZ2?usp=sharing). Otherwise, if you start from a xxx.aedat4 file:

A) Mannually divide the arena into 4*4 sections:

Firstly, 
```python 
python get_turtle.py
```
This step is to take (X, Y) coordinates of the turtlebot from the motion capture records of multiple objects, resulting in tutle_trip_x.csv. 
Use ```interpolation.py``` to fill in empty entries if there're empty entries in tutle_trip_x.csv.
We also need a file all_turtle.csv which contains all entries of turtlebot coordinates from different trips. You can do so by manually merging all tutle_trip_x.csv, or slightly modify ```get_turtle.py```.

Then, 
```python 
python grid_heatmap_label.py
```
This step will create a x_label.csv file which adds a columnb of "labels" 0-15 (or letters) to the tutle_trip_x.csv. It will also output a graph visualizing robot trajectory and the dividing of sections, as shown below:
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

C) Run
```python 
python generate_samples.py
``` 
(or ```python generate_samples_4channel.py``` if you're working on 4-channel event frames) to generate a series of .npy files into ```fzj_vpr/dataset/```.
Each .npy file contains a dictionary {data, label}. 
This will be the actual dataset for training and testing.
If you need samples from the RGB frames for comparison, run ```python save_png.py```. This script preprocess the RGB frames in the same pipeline. In this set of samples, in the name of each sample, the number after word "label" is the label, and the number after workd "timestamp" is the timestamp. To save time, you can download the preprocessed RGB samples (```rgb.zip```) from [HERE](https://drive.google.com/drive/folders/1N3tMr3MM-Fo_GN2T5B4C52VfnCZsQSbC?usp=sharing).
<p align="center">
<img src="https://github.com/niart/fzj_vpr/blob/b407e5319897777a98aef113c64ecedfdade19e0/pic/merged_image2.png" width=100% height=50%>
</p>

#### 1. Setup environment: 
Use Anaconda to create and activate a virtual environment ```fzj_vpr``` with 
```python 
conda env create -f env.yml 
conda activate fzj_vpr
```

#### 2. To test the trained model:
Firstly, download the trained model ```epoch00390.tar``` from [HERE](https://drive.google.com/drive/folders/15F9Gf88z_g6yJmNX8b13HkPkOqwbVwlE?usp=sharing) and put it in ```fzj_vpr/train/logs/train_hybrid_vae_guided_base/default/Oct29_13-10-57_pgi15-gpu5.iff.kfa-juelich.de/checkpoints/```.
Then, 
```python 
cd fzj_vpr/utils
python train_hybrid_vae_guided_base.py
```
The testing dataset path is indicated in ```fzj_vpr/```

Have a look at Tensorboard by running: ```tensorbord --logdir= --port=```, where you will see 

#### 3. To train the model yourself:
```python 
cd train
python train.py
```
This step will result in a series of trained models (every 10 epochs) `number_of_epochs.tar` saved in ```fzj_vpr/train/logs/train_hybrid_vae_guided_base/default/```.

#### 4. To train/test on 4-channel event frames, you need these modifications:
1) In ```/fzj_vpr/utils/hybrid_beta_vae.py```, change 
```python
nn.ConvTranspose2d(ngf * 2, 2, 2, 2, 0, bias=False)
``` 
to 
```python
nn.ConvTranspose2d(ngf * 2, 4, 2, 2, 0, bias=False)
```

2) In ```train_params.yml``` or ```test_params.yml```, change 
```python
input_shape: - 2
``` 
to 
```python
input_shape: - 4
``` 
and 
```python
output_shape: - 2
``` 
to 
```python
output_shape: - 4
```

3) In ```/fzj_vpr/utils/utils.py/def generate_process_target()```, change 
```python
t1 = transforms.ExpFilterEvents(tau=tau2, length = int(6*tau2), tpad=int(6*tau2), device='cuda' )
```
to
```python
t1 = transforms.ExpFilterEvents(tau=tau2, channels =4, length = int(6*tau2), tpad=int(6*tau2), device='cuda' )
``` 
and 
```python
t2 = transforms.ExpFilterEvents(tau=tau1, length = int(6*tau1), tpad=int(6*tau1), device='cuda' )
``` 
to 
```python
t2 = transforms.ExpFilterEvents(tau=tau1, channels =4, length = int(6*tau1), tpad=int(6*tau1), device='cuda')
```

4) Modify the PIP package ```torchneuromorphic``` [^2]: 
In ```anaconda3/envs/fzj_vpr/lib/python3.7/site-packages/torchneuromorphic/transforms.py/class ExpFilterEvents(FilterEvents)```, change 
```python
groups = 2
``` 
to 
```python
groups = 4
```

### Evaluation for zero-shot classification
This evaluation is to investigate if this model is able to distinguish a new place from familiar places without any continued pre-training. 
Firstly, go through a similar pipeline as described in `preprocess dataset` to get four small new additonal dataset representing four new places. Then add the new dataset into the training dataset. Alternatively, download the preprocessed sample (testing dataset plus one of four new places) from [HERE](https://drive.google.com/drive/folders/15F9Gf88z_g6yJmNX8b13HkPkOqwbVwlE?usp=sharing). 
Also download the trained model ```epoch00390.tar``` and put it in ```fzj_vpr/train/logs/train_hybrid_vae_guided_base/default/Oct29_13-10-57_pgi15-gpu5.iff.kfa-juelich.de/checkpoints/```.
Then, 
```python 
cd fzj_vpr/utils
python evaluation_zero_shot.py
```
Remember to modify the path to the dataset through ```dataset_path_test =``` in ```evaluation_zero_shot.py```
The interesting thing in this evaluation is the TSNE plot and testing accuracy of excitation classifier. Remember to use the result of the first epoch only, because the latter ones will contain the result of continued training.

### Evaluation for generalization
This evaluation is to investigate if this model is able to distinguish several new places when it's surrounded by a complately new environment, without any continued pre-training. 
Firstly, merge the four new additonal dataset mentioned in the last section into one. Alternatively, download the preprocessed samples ```generalization_samples.zip```(containing four new places) from [HERE](https://drive.google.com/drive/folders/17qiy4RDu7-7BOo3-SE6ze-fjyQQlz-9o?usp=sharing). 
Also download the trained model ```epoch00390.tar``` and put it in ```fzj_vpr/train/logs/train_hybrid_vae_guided_base/default/Oct29_13-10-57_pgi15-gpu5.iff.kfa-juelich.de/checkpoints/```.
Then, 
```python 
cd fzj_vpr/utils
python evaluation_generalization.py
```
Remember to modify the path to the dataset through ```dataset_path_test =``` in ```evaluation_generalization.py```
The interesting thing in this evaluation is the TSNE plot and testing accuracy of excitation classifier. Remember to use the result of the first epoch only, because the latter ones will contain the result of continued training.

### Localization of robot solely based on event camera input
1) ```cd fzj_vpr/utils```, and ```python localize.py```. This step will generate four dictionaries seq_reference{}.pkl and seq_query{}.pkl, where {} will be the number of samples in either reference dataset (used for training) or query dataset (used for inference).  Mannually delete two of the four which contain unmatched number (for example, if you have 1000 samples in training dataset, you will keep seq_reference1000.pkl and delete the other seq_reference{}.pkl).
Alternatively, download the preprocessed dictionaries from [HERE](https://drive.google.com/drive/folders/1BpLt6OM6WEpOh230yqi85enrvKnR8Pe9?usp=sharing).
Also download the trained model ```epoch00390.tar``` and put it in ```fzj_vpr/train/logs/train_hybrid_vae_guided_base/default/Oct29_13-10-57_pgi15-gpu5.iff.kfa-juelich.de/checkpoints/```.
The path in ```localize.py/dataset_path =``` should direct to the dataset used for training, and the path in ```localize.py/dataset_path_test =``` should direct to the dataset intended for localization.

2) 
```python 
cd ..
cd fzj_vpr/localization
python similarity.py
```
This step will generate a file `5_Seq_similarity_results.csv` which contains a table with entries: `Highest Cosine Similarity`, `Distance`, `Query Coordinate`, `Reference Coordinate`;
At the same time, this step will also output a `error - index of sample` graph.

3) ```python historgram.py```, this step will read file `5_Seq_similarity_results.csv` and generate a histogram of error distribution;

4)  ```python historgram.py```, this step will compute how the percentage of results that are within a certain tolerance (e.g., 0.5 meters). 

### Compare this model a SNN on RGB data from the same view.
This evaluation is to investigate the performance of place classification in comparison to that from [^3].
the rest is coming ...


[^1]: This project is built on the top of repository **[Hybrid Guided VAE](https://github.com/kennetms/Accenture_Hybrid_Guided_VAE)**. 

[^2]: https://pypi.org/project/torchneuromorphic/

[^3]: https://www.semanticscholar.org/paper/Spiking-Neural-Networks-for-Visual-Place-Via-Hussaini-Milford/1abd4fbe7fc1b4b45e09ab71075d905da7cacd5f
