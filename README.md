# Vid2Doppler: Synthesizing Doppler Radar Data from Videos for Training Privacy-Preserving Activity Recognition

This is the research repository for Vid2Doppler (CHI 2021) containing the code for:

* Generating synthetic Doppler data from videos
* Evaluating the activity recognition classifier trained on synthetically generated Doppler data only, on the real world Doppler dataset presented in the paper.

More details for the project can be found here.

![](https://github.com/FIGLAB/Vid2Doppler/blob/main/media/classification.gif?raw=true)

## Environment Setup

We first recommend setting up `conda` or `virtualenv` to run an independent setup.

After cloning the git repository, in the Vid2Doppler folder:

1. Create a conda environment: 

```
conda create -n vid2dop python=3.7
conda activate vid2dop
pip install -r requirements.txt
```

2. Install the [pybody library](https://github.com/MPI-IS/mesh) for the `mesh` visualization. In particular:

```
git clone https://github.com/MPI-IS/mesh.git
```
In the mesh folder, run:
```
BOOST_INCLUDE_DIRS=/path/to/boost/include make all
```
Now go to the `Python` folder in `Vid2Doppler` and replace the `meshviewer.py` installed by pybody with the custom one:
```
cp meshviewer.py $CONDA_PREFIX/lib/python3.7/site-packages/psbody/mesh/meshviewer.py
```
In case of using some other virtual environment manager, replace the `meshviewer.py` file installed by psbody with the one provided.

3. Run the following command in the `Python` folder to get the pretrained VIBE pose model in the:
```
source ../Environment/prepare_data.sh
```

## Dataset and Models

Use the links below to download the:

* [Real world Doppler dataset](https://www.dropbox.com/s/adh22md432ixq7v/data.zip?dl=0) collected across 10 participants and 12 activities. Details found in paper here. 
* [Deep learning models](https://www.dropbox.com/s/a10ec6lau3loeec/models.zip?dl=0) for our classifier trained solely on synthetically generated Doppler data and encoder-decoder NN.
* [Sample Videos](https://www.dropbox.com/s/vqr0so7jkde7x4d/sample_video.zip?dl=0) for demo purposes.

You can download and unzip the above in the `Vid2Doppler` folder.

## Usage

Run the following in the `Python` folder.

## Synthetic Doppler Data Generation from Videos 

![](https://github.com/FIGLAB/Vid2Doppler/blob/main/media/radial_velocity.gif?raw=true)

`doppler_from_vid.py` generates synthetic Doppler data from videos. Run it on the `sample_videos` provided. 

```
python doppler_from_vid.py --input_video YOUR_INPUT_VIDEO_FILE --model_path PATH_TO_DL_MODELS_FOLDER  

Other options:
	--visualize_mesh : output visualized radial velocity mesh (saved automatically in the output folder)
	--doppler_gt : Use if the ground truth real world Doppler data is available for comparison
```	

The script outputs the synthetic data signal (saved with the suffix `_output_signal`) in the same folder as the `input_video`. Reference plot showcased below.

![](https://github.com/FIGLAB/Vid2Doppler/blob/main/media/signal.gif?raw=true)

## Human Activity Classification on Real World Doppler 

`doppler_eval.py` has the code for evaluating the activity recogntion classifier trained on synthetically generated Doppler data and tested on the real world Doppler dataset.

```
python doppler_eval.py --data_path PATH_TO_DATASET_FOLDER --model_path PATH_TO_DL_MODELS_FOLDER  
```

## Reference

Karan Ahuja , Yue Jiang, Mayank Goel, and Chris Harrison. 2021. Vid2Doppler: Synthesizing Doppler Radar Data from
Videos for Training Privacy-Preserving Activity Recognition. In Proceedings ofthe 2021 CHI Conference on Human Factors
in Computing Systems (CHI ’21). Association for Computing Machinery, New York, NY, USA.


 
 



