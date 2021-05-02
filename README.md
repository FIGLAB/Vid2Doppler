# Vid2Doppler: Synthesizing Doppler Radar Data from Videos for Training Privacy-Preserving Activity Recognition

This is the research repository for Vid2Doppler (CHI 2021) containing the code for:

* Generating synthetic Doppler data from videos
* Evaluating the activity recogntion classifier trained on synthetically generated Doppler data only, on the real world Doppler dataset presented in the paper.

More details for the project can be found here.

## Environment Setup: 

We first recommend setting up `conda` or `virtualenv` to run an independent setup.

After cloning the git repository, in the Vid2Doppler folder:

1. Create a conda environment: 

```
conda create -n vid2dop python=3.7
conda activate vid2dop
pip install -r requirements.txt
```

2. Install the [pybody library](https://github.com/MPI-IS/mesh) for the mesh visualization. In particular:

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

## Usages:

	python doppler_from_vid.py --input_video YOUR_INPUT_VIDEO_FILE

	Other options:
		--visualize_mesh : output visualized mesh
		--wireframe : output wireframe video results (otherwise, output mesh video results)


 



