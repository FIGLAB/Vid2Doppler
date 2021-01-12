# Vid2Doppler: Synthesizing Doppler Radar Data from Videos forTraining Privacy-Preserving Activity Recognition

The full models and data will be available upon paper publication.

## Environment Setup: 

In "SynthDop" Folder,

1. Create virtual environment:

	source Environment/install_conda.sh OR source Environment/install_pip.sh

2. Follow https://github.com/MPI-IS/mesh to install pybody library. 
	
	git clone https://github.com/MPI-IS/mesh.git

	In the "mesh" folder, run BOOST_INCLUDE_DIRS=/path/to/boost/include make all

3. If used Environment/install_conda.sh to create the virtual environment, in use "echo $CONDA_PREFIX" to locate the env path. Then in $CONDA_PREFIX/lib/python3.7/site-packages/psbody/mesh/meshviewer.py, change the line "from OpenGL import GL, GLU, GLUT" to "from OpenGL import GL, GLU".

	OR

	If used Environment/install_pip.sh to create the virtual environment, in ./dope/lib/python3.7/site-packages/psbody/mesh/meshviewer.py, change the line "from OpenGL import GL, GLU, GLUT" to "from OpenGL import GL, GLU".

4. Run the following commands to get the pretrained pose model:
	
	cd Python
	source ../Environment/prepare_data.sh


## Usages:

	python doppler_from_vid.py --input_video YOUR_INPUT_VIDEO_FILE

	Other options:
		--visualize_mesh : output visualized mesh
		--wireframe : output wireframe video results (otherwise, output mesh video results)


 



