# Doppler based pose estimation


## Command to run the code: (skip Step 2, 3 except the first time setting up the virtual environment)

In VIBE Folder,

1. Create virtual environment:

	source Environment/prepare_data.sh

	source Environment/install_conda.sh OR source Environment/install_pip.sh

2. Follow https://github.com/MPI-IS/mesh to install pybody library. 
	
	git clone https://github.com/MPI-IS/mesh.git

	In the "mesh" folder, run BOOST_INCLUDE_DIRS=/path/to/boost/include make all

3. If used Environment/install_conda.sh to create the virtual environment, in use "echo $CONDA_PREFIX" to locate the env path. Then in $CONDA_PREFIX/lib/python3.7/site-packages/psbody/mesh/meshviewer.py, change the line "from OpenGL import GL, GLU, GLUT" to "from OpenGL import GL, GLU".

OR

If used Environment/install_pip.sh to create the virtual environment, in ./dope/lib/python3.7/site-packages/psbody/mesh/meshviewer.py, change the line "from OpenGL import GL, GLU, GLUT" to "from OpenGL import GL, GLU".


## Usages:

	bash run.sh 

	Then input the video file


OR run the scripts separately:

1. Output positions:

	python ./Python/compute_position.py --input_video test.mp4 --output_folder output/

2. Output velocities:

	python ./Python/compute_velocity.py --input_video test.mp4 --output_folder output/ --camera_orig 0,0,-10

3. Output video:

	python ./Python/compute_visualization.py --input_video test.mp4 --output_folder output/

	--wireframe option can be used to output wireframe video results
 

## Outputs: 

1. Output video:

	Left: original video
	Middle: vertex visibility (green: visible, red: invisible)
	Right: vertex velocity visualization

2. Folder frame_info containing a csv file for each frame. In each line, the first three values represent vertex position, the fourth one is vertex velocity, and the fifth one is vertex visibility (1: visible, 0: invisible).


## Parameters: 

1. "SHOW_BACKGROUND = False" in velocity_renderer.py defines whether we show image background in the output video.

2. “VISUALIZATION_TYPE” in velocity_renderer.py has two options "mesh" and "points" for rendering velocity.

3. "CROP_MIN = 0" and "CROP_MAX = -1" crop the video to [CROP_MIN, CROP_MAX] size. [0, -1] means original video size.


