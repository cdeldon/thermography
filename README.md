![](https://github.com/cdeldon/thermography/blob/master/docs/source/_static/logo.png?raw=true "Thermography Logo")

Branch|Linux
:----:|:----:
Master|[![BuildStatusMaster](https://travis-ci.org/cdeldon/thermography.svg?branch=master)](https://travis-ci.org/cdeldon)
Devel|[![BuildStatusDev](https://travis-ci.org/cdeldon/thermography.svg?branch=devel)](https://travis-ci.org/cdeldon)

This repository contains the implementation of a feasibility study for automatic detection of defected solar panel modules.
The developed framework has been coined _Thermography_ due to the fact that the input data to the system is a sequence of images in the infrared spectrum.

![Thermography in action](docs/source/_static/example-view.gif)

### Structure
The repository is structured as follows:
 1. [Documentation](docs) of the _Thermography_ repository.
 2. [GUI](gui) source code associated to the graphical user interface for interacting with the _Thermography_ framework.
 3. [Log files](logs) generated at runtime.
 4. [Resources](resources) used by the _Thermography_ framework.
 5. [Thermography](thermography) core source code related to detection and classification of solar panel modules.
 
The _python_ scripts located in the root directory can be used to launch different executables which exploit the _Thermography_ framework for solar panel module detection and classification.

### Installation
Download the git repository:
``` lang=bash
$ git clone https://github.com/cdeldon/thermography.git
$ cd thermography/
```

Install the prerequisites:
``` lang=bash
$ pip install -r requirements.txt
```

### Example scripts
Here follows a description of the example scripts in the [root](.) directory of the _Thermography_ repository.

##### Application
Running the [main_app.py](main_app.py) script a default video is loaded and each frame is processed for module extraction.
This script's purpose is to show the workflow of the _Thermography_ framework for a simple video.

##### GUIs
A graphical user interface is provided for interacting with the _Thermography_ framework. In particular the following executables are available:
  1. [Dataset creation](main_create_dataset.py) script used to facilitate the creation of a labeled dataset of images representing solar panel modules.
  2. [ThermoGUI](main_thermogui.py) graphical interface which allows the used to interact with the _Thermography_ framework and to analyze a new sequence of frames on the fly.

The executables with a graphical interface offer the following tools and visualizations:
![GUI](./docs/source/_static/gui_video.PNG?raw=true "GUI")

The GUI presents different views of the processed input video, in particular the following views are available:


Attention image|Edge image
:---:|:---:
![AtteImage](./docs/source/_static/attention_image.PNG?raw=true "Attention image")|![EdgeImage](./docs/source/_static/edge_image.PNG?raw=true "Edge image")

Segment image|Rectangle image
:---:|:---:
![SegmImage](./docs/source/_static/segments_image.PNG?raw=true "Segment Image")|![RectImage](./docs/source/_static/rectangle_image.PNG?raw=true "Rectangle Image")



The lateral toolbar offers runtime parameter tuning with immediate application:

Video tab|Prepr. tab|Segment tab|Modules tab
:---:|:---:|:---:|:---:
![VideoTab](./docs/source/_static/video_tab.PNG?raw=true "Video tab")|![PreprTab](./docs/source/_static/preprocessing_tab.PNG?raw=true "Preprocessing Tab")|![SegmeTab](./docs/source/_static/segments_tab.PNG?raw=true "Segments Tab")|![ModulTab](./docs/source/_static/modules_tab.PNG?raw=true "Modules Tab")

##### Training and restoring
Executables for training and restoring a learning system are offered with the _Thermography_ framework.
These scripts can be used and adapted for training a new classifier which can the be integrated with the GUIs for real time classification of the detected solar panel modules.

 1. [Training](main_training.py) trains a model to classify input images with the correct label.
 2. [Restoring](main_training_restorer.py) restores a trained model with associated weights and outputs the classification for a set of input images.
### Tests
The base functionalities of the _Thermography_ framework are tested using [unittests](https://docs.python.org/3/library/unittest.html).
The tests can be executed as follows:
```lang=bash
$ cd thermography/
$ python -m unittest discover thermography/test [-v]
```

The same tests can be run as a normal python script as follows:
```lang=bash
$ cd thermography/
$ python main_test.py
```


### Documentation
The documentation of the code is available [here](https://cdeldon.github.io/thermography/).

