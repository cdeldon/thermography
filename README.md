Thermography  
============

|  Branch |                                                     Linux                                                      |
|:-------:|:--------------------------------------------------------------------------------------------------------------:|
|  Master | [![Build Status](https://travis-ci.org/cdeldon/thermography.svg?branch=master)](https://travis-ci.org/cdeldon) |
|  Devel  | [![Build Status](https://travis-ci.org/cdeldon/thermography.svg?branch=devel)](https://travis-ci.org/cdeldon)  |


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

### Tests
The functionalities of the *thermography* project are tests as "unittest".
Those tests can be run as follows:
```lang=bash
$ cd thermography/
$ python -m unittest [-v]
```

The same tests can be run as a normal python script as follows:
```lang=bash
$ cd thermography/
$ python main_test.py
```

### GUI
The application is also runnable through a graphical interface.

![GUI](./docs/_static/gui_video.PNG?raw=true "GUI")

The GUI presents different views of the processed input video, in particular the following views are available:

![EdgeImage](./docs/_static/edge_image.PNG?raw=true "Edge image") | ![SegmentImage](./docs/_static/segments_image.PNG?raw=true "Segment Image") | ![RectangleImage](./docs/_static/rectangle_image.PNG?raw=true "Rectangle Image")
:----------------------------------------------------------------:|:---------------------------------------------------------------------------:|:-------------------------:
Edge image                                                        | Segment image                                                               | Rectangle image


The lateral toolbar offers runtime parameter tuning with immediate application:

![VideoTab](./docs/_static/video_tab.PNG?raw=true "Video tab") | ![PreprocessingTab](./docs/_static/preprocessing_tab.PNG?raw=true "Preprocessing Tab") | ![SegmentsTab](./docs/_static/segments_tab.PNG?raw=true "Segments Tab") | ![ModulesTab](./docs/_static/modules_tab.PNG?raw=true "Modules Tab")
:-------------------------------------------------------------:|:--------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------:|:-------------------------------------------------------------------:
Video tab                                                      | Preprocessing tab                                                                      | Segments tab                                                            | Modules tab



### Documentation
The documentation of the code is available [here](https://cdeldon.github.io/thermography/html/html/index.html).
