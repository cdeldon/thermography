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

### Documentation
Build the documentation as follows:
```lang=bash
$ cd docs/
$ make html
```

This generates an *html* documentation of the source code in `docs/_build/html`
and the root `.html` file is indicated by `docs/_build/html/index.html`.