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

### Documentation
Build the documentation as follows:
```lang=bash
$ cd docs/
$ make html
```

This generates an *html* documentation of the source code in `docs/_build/html`
and the root `.html` file is indicated by `docs/_build/html/index.html`.