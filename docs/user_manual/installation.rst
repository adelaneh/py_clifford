============
Installation
============

Requirements
------------
* Python 2.7 or Python 3.4+

Platforms
---------
*py_clifford* has been tested on Linux.

Dependencies
------------
* pandas (provides data structures to store and manage tables)
* numpy (used to store similarity matrices and required by pandas)
* tensorflow (used to implement the RNNs and optimize them)
* tqdm (provides progress tracking facilities for training the networks)

Installing Using pip
--------------------
To install the package using pip, execute the following
command::

    pip install -U py_clifford


The above command will install *py_clifford* and all of its dependencies.


Installing from Source Distribution
-----------------------------------
Clone the *py_clifford* package from GitHub

    git clone https://github.com/adelaneh/py_clifford.git

Then,  execute the following commands from the package root::

    python setup.py install

which installs *py_clifford* into the default Python directory on your machine. If you do not have installation permission for that directory then you can install the package in your
home directory as follows::

    python setup.py install --user

For more information see this StackOverflow `link <http://stackoverflow.com/questions/14179941/how-to-install-python-packages-without-root-privileges>`_.

The above commands will install *py_clifford* and all of its
dependencies.
