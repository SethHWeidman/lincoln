Speeding up conv with Cython

Following here: https://discuss.pytorch.org/t/fast-tensor-access-in-python/334/2. Speed it up!

Followed [here](http://docs.cython.org/en/latest/src/tutorial/numpy.html#adding-types) to get basic example working.

Followed [here](https://stackoverflow.com/questions/14657375/cython-fatal-error-numpy-arrayobject-h-no-such-file-or-directory) to add the `numpy` thing to `setup.py`.

Instructions:

1. `python setup.py build_ext --inplace`
2. `python test_conv.py`
