Patata
Patata is a Python package for K-NN regression and data preprocessing. It provides two main functions:
fritas for encoding categorical columns using Label Encoding, and bravas for finding the best k value (number of neighbors) in K-NN regression using cross-validation.

Installation
You can install the package using pip:

pip install patata

Usage
First, import the necessary functions from the package:

from patata import fritas, bravas

fritas
fritas encodes all categorical columns in a given pandas DataFrame using Label Encoding.


bravas
bravas finds the best k value (number of neighbors) in K-NN regression using cross-validation based on the mean squared error.

License
This package is released under the MIT License.

Author
Patata Team

Email: demstalfer@gmail.com