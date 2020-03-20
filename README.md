# Reduced Dilation-Erosion Perceptron for Binary Classification

Dilation and erosion are two elementary operations from mathematical morphology. The dilation-erosion perceptron (DEP) is a morphological neural network obtained by a convex combination of a dilation and an erosion followed by the application of a hard-limiter function for binary classification tasks. The file rDEP.py implements the DEP classifier based on convex-concave procedure proposed by Charisoupoulus and Maragos.

As a lattice computing model, however, the DEP classifier assumes the feature and class spaces are partially ordered sets. In many practical situations, however, there is no natural ordering for the feature patterns. Using concepts from multi-valued mathematical morphology, we introduced the reduced dilation-erosion (r-DEP) classifier (see ArXiv preprinte entitled "Reduced Dilation-Erosion Perceptron for Binary Classification). An r-DEP classifier is obtained by endowing the feature space with an appropriate reduced ordering. Such reduced ordering can be determined using two approaches: One based on an ensemble of support vector classifiers (SVCs) with different kernels and the other based on a bagging of similar SVCs trained using different samples of the training set. The file rDEP.py also implements both ensemble and bagging r-DEP classifiers. The Example - Double Moon illustrates these two approaches.

Using several binary classification datasets from the OpenML repository, the ensemble and bagging r-DEP classifiers yielded in mean higher balanced accuracy scores than the linear, polynomial, and radial basis function (RBF) SVCs as well as their ensemble and a bagging of RBF SVCs. You can reproduce the computational experiments calling the file Experiment_Binary_Datasets.py.

## Getting Started

This repository contain the python source-codes of reduced dilation-erosion perceptron (r-DEP) classifier, as described in the paper "Reduced Dilation-Erosion Perceptron for Binary Classification" by Marcos Eduardo Valle (see ArXiv paper). The Jupyter-notebook of an example is available in this repository. The source-code of the computational experiment considering 30 binary classification tasks is also available.

## Usage and required modules

Import the DEP classifier and the reduced mapping transform using:
```
from rDEP import DEP, EnsembleTransform 
```
The rDEP module require and have been tested using the following modules and versions:
* python: 3.7.3 
* numpy: 1.16.2
* matplotlib: 3.1.0
* sklearn: 0.21.2
* cvxpy: 1.0.25
* dccp: 1.0.0
* MOSEK: 9.1.13

*Remark:* We used MOSEK, which can be obtained at https://www.mosek.com/, as the default solver for the convex-concave optimization problem. Other solvers can be used instead of the MOSEK. For example, the CVXOPT solver which is available on cvxpy can be used but, unfortunately, this solver is very slow making it inappropriate for medium and large scale problems.  

## Usage of the DEP classifier

The DEP classifier is compatible with scikit-learn API. You create a DEP classifier with the command:
```
clf = DEP(weighted = True, ref = "maximum", C = 1.e-2, 
                 beta = None, beta_loss = "hinge", Split2Beta = False, 
                 solver = cp.Mosek, verbose = False)
```
where all the parameters are optimal (the default values are specified above). 
### Parameters:

   * *weighted*: True or False. The slack variables are weighted according to the weighting scheme described in the paper.
   * *ref*: "maximum" (default), "minimum", or "mean". Establishes the reference in the regularization term on the objective  of the convex-concave procedure. Maximum favors the largest maximum number of patterns while the minimum seeks the least number of patterns.
   * *C*: The regularization parameter (default C = 1.e-2).
   * *beta*: None (default) or a float between 0 and 1. Beta determines the convex combination of the dilation and erosion. A minimization procedure is used to find determine beta when it is None.
   * *beta_loss*: "hinge" (default) or "squared_hinge". Hinge or squared hinge loss functions are used to determine beta when its value is not provided (beta = None).
   * *Split2Beta*: False (default) or True. If True, the training data is splited into 3 stratified folds. Two folds are used to determine the synaptic weights while the remaining is used to determine beta. The whole training set is used to determine both synaptic weights and beta when Split2Beta is False.
   * *solver*: Solver used to solve the convex-concave optimization problem. Default solver = cp.Mosek. The option solver = cp.CVXOPT is an alternative but it will take long time to solve medium and large optimization problems.  
   * *verbose*: False (default) or True. Enable verbose output.

### Attributes:
  * classes_: The classes labels.
  * dil_: Synaptic weights of the dilation-based perceptron.
  * ero_: Synaptic weights of the erosion-based perceptron.
  * beta: Value used to compute the convex combination of the dilation-based and erosion-based perceptrons.
  
### Methods:
  * clf.fit(X, y): Fit the DEP classifier clf using the training data X and y.
  * clf.decision_function(X): Evaluates the decision function for the samples in X.
  * clf.predict(X): Perform classification on samples in X.
  * clf.score(X,y): Return the mean accuracy on the given test data and labels.
  * clf.show(X,y, ind = [0,1], show_boxes = True, decision_boundary = True, Nh = 101): Shows the scatter plot of the data in the dimensions ind[0] and ind[1]. 
    * show_boxes = True shows the projection of the hyperboxes delimited by the dilation-based and the erosion-based perceptrons. 
    * decision_boundary = True shows the decision boundary when the data are two-dimensional. 
    * Nh is the number of points on each axis used to draw the decision boundary.
    
## Usage of the Ensemble (or Bagging) Transform

The Ensemble Transform is compatible with scikit-learn transform API. You create a transformation mapping rho using the command:
```
rho = EnsembleTransform(clfs)
```
where *clfs* contains an ensemble of classifiers that allows for a decision function evaluation. For example, *clfs* can be a scikit-learn VotingClassfier (ensemble) or a BaggingClassifier: 
   * clfs = VotingClassifier([("RBF SVC",SVC(gamma="scale")),("Linear SVC",SVC(kernel="linear"))]) 
   * clfs = BaggingClassifier(base_estimator=SVC(gamma="scale"))
The parameters and attributes are determined by clfs.

### Methods:
  * rho.fit(X,y): Fit ensemble transformation rho using the training data X and y.
  * rho.transform(X): Transform the samples on X.
  
## Example 

See the Jupyter-notebook **Example - Double Moon** for an example of the usage of the DEP classifier and the Ensemble Transform as well as their effective combination (pipeline) on a r-DEP classifier.
  
## Authors

* **Marcos Eduardo Valle** - *University of Campinas*
