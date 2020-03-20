import numpy as np
import cvxpy as cp
import dccp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import StratifiedKFold
import time

# ##################################################
# Plot the decision boundary of a classifier
# ##################################################

def decision_boundary(self, X, y, ind=[0,1], Nh = 101, colors="black", label = None):  
    # Scatter plot
    sc = plt.scatter(X[:,ind[0]], X[:,ind[1]], c = y.astype(int))
    xlimits = plt.xlim()
    ylimits = plt.ylim()
    
    if X.shape[1]>2:
        print("Dimension larger than two! Cannot show the decision boundary!")
    else:
        # create a mesh to plot in
        x_min, x_max = xlimits[0], xlimits[1]
        y_min, y_max = ylimits[0], ylimits[1]
        hx = (x_max-x_min)/Nh
        hy = (y_max-y_min)/Nh
        xx, yy = np.meshgrid(np.arange(x_min, x_max, hx),np.arange(y_min, y_max, hy))
            
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        Z = np.array(self.predict(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)
            
        # Put the result into a color plot
        plt.contourf(xx, yy, Z, alpha = 0.1, cmap='plasma')
        plt.contour(xx, yy, Z, colors=colors, linestyles = 'dashed')
        
    plt.grid("True")
    plt.xlabel("Variable %d" % ind[0])
    plt.ylabel("Variable %d" % ind[1])
    return sc

# ##################################################
# Ensemble (or Bagging) Transform
# ##################################################
class EnsembleTransform(TransformerMixin, BaseEstimator):
    def __init__(self,ensemble):
        self.ensemble = ensemble
    
    def fit(self, X, y):
        (self.ensemble).fit(X, y)
        return self
    
    def transform(self, X):
        return np.vstack([clf.decision_function(X) for clf in (self.ensemble).estimators_]).T


# ##################################################
# Dilation-Erosion Perceptron with DCCP
# ##################################################    
class DEP(BaseEstimator, ClassifierMixin):
    
    def __init__(self, weighted = True, ref = "maximum", C = 1.e-2, 
                 beta = None, beta_loss = "hinge", Split2Beta = False, 
                 solver = cp.MOSEK, verbose = False):
        self.verbose = verbose
        self.solver = solver
        self.weighted = weighted
        self.ref = ref
        self.C = C
        self.beta = beta
        self.beta_loss = beta_loss
        self.Split2Beta = Split2Beta
    
    def fit(self, X, y):
        start_time = time.time()
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        if len(self.classes_)>2:
            print("Dilation-Erosion Morphological Perceptron can be used for binary classification!")
            return        
        
        if self.Split2Beta == True:
            skf = StratifiedKFold(n_splits=3, shuffle=True)
            WM_index, beta_index = next(iter(skf.split(X,y)))
            X_WM, X_beta = X[WM_index], X[beta_index]
            y_WM, y_beta = y[WM_index], y[beta_index]
        else:
            X_WM, X_beta = X, X
            y_WM, y_beta = y, y
        
        M, N = X_beta.shape
        
        indPos = (y_WM == self.classes_[1])
        Xpos = X_WM[indPos,:]
        Xneg = X_WM[~indPos,:]
        Mpos = Xpos.shape[0]
        Mneg = Xneg.shape[0]

        if self.weighted == True:
            Lpos = 1/pairwise_distances(Xpos,[np.mean(Xpos,axis=0)],metric="euclidean").flatten()
            Lneg = 1/pairwise_distances(Xneg,[np.mean(Xneg,axis=0)],metric="euclidean").flatten()
            nuPos = Lpos/Lpos.max()
            nuNeg = Lneg/Lneg.max()
        else:
            nuPos = np.ones((Mpos))
            nuNeg = np.ones((Mneg))
        
        # Solve DCCP problem for dilation
        if self.ref == "mean":
            ref = -np.mean(Xneg,axis=0).reshape((1,N))
        elif self.ref == "maximum":
            ref = -np.max(Xneg,axis=0).reshape((1,N))
        elif self.ref == "minimum":
            ref = -np.min(Xneg,axis=0).reshape((1,N))
        else:
            ref = np.zeros((1,N))
            
        w = cp.Variable((1,N))
        xiPos = cp.Variable((Mpos))
        xiNeg = cp.Variable((Mneg))
        
        lossDil = cp.sum(nuPos*cp.pos(xiPos))/Mpos+cp.sum(nuNeg*cp.pos(xiNeg))/Mneg+self.C*cp.norm(w-ref,1)
        objectiveDil = cp.Minimize(lossDil)
                
        ZposDil = cp.max(np.ones((Mpos,1))@w + Xpos, axis=1)
        ZnegDil = cp.max(np.ones((Mneg,1))@w + Xneg, axis=1)  
        constraintsDil = [ZposDil >= -xiPos, ZnegDil <= xiNeg]

        probDil = cp.Problem(objectiveDil,constraintsDil)            
        probDil.solve(solver=self.solver, method = 'dccp', verbose = self.verbose)
        self.dil_ = (w.value).flatten()
        
        # Solve DCCP problem for erosion
        if self.ref == "mean":
            ref = -np.mean(Xpos,axis=0).reshape((1,N))
        elif self.ref == "maximum":
            ref = -np.min(Xpos,axis=0).reshape((1,N))
        elif self.ref == "minimum":
            ref = -np.max(Xpos,axis=0).reshape((1,N))
        else:
            ref = np.zeros((1,N))
            
        m = cp.Variable((1,N))
        etaPos = cp.Variable((Mpos))
        etaNeg = cp.Variable((Mneg))
        
        lossEro = cp.sum(nuPos*cp.pos(etaPos))/Mpos+cp.sum(nuNeg*cp.pos(etaNeg))/Mneg+self.C*cp.norm(m-ref,1)
        objectiveEro = cp.Minimize(lossEro)
                
        ZposEro = cp.min(np.ones((Mpos,1))@m + Xpos, axis=1)
        ZnegEro = cp.min(np.ones((Mneg,1))@m + Xneg, axis=1)  
        constraintsEro = [ZposEro >= -etaPos, ZnegEro <= etaNeg]

        probEro = cp.Problem(objectiveEro,constraintsEro)            
        probEro.solve(solver=self.solver, method = 'dccp', verbose = self.verbose)
        self.ero_ = (m.value).flatten()
        
        # Fine tune beta
        if self.beta == None:
            beta = cp.Variable(nonneg=True)
            beta.value = 0.5
            
            if self.beta_loss == "squared_hinge":
                # Squared Hinge Loss
                lossBeta = cp.sum_squares(cp.pos(-cp.multiply(2*((y_beta == self.classes_[1]).astype(int))-1,
                                beta*cp.max(np.ones((M,1))@w.value + X_beta, axis=1) +
                                (1-beta)*cp.min(np.ones((M,1))@m.value + X_beta, axis=1))))
            else:
                # Hinge Loss
                lossBeta = cp.sum(cp.pos(-cp.multiply(2*((y_beta == self.classes_[1]).astype(int))-1,
                                beta*cp.max(np.ones((M,1))@w.value + X_beta, axis=1) +
                                (1-beta)*cp.min(np.ones((M,1))@m.value + X_beta, axis=1))))
            
            constraintsBeta = [beta<=1]
            probBeta = cp.Problem(cp.Minimize(lossBeta),constraintsBeta)
            probBeta.solve(solver = cp.SCS, verbose = self.verbose, warm_start=True)
            self.beta = beta.value
        
        if self.verbose == True:
            print("\nTime to train: %2.2f seconds." % (time.time() - start_time))
        return self
            
    def decision_function(self, X):
        # Check is fit had been called
        check_is_fitted(self,attributes="dil_")
        
        # Input validation
        X = check_array(X)
        
        M,N = X.shape
        Y = np.zeros((M,2))
        # Compute the dilation
        Y[:,0] = np.amax(np.ones((M,1))@self.dil_.reshape((1,N))+X,axis=1)
        # Compute the erosion
        Y[:,1] = np.amin(np.ones((M,1))@self.ero_.reshape((1,N))+X,axis=1)
        
        return np.dot(Y,np.array([self.beta,1-self.beta]))
    
    def predict(self, X):
        return np.array([self.classes_[(y>=0).astype(int)] for y in self.decision_function(X)])

    def show(self, X, y, ind=[0,1], show_boxes = True, decision_boundary = True, Nh = 101):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Check is fit had been called
        check_is_fitted(self,attributes="dil_")

        plt.figure(figsize=(10, 8))
    
        # Scatter plot
        sc = plt.scatter(X[:,ind[0]], X[:,ind[1]], c = y.astype(int))
        xlimits = plt.xlim()
        ylimits = plt.ylim()
    
        if decision_boundary:
            if X.shape[1]>2:
                print("Dimension larger than two! Cannot show the decision boundary!")
            else:
                # create a mesh to plot in
                x_min, x_max = xlimits[0], xlimits[1]
                y_min, y_max = ylimits[0], ylimits[1]
                hx = (x_max-x_min)/Nh
                hy = (y_max-y_min)/Nh
                xx, yy = np.meshgrid(np.arange(x_min, x_max, hx),np.arange(y_min, y_max, hy))
            
                # Plot the decision boundary. For that, we will assign a color to each
                # point in the mesh [x_min, m_max]x[y_min, y_max].
                Z = np.array(self.predict(np.c_[xx.ravel(), yy.ravel()]))
                Z = Z.reshape(xx.shape)
            
                # Put the result into a color plot
                plt.contourf(xx, yy, Z, alpha = 0.1, cmap='plasma')
                plt.contour(xx, yy, Z, colors='black', linestyles = 'dashed')
        
        if show_boxes:
            # Draw dilation box
            box = [-1000*np.ones((X.shape[1],)),-self.dil_]
            Vertices = np.array([box[0][ind],[box[1][ind[0]],box[0][ind[1]]],box[1][ind],[box[0][ind[0]],box[1][ind[1]]]])
            plt.gca().add_patch(Polygon(Vertices, alpha = 0.3, color=sc.to_rgba(0)))
    
            # Draw erosion box
            box = [-self.ero_,1000*np.ones((X.shape[1],))]
            Vertices = np.array([box[0][ind],[box[1][ind[0]],box[0][ind[1]]],box[1][ind],[box[0][ind[0]],box[1][ind[1]]]])
            plt.gca().add_patch(Polygon(Vertices, alpha = 0.3, color=sc.to_rgba(1)))
    
        plt.grid("True")
        plt.xlabel("Variable %d" % ind[0])
        plt.ylabel("Variable %d" % ind[1])
