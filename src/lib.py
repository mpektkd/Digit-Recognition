import numpy as np
import matplotlib.pyplot as plt
import copy
from seaborn.miscplot import dogplot


import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

def create_tables(file):
  y_temp = []   
  X_temp = []   # lists to append data

  with open(file, 'r') as f:  # open file
    for line in f:
      line = line.strip(' \n')
      num = line.split(' ')
      num = list(map(float,num))  # read number of each line as a float
      y_temp.append(num[0])
      X_temp.append(num[1:257])   # create lists of numbers
    y_temp = np.array(y_temp)
    X_temp = np.array(X_temp)     # tranform lists into ndarrays

  return y_temp, X_temp

def plot_digits_samples(index, digits, X, y=None):
    '''Takes a dataset and selects one example from each label and plots it in subplots

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
    '''

    if y is not None:
        for i in range (digits):
            w = np.where(y == i)[0][0]  # find index
            index.append(w)             # append it to list

    fig = plt.figure(figsize=(12,12))
    k = 0
    for i in range(digits):
        k = 251+i%5 if i <= 4 else 151+i%5
        fig.add_subplot(k)
        plt.imshow(np.reshape(X[index[i]],(16,16))) 
    plt.show()

def digit_mean(X, y, digit):
    '''Calculates the mean for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The mean value of the digits for every pixel
    '''
    return np.mean(X[y==digit], axis=0)

def digit_variance(X, y, digit):
    '''Calculates the variance for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The variance value of the digits for every pixel
    '''
    return np.var(X[y==digit], axis=0)

# Plot Function for desicion surfaces
def plot_clf(clf, X, y, labels):
  # set figsize and title for the plots
  plt.figure(figsize=(8,8))
  title = ('Decision surface of Classifier')

  # Set-up grid for plotting.

  X0, X1 = X[:, 0], X[:, 1]
  x_min, x_max = X0.min() - 1, X0.max() + 1
  y_min, y_max = X1.min() - 1, X1.max() + 1

  xx, yy = np.meshgrid(np.arange(x_min, x_max, .05), np.arange(y_min, y_max, .05))

  Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  colors = ['blue', 'red', 'lime', 'yellow', 'peachpuff', 'gray', 'rosybrown', 'cyan', 'orange', 'violet']

  # Set-up grid for plotting.
  levels = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  lines = plt.contour(xx, yy, Z, levels=10, colors='black', linewidths=1.5)
  out = plt.contourf(xx, yy, Z, levels=levels, colors=colors, alpha=0.8)

  for i in labels:
    plt.scatter( X0[y==i], X1[y==i], label=labels[i], c=colors[i], s=60, alpha=0.9, edgecolors='k')

  plt.title(title)
  plt.legend()
  plt.show()

# Plot Function for learning-curves(shows how accuracy is affected w.r.t. the train-size)
def plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(0, 1)):
    plt.figure()
    plt.title("Learning Curve")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt

class EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin):
    """Classify samples based on the distance from the mean feature value"""

    def __init__(self,):
        self.digits = 10
        self.X_mean_ = None


    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Calculates self.X_mean_ based on the mean
        feature values in X for each class.

        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)

        fit always returns self.
        """
        self.X_mean_ = np.array([digit_mean(X, y, i) for i in range(self.digits)])

        return self

    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        '''Classifiece based on the euclidean distance between samples in X and template vectors in X_mean

        Args:
            X (np.ndarray): Digits data (nsamples x nfeatures)
            X_mean (np.ndarray): Digits data (n_classes x nfeatures)

        Returns:
            (np.ndarray) predictions (nsamples)
        '''
        # Clone X_mean as rows of a general array to speed-up the array substraction 
        norms = np.linalg.norm(
                        np.tile(
                                self.X_mean_, (X.shape[0],1,1)
                            ) - X[:,None]
                        , axis=2
                    )
        return np.argmin(norms, axis=1)


    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """

        ### Or use accuracy_score of sklearn ###

        y_pred = self.predict(X)    # calculate predictions 
        cor = 0   # variable to calculate correct predictions
        for i in range(len(y)): # for all the test set
            if (y_pred[i] == y[i]): # compare prediction with actual result
                cor += 1  # if correct, count it

        return cor/len(y)  # calculate accuracy and return

# Class for the custom impelemntation of Naive-Bayes Classifier
class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, use_unit_variance=False):
        self.use_unit_variance = use_unit_variance
        self.digits = 10

        self.n_samples = 0  #number of samples  
        self.n_feats = 0    # number of features
        self.n_classes = 0   # number of classes
        self.classes_ = np.array([])    # tags of the classes

        self.X_mean_ = np.array([]) # array of means for each class
        self.X_var_ = np.array([])  # array of variance for each class

        self.pC = np.zeros((self.digits,)) # pC is a vector with the probability of each class

    def _mean(self, X, y): 
        return np.array([digit_mean(X, y, i) for i in range(self.digits)])

    def _var(self, X, y): 
        return np.array([digit_variance(X, y, i) for i in range(self.digits)])

    def _prior(self,y): 
        """ Prior probability, P(y) for each y """
        pC = np.zeros((int(self.n_classes),))  # create an array to save prior probabilities
        for dig in y: 
            pC[int(dig)] += 1   
        pC /= y.shape[0]  # calculate probabilities for every class

        return pC


    def _normal(self,x,mean,var): 
        """ Gaussian Normal Distribution """
        try:
            multiplier = (1/ float(np.sqrt(2 * np.pi * var))) 
            exp = np.exp(-((x - mean)**2 / float(2 * var)))     # create pdf of normal distribution
            product = multiplier * exp
        except:
            product = 0.0   # if var is the nearest value to zero that is not 0 -> we get product=0

        return product


    def _observation(self, x, c):
        """Uses Normal Distribution to get, P(x|C) = P(x1|C) * P(x2|C) .. * P(xn|C)
        
        Args:
            x (np.ndarray): 1D-Array (nfeatures) 
            c (int): class

        Returns:
            (int): Observation Probability for that class    
        """
        pdfs = []
        for i in range(self.n_feats):
            mu = self.X_mean_[c][i]     # mean
            var = self.X_var_[c][i] if self.X_var_[c][i] > 1e-2 else 1e-2   # threshold variance for better results

            pdfs.append( self._normal(x[i],mu,var) )    # calculate pdfs

        pxC = np.prod(pdfs)

        return pxC

    def fit(self, X, y):
        self.n_samples, self.n_feats = X.shape
        self.n_classes = np.unique(y).shape[0]
        self.classes_ = np.array([int(x) for x in np.unique(y)])

        self.X_mean_ = self._mean(X,y) 
        self.X_var_ = ( self._var(X,y) if self.use_unit_variance is False else np.ones((int(self.n_classes),int(self.n_feats))) ) 

        self.pC = self._prior(y)
        return self


    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        samples, _ = X.shape
        result = []

        for i in range(samples):    # for every sample
            posterior = []
            for c in self.classes_:
                posterior.append( self._observation(X[i],c)*self.pC[c] )    # calculate posterior
            
            # we omitt the evidence prob, as we need the argmax and the denominator is common
            idx = np.argmax(posterior)  #find argmax
            result.append(self.classes_[idx])
        
        return np.array(result)

   
    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        y_pred = self.predict(X)    # calculate predictions 
        cor = 0   # variable to calculate correct predictions
        for i in range(0, len(y),1): # for all the test set
            if (y_pred[i] == y[i]): # compare prediction with actual result
                cor += 1  # if correct, count it

        return cor/len(y)  # calculate accuracy and return

# Transformation Class(preprocessing)
class CenteringApplier(object):
    """
    class which mean-centers and makes std=1
    """
    def __init__(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def __call__(self, datum):
        if len(datum) == 2:
            x, y = datum[0], datum[1]
            return (x - self.mean)/ self.std, y
        else:
            x = datum
            return (x - self.mean)/ self.std

# Class to organize the dataset
class DigitData(Dataset):

    def __init__(self, X, Y, transform=None):
        self.transform = transform
        self.data = list(zip(X,Y)) if Y is not None else X

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        datum = self.data[idx]
        return self.transform(datum) if self.transform else datum

# Class for the NN-layer
class LinearWActivation(nn.Module):

    def __init__(self, in_features, out_features):
        super(LinearWActivation,self).__init__()
        self.f = nn.Linear(in_features,out_features, dtype=torch.float64)  
        #self.a = nn.LogSigmoid()                      
        self.a = nn.ReLU()      # it has been noticed that, ReLU gives better results
        #self.a = nn.Sigmoid()

    def forward(self,x):
        return self.a(self.f(x))

# Class for the MLP(simple MLP with only input, output layer without a reguralization e.g dropout)
class MyPredictionNet(nn.Module):

    def __init__(self, layers, n_features, h_features, n_classes):
        super(MyPredictionNet, self).__init__()
        layers_in = [n_features] + layers
        layers_out = layers + [h_features]
        self.f = nn.Sequential(
            *[LinearWActivation(in_feats, out_feats) 
             for in_feats, out_feats in zip(layers_in,layers_out)])
        self.clf = nn.Linear(h_features, n_classes, dtype=torch.float64)

    def forward(self, x):
        return self.clf(self.f(x))
        #return self.f(x)

# Class for MLP-Classifier
class PytorchNNModel(nn.Module, BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        super(PytorchNNModel, self).__init__()
        self.batch_sz = kwargs['BATCH_SZ'] if kwargs else 128
        self.epochs = kwargs['EPOCHS'] if kwargs else 30
        self.layers = kwargs['layers'] if kwargs else []
        self.n_features = kwargs['n_features'] if kwargs else 256
        self.hidden_layer = kwargs['hidden_layer'] if kwargs else 128
        self.n_classes = kwargs['n_classes'] if kwargs else 10
        self.ETA = kwargs['ETA'] if kwargs else 1e-2
        self.weight_decay = kwargs['weight_decay'] if kwargs else 1e-7
        self.model = MyPredictionNet(self.layers, self.n_features, self.hidden_layer, self.n_classes)   
        self.criterion = nn.CrossEntropyLoss() 
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.ETA, weight_decay=self.weight_decay)
        print(f"The network architecture is: \n {self.model}")

    def fit(self, X, y):
        tr_dataset = DigitData(X, y, CenteringApplier(X))

        l = len(tr_dataset)
        train_split, val_split = int(.8*l), l-int(.8*l)
        train_dataset, val_dataset = torch.utils.data.random_split(tr_dataset, (train_split, val_split))

        train_dl = DataLoader(train_dataset, batch_size=self.batch_sz, shuffle=True)
        val_dl = DataLoader(val_dataset, batch_size=self.batch_sz, shuffle=True)
        error = []
        for epoch in range(self.epochs): # loop through dataset
            self.model.train() # gradients "on"  
            for i, data in enumerate(train_dl): # loop through batches
                X_batch, Y_batch = data # get the features and labels
                self.optimizer.zero_grad() # ALWAYS USE THIS!! 
                out = self.model(X_batch) # forward pass
                loss = self.criterion(out, Y_batch) # compute per batch loss 
                loss.backward() # compurte gradients based on the loss function
                self.optimizer.step() # update weights


            self.model.eval() # turns off batchnorm/dropout 
            running_average_loss = 0
            with torch.no_grad(): # no gradients required!! eval mode, speeds up computation
                for i, data in enumerate(val_dl):
                    X_batch, Y_batch = data # test data and labels
                    out = self.model(X_batch) # get net's predictions
                    loss = self.criterion(out, Y_batch) # compute per batch loss 
            running_average_loss += loss.detach().item()
            print("Epoch error: {} \t Epoch: {}".format(running_average_loss, epoch))
            # We keep the best model, till every single epoch-another tecnhic is early stopping, (in another repo)-
            if epoch == 0:
                error.append(running_average_loss)
                error.append(copy.deepcopy(self.model))
                error.append(epoch)
            if error[0] > running_average_loss:
                error[0] = running_average_loss
                error[1] = copy.deepcopy(self.model)
                error[2] = epoch

        self.best_model = copy.deepcopy(error[1])
        print("Less error: {} \t Epoch: {}".format(error[0], error[2]))
        return self
    
    def predict(self, X):
        test_dataset = DigitData(X, None, CenteringApplier(X))        

        test_dl = DataLoader(test_dataset, batch_size=self.batch_sz)
        
        # IMPORTANT: switch to eval mode
        # disable regularization layers, such as Dropout
        self.best_model.eval()

        y_pred = []  # the predicted labels

        # obtain the model's device ID
        device = next(self.best_model.parameters()).device

        # IMPORTANT: in evaluation mode, we don't want to keep the gradients
        # so we do everything under torch.no_grad()
        with torch.no_grad():
            for _, batch in enumerate(test_dl, 1):
                # get the inputs (batch)
                inputs = batch

                # Step 1 - move the batch tensors to the right device
                inputs = inputs.to(device) # EX9

                # Step 2 - forward pass: y' = model(x)
                outputs = self.best_model(inputs) # EX9

                _, pred = outputs.max(1) # argmax since output is a prob distribution  # EX9

                y_pred += list(pred) # EX9
        return y_pred

    def score(self, X, y):
        # Return accuracy score.
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

def evaluate_sklearn_nb_classifier(X, y, folds=5):
    """ Create an sklearn naive bayes classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    return cross_val_score(sklearn.naive_bayes.GaussianNB(), X, y, cv=KFold(n_splits=folds), scoring="accuracy")  # calculate  k-fold-cross-validation 
    
def evaluate_custom_nb_classifier(X, y, folds=5):
    """ Create a custom naive bayes classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    return cross_val_score(CustomNBClassifier(), X, y, cv=KFold(n_splits=folds), scoring="accuracy")  # calculate  k-fold-cross-validation 

def evaluate_nn_classifier(X, y, folds=5):
    """ Create a pytorch nn classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    kwargs = {
        'BATCH_SZ': 128,
        'EPOCHS': 30,
        'layers': [],
        'n_features': 256,
        'hidden_layer': 128,
        'n_classes': 10,
        'ETA': 1e-2,
        'weight_decay': 1e-7
    }

    return cross_val_score(PytorchNNModel(**kwargs), X, y, cv=KFold(n_splits=folds), scoring="accuracy")  # calculate  k-fold-cross-validation 

    

def evaluate_voting_classifier(X, y, folds=5):
    """ Create a voting ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """   
    clf1 = EuclideanDistanceClassifier()
    clf2 = sklearn.naive_bayes.GaussianNB()
    clf3 = KNeighborsClassifier(n_neighbors=5)
    clf_V = VotingClassifier(estimators=[('eucl', clf1), ('gnb', clf2), ('knn', clf3)], voting='hard')
    return cross_val_score(clf_V, X, y, cv=KFold(n_splits=folds), scoring="accuracy")  # calculate  k-fold-cross-validation 

    
    

def evaluate_bagging_classifier(X, y, folds=5):
    """ Create a bagging ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = EuclideanDistanceClassifier()
    clf_B = BaggingClassifier(base_estimator=clf, n_estimators=10, random_state=0)
    return cross_val_score(clf_B, X, y, cv=KFold(n_splits=folds), scoring="accuracy")  # calculate  k-fold-cross-validation 

    
