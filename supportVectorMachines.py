
# Optimisation
import cvxopt
from cvxopt import matrix, solvers
# Math
import numpy as np
from numpy import linalg #import linear algebra
#Scikit
import math
from sklearn.svm import SVC
import plotting
import generate_data
def predict(X,Y,alpha,b,x,sigma):
  result=0.0
  for i in range(X.shape[0]):
    norm= np.linalg.norm(np.subtract(X[i,:] , x))
    res= math.exp(-(norm**2)/(2*(sigma**2)))
    result+=(alpha[i]*Y[i]*res)
  result+=b
  return(result)

class SMO(object):
  def __init__(self,X,Y,C =1.0, tol = math.pow(10,-3), max_iter =50, sigma=1):
    '''
      tol: is the convergence tolerance parameter, and is typically set to around 0.01 to 0.001
      X: input data
      Y : label
      C: regukarization paramter
      max_iter: max iterations
    '''
    self.X = X
    self.Y = Y
    self.tol = float(tol)
    self.max_iter = int(max_iter)
    self.C = float(C)
    self.sigma = sigma

  def smo(self):
    alpha = np.zeros(shape=(self.X.shape[0],1))
    b = 0
    passes = 0
    E = np.zeros(shape=(self.X.shape[0],1))
    while(passes <= self.max_iter):
      num_changed_alphas=0
      for i in range(self.X.shape[0]):
        #Calculate Ei = f(x(i)) − y(i) using (2)
        E[i]=(predict(self.X,self.Y,alpha,b,self.X[i,:],self.sigma)-self.Y[i])
        if ( (-self.Y[i]*E[i]>tol and -alpha[i]>-C) or (self.Y[i]*E[i]>self.tol and alpha[i]>0) ):
          j=i
          while(j==i):
            j=random.randrange(self.X.shape[0]) #get any other data point other than i
    
          E[j] = (predict(self.X,self.Y,alpha,b,self.X[j,:],self.sigma)-self.Y[j]) #for other data point

          alpha_old[i]=alpha[i]
          alpha_old[j]=alpha[j]
          
          #computing L and h values

          if (self.Y[i]!=self.Y[j]):
            L=max(0,alpha[j]-alpha[i])
            H=min(self.C,self.C+alpha[j]-alpha[i])
          else:
            L=max(0,alpha[i]+alpha[j]-self.C)
            H=min(self.C,alpha[i]+alpha[j])
      
          if (L==H):
            continue
          eta = 2*gaussian_kernel(self.X[i,:],self.X[j,:],self.sigma)
          eta=eta-gaussian_kernel(self.X[i,:],self.X[i,:],self.sigma)
          eta=eta-gaussian_kernel(self.X[j,:],self.X[j,:],self.sigma)
        
          if (eta >= 0):
            continue
        
          #clipping
          
          alpha[j]= alpha_old[j]-((self.Y[j]*(E[i]-E[j]))/eta)

          if (alpha[j] > H):
            alpha[j]=H
          elif (alpha[j]<L):
            alpha[j]=L
          else:
            pass  #do nothing
    
          if (abs(alpha[j]-alpha_old[j]) < self.tol):
            continue
        
          alpha[i] += (self.Y[i]*self.Y[j]*(alpha_old[j] - alpha[j])) #both alphas are updated


          ii = gaussian_kernel(self.X[i,:],self.X[i,:],self.sigma)
          ij = gaussian_kernel(self.X[i,:],self.X[j,:],self.sigma)
          jj = gaussian_kernel(self.X[j,:],self.X[j,:],self.sigma)          

          b1= b-E[i]- (self.Y[i]*ii*(alpha[i]-alpha_old[i]))- (self.Y[j]*ij*(alpha[j]-alpha_old[j]))
          b2= b-E[j]- (self.Y[i]*ij*(alpha[i]-alpha_old[i]))- (self.Y[j]*jj*(alpha[j]-alpha_old[j]))
          if (alpha[i] > 0 and alpha[i]<C):
            b=b1
          elif (alpha[j] > 0 and alpha[j] <C):
            b=b2
          else:
            b=(b1+b2)/2.0
        
          num_changed_alphas+=1
    
        #ended if
      #ended for
      if (num_changed_alphas == 0):
        passes+=1
      else:
        passes=0
    #end while

    return alpha,b  
class SVM(object):
    '''
        parameters:::
        C:  is a regularization parameter for SVMs.
        gamma:  tries to exactly fit the training data set 
        Degree: of the polynomial kernel function (‘poly’). Ignored by all other kernels
    '''
    def __init__(self, kernel='linear', C=1.0, gamma=1.0, degree = 3):
        self.C = float(C)
        self.gamma = float(gamma)
        self.degree = int(degree)
        self.kernel = str(kernel)
        self.iterations = 50
        self.bias = 0
      
    def linear_kernel(self, x1, x2):
      '''
        Linear Kernel is used when the data is Linearly separable, that is, 
        it can be separated using a single Line.

        For linear kernel the equation for prediction for a new input 
        using the dot product between the input (x) and each support vector(xi) 
        is calculated as follows:
        f(x) = B(0) + sum(ai * (x,xi))

        params:
        x1: input vector
        x2: support vectors

      '''
      return np.dot(x1, x2)

    def polynomial_kernel(self, x, y,C=1, d=3):
        # Inputs:
        #   x   : input var
        #   y   : support vectors
        #   c   : param svm
        #   d   : degree of polynomial.
        #   K(x,xi) = C + sum(x * xi)^d
        return (np.dot(x, y) + C) ** d

    def build_Kernelmatrix(self,X, n_samples):
     '''
        param: input-samples
     '''
     K = np.zeros((n_samples, n_samples))


    def build_Kernelmatrix(self, X, n_samples): #Gram matrix
     K = np.zeros((n_samples, n_samples))
     for i in range(n_samples):
        for j in range(n_samples):
          if self.kernel == 'linear':
            K[i, j] = self.linear_kernel(X[i], X[j])
          if self.kernel=='gaussian':
            K[i, j] = self.gaussian_kernel(X[i], X[j], self.gamma)  
            self.C = None  
          if self.kernel == 'polynomial':
            K[i, j] = self.polynomial_kernel(X[i], X[j], self.C, self.degree)
     return K

    def gaussian_kernel(self, x, y, gamma=0.5):
        # Inputs:
        #   x   : input var
        #   y   : support vectors
        #   gamma   : param
        # K(x,xi) = exp(-gamma * sum((x — xi²)).
        return np.exp(-gamma*linalg.norm(x - y) ** 2 )

    def fit(self, X, y):
        '''
          param:
            X: input smaples
            y: num features

          Let's train our model,
            [Step1] Get num_samples and num_features
            [Step2] Precompute Kernel Matrix since our dataset is small

        '''
        #[step1]:
        n_samples, n_features = X.shape #get datasize_shape
        #[Step2]:
        K = self.build_Kernelmatrix(X, n_samples)
        #Hypothesis: sign(sum^S a * y * kernel + b)
        n = np.outer(y,y)
        P = matrix(n * K)
        q = matrix(-np.ones((n_samples, 1)))
        G = matrix(np.diag(np.ones(n_samples) * -1))
        h = matrix(np.zeros(n_samples))
        b = matrix(np.zeros(1))
        A = matrix(y.reshape(1, -1))
        solvers.options['show_progress'] = False
        # Solve Quadratic Programming problem:
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
      

        alphas = np.ravel(solution['x'])       
        # Support vectors have non zero lagrange multipliers
        sv = alphas > 1e-4
        ind = np.arange(len(alphas))[sv]
        self.alphas = alphas[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        # Bias (For linear it is the intercept):
        
        self.b = 0
        for n in range(len(self.alphas)):
            # For all support vectors:
            self.b += self.sv_y[n]
            self.b -= np.sum(self.alphas * self.sv_y * K[ind[n], sv])
        self.b = self.b / len(self.alphas)

        # Weight vector
        if self.kernel == 'linear':
            self.w = np.zeros(n_features)
            for n in range(len(self.alphas)):
                self.w += self.alphas[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None
        
    def helper(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.alphas, self.sv_y, self.sv):
                    # a : Lagrange multipliers, sv : support vectors.
                    # Hypothesis: sign(sum^S a * y * kernel + b)
                    if self.kernel == 'linear':
                        s += a * sv_y * self.linear_kernel(X[i], sv)
                    if self.kernel=='gaussian':
                        s += a * sv_y * self.gaussian_kernel(X[i], sv, self.gamma)   # Kernel trick.
                        self.C = None   
                    if self.kernel == 'polynomial':
                        s += a * sv_y * self.polynomial_kernel(X[i], sv, self.C, self.degree)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        #Sign values shows the class value of the training sample
        return np.sign(self.helper(X))


def svm(kernel='linear', C=0, gamma=0.1, degree=3, dataset='linearly_separable',split_data = 0.7):

    #Create dataset object
    Data_obj = generate_data.Data();                                                   
    Plot_obj = plotting.Plotting();                                                   # Plotting object.

    # Generate Data:
    if kernel == 'linear':
        if dataset == 'linearly_separable':
            X1, y1, X2, y2 = Data_obj.generate_linearly_separable_data(seed=1)
        else:
            X1, y1, X2, y2 = Data_obj.gen_lin_separable_overlap_data(seed=1)  # Generate the examples.
    
    elif kernel=='polynomial' or kernel=='gaussian':
        X1, y1, X2, y2 = Data_obj.gen_non_lin_separable_data(seed=1)
 
    #Create SVM object
    objFit = SVM(kernel=kernel, C=C, gamma=gamma, degree=degree)  
    #Split dataset into training and testing samples                                           
    X_train, y_train, X_test, y_test = generate_data.Data.split_data(X1, y1, X2, y2,split_data)   
    #Fit model
    objFit.fit(X_train, y_train)                                            
   # y_predict = objFit.predict(X_test)                                     
    Plot_obj.plot_margin(X_train[y_train == 1], X_train[y_train == -1], objFit); 


if __name__ == "__main__":
    svm(kernel='linear',C=100)
    svm('polynomial',C=1, degree=3)
    svm('gaussian', gamma = 0.1)
    #print("Finished")
