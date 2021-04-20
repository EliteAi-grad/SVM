import numpy as np
# Copyright 2021 Google Inc. All Rights Reserved.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

   # http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#==============================================================================
#

class Data(object):
    #Dataset1: Generating linearly seperable dataset
    def generate_linearlydataset_linear(self,seed=1):
        np.random.seed(seed)
        mean1 = np.array([0,3])
        mean2 = np.array([3,0])
        return mean1, mean2
    
    def generate_helperdataset(self,mean1, cov, mean2):
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def generate_linearly_separable_data(self,seed=1):
        mean1, mean2 = self.generate_linearlydataset_linear()
        cov = np.array([[0.4, 0.7], [0.7, 0.4]])
        return self.generate_helperdataset(mean1,cov,mean2)

    def gen_non_lin_separable_data(self,seed=1):
        np.random.seed(seed)
        mean1 = [-3, 5]
        mean2 = [5, -3]
        mean3 = [5, -7]
        mean4 = [-7, 5]
        cov = [[1.0, 0.7], [0.7, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_lin_separable_overlap_data(self,seed=1):
        np.random.seed(seed)
        mean1 = np.array([-1, 4])
        mean2 = np.array([4, -1])
        cov = np.array([[4.5, 1.7], [1.7, 4.5]])
        return self.generate_helperdataset(mean1,cov,mean2)

    def split_data(X1, y1, X2, y2,percent):
        dataset_size = len(X1)
        threshold = int(dataset_size*percent);

        # Training data: binary classifier X1, X2
        X1_train = X1[:threshold]
        y1_train = y1[:threshold]
        X2_train = X2[:threshold]
        y2_train = y2[:threshold]

        #stack datasets
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))

        # Test data:
        X1_test = X1[threshold:]
        y1_test = y1[threshold:]
        X2_test = X2[threshold:]
        y2_test = y2[threshold:]

        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))

        return X_train, y_train, X_test, y_test
