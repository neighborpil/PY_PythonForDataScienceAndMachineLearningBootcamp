```python
import numpy as np
import pandas as pd
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
%matplotlib inline
```


```python
from sklearn.datasets import load_breast_cancer
```


```python
cancer = load_breast_cancer()
```


```python
cancer.keys()
```




    dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])




```python
print(cancel['DESCR'])
```

    .. _breast_cancer_dataset:
    
    Breast cancer wisconsin (diagnostic) dataset
    --------------------------------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 569
    
        :Number of Attributes: 30 numeric, predictive attributes and the class
    
        :Attribute Information:
            - radius (mean of distances from center to points on the perimeter)
            - texture (standard deviation of gray-scale values)
            - perimeter
            - area
            - smoothness (local variation in radius lengths)
            - compactness (perimeter^2 / area - 1.0)
            - concavity (severity of concave portions of the contour)
            - concave points (number of concave portions of the contour)
            - symmetry
            - fractal dimension ("coastline approximation" - 1)
    
            The mean, standard error, and "worst" or largest (mean of the three
            worst/largest values) of these features were computed for each image,
            resulting in 30 features.  For instance, field 0 is Mean Radius, field
            10 is Radius SE, field 20 is Worst Radius.
    
            - class:
                    - WDBC-Malignant
                    - WDBC-Benign
    
        :Summary Statistics:
    
        ===================================== ====== ======
                                               Min    Max
        ===================================== ====== ======
        radius (mean):                        6.981  28.11
        texture (mean):                       9.71   39.28
        perimeter (mean):                     43.79  188.5
        area (mean):                          143.5  2501.0
        smoothness (mean):                    0.053  0.163
        compactness (mean):                   0.019  0.345
        concavity (mean):                     0.0    0.427
        concave points (mean):                0.0    0.201
        symmetry (mean):                      0.106  0.304
        fractal dimension (mean):             0.05   0.097
        radius (standard error):              0.112  2.873
        texture (standard error):             0.36   4.885
        perimeter (standard error):           0.757  21.98
        area (standard error):                6.802  542.2
        smoothness (standard error):          0.002  0.031
        compactness (standard error):         0.002  0.135
        concavity (standard error):           0.0    0.396
        concave points (standard error):      0.0    0.053
        symmetry (standard error):            0.008  0.079
        fractal dimension (standard error):   0.001  0.03
        radius (worst):                       7.93   36.04
        texture (worst):                      12.02  49.54
        perimeter (worst):                    50.41  251.2
        area (worst):                         185.2  4254.0
        smoothness (worst):                   0.071  0.223
        compactness (worst):                  0.027  1.058
        concavity (worst):                    0.0    1.252
        concave points (worst):               0.0    0.291
        symmetry (worst):                     0.156  0.664
        fractal dimension (worst):            0.055  0.208
        ===================================== ====== ======
    
        :Missing Attribute Values: None
    
        :Class Distribution: 212 - Malignant, 357 - Benign
    
        :Creator:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian
    
        :Donor: Nick Street
    
        :Date: November, 1995
    
    This is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets.
    https://goo.gl/U2Uwz2
    
    Features are computed from a digitized image of a fine needle
    aspirate (FNA) of a breast mass.  They describe
    characteristics of the cell nuclei present in the image.
    
    Separating plane described above was obtained using
    Multisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree
    Construction Via Linear Programming." Proceedings of the 4th
    Midwest Artificial Intelligence and Cognitive Science Society,
    pp. 97-101, 1992], a classification method which uses linear
    programming to construct a decision tree.  Relevant features
    were selected using an exhaustive search in the space of 1-4
    features and 1-3 separating planes.
    
    The actual linear program used to obtain the separating plane
    in the 3-dimensional space is that described in:
    [K. P. Bennett and O. L. Mangasarian: "Robust Linear
    Programming Discrimination of Two Linearly Inseparable Sets",
    Optimization Methods and Software 1, 1992, 23-34].
    
    This database is also available through the UW CS ftp server:
    
    ftp ftp.cs.wisc.edu
    cd math-prog/cpo-dataset/machine-learn/WDBC/
    
    .. topic:: References
    
       - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction 
         for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on 
         Electronic Imaging: Science and Technology, volume 1905, pages 861-870,
         San Jose, CA, 1993.
       - O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and 
         prognosis via linear programming. Operations Research, 43(4), pages 570-577, 
         July-August 1995.
       - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques
         to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) 
         163-171.
    


```python
df_feat = pd.DataFrame(cancer['data'], columns=cancel['feature_names'])
```


```python
df_feat.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst radius</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>0.09744</td>
      <td>...</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>0.05883</td>
      <td>...</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
cancer['target_names'] # malignant: 악성의, benign: 양성의
```




    array(['malignant', 'benign'], dtype='<U9')




```python
from sklearn.model_selection import train_test_split
```


```python
X = df_feat
y = cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=auto)
```


```python
from sklearn.svm import SVC
```


```python
model = SVC()
```


```python
model.fit(X_train, y_train)
```




    SVC()




```python
predictions = model.predict(X_test)
```


```python
from sklearn.metrics import classification_report, confusion_matrix
```


```python
print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))
```

    [[ 57   8]
     [  3 103]]
    
    
                  precision    recall  f1-score   support
    
               0       0.95      0.88      0.91        65
               1       0.93      0.97      0.95       106
    
        accuracy                           0.94       171
       macro avg       0.94      0.92      0.93       171
    weighted avg       0.94      0.94      0.94       171
    
    


```python
from sklearn.model_selection import GridSearchCV
```


```python
param_grid = {'C':[0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
```


```python
grid = GridSearchCV(SVC(), param_grid, verbose=3)
```


```python
grid.fit(X_train, y_train)
```

    Fitting 5 folds for each of 25 candidates, totalling 125 fits
    [CV 1/5] END ....................C=0.1, gamma=1;, score=0.637 total time=   0.0s
    [CV 2/5] END ....................C=0.1, gamma=1;, score=0.625 total time=   0.0s
    [CV 3/5] END ....................C=0.1, gamma=1;, score=0.625 total time=   0.0s
    [CV 4/5] END ....................C=0.1, gamma=1;, score=0.633 total time=   0.0s
    [CV 5/5] END ....................C=0.1, gamma=1;, score=0.633 total time=   0.0s
    [CV 1/5] END ..................C=0.1, gamma=0.1;, score=0.637 total time=   0.0s
    [CV 2/5] END ..................C=0.1, gamma=0.1;, score=0.625 total time=   0.0s
    [CV 3/5] END ..................C=0.1, gamma=0.1;, score=0.625 total time=   0.0s
    [CV 4/5] END ..................C=0.1, gamma=0.1;, score=0.633 total time=   0.0s
    [CV 5/5] END ..................C=0.1, gamma=0.1;, score=0.633 total time=   0.0s
    [CV 1/5] END .................C=0.1, gamma=0.01;, score=0.637 total time=   0.0s
    [CV 2/5] END .................C=0.1, gamma=0.01;, score=0.625 total time=   0.0s
    [CV 3/5] END .................C=0.1, gamma=0.01;, score=0.625 total time=   0.0s
    [CV 4/5] END .................C=0.1, gamma=0.01;, score=0.633 total time=   0.0s
    [CV 5/5] END .................C=0.1, gamma=0.01;, score=0.633 total time=   0.0s
    [CV 1/5] END ................C=0.1, gamma=0.001;, score=0.637 total time=   0.0s
    [CV 2/5] END ................C=0.1, gamma=0.001;, score=0.625 total time=   0.0s
    [CV 3/5] END ................C=0.1, gamma=0.001;, score=0.625 total time=   0.0s
    [CV 4/5] END ................C=0.1, gamma=0.001;, score=0.633 total time=   0.0s
    [CV 5/5] END ................C=0.1, gamma=0.001;, score=0.633 total time=   0.0s
    [CV 1/5] END ...............C=0.1, gamma=0.0001;, score=0.938 total time=   0.0s
    [CV 2/5] END ...............C=0.1, gamma=0.0001;, score=0.875 total time=   0.0s
    [CV 3/5] END ...............C=0.1, gamma=0.0001;, score=0.938 total time=   0.0s
    [CV 4/5] END ...............C=0.1, gamma=0.0001;, score=0.924 total time=   0.0s
    [CV 5/5] END ...............C=0.1, gamma=0.0001;, score=0.899 total time=   0.0s
    [CV 1/5] END ......................C=1, gamma=1;, score=0.637 total time=   0.0s
    [CV 2/5] END ......................C=1, gamma=1;, score=0.625 total time=   0.0s
    [CV 3/5] END ......................C=1, gamma=1;, score=0.625 total time=   0.0s
    [CV 4/5] END ......................C=1, gamma=1;, score=0.633 total time=   0.0s
    [CV 5/5] END ......................C=1, gamma=1;, score=0.633 total time=   0.0s
    [CV 1/5] END ....................C=1, gamma=0.1;, score=0.637 total time=   0.0s
    [CV 2/5] END ....................C=1, gamma=0.1;, score=0.625 total time=   0.0s
    [CV 3/5] END ....................C=1, gamma=0.1;, score=0.625 total time=   0.0s
    [CV 4/5] END ....................C=1, gamma=0.1;, score=0.633 total time=   0.0s
    [CV 5/5] END ....................C=1, gamma=0.1;, score=0.633 total time=   0.0s
    [CV 1/5] END ...................C=1, gamma=0.01;, score=0.637 total time=   0.0s
    [CV 2/5] END ...................C=1, gamma=0.01;, score=0.625 total time=   0.0s
    [CV 3/5] END ...................C=1, gamma=0.01;, score=0.625 total time=   0.0s
    [CV 4/5] END ...................C=1, gamma=0.01;, score=0.633 total time=   0.0s
    [CV 5/5] END ...................C=1, gamma=0.01;, score=0.633 total time=   0.0s
    [CV 1/5] END ..................C=1, gamma=0.001;, score=0.950 total time=   0.0s
    [CV 2/5] END ..................C=1, gamma=0.001;, score=0.925 total time=   0.0s
    [CV 3/5] END ..................C=1, gamma=0.001;, score=0.912 total time=   0.0s
    [CV 4/5] END ..................C=1, gamma=0.001;, score=0.937 total time=   0.0s
    [CV 5/5] END ..................C=1, gamma=0.001;, score=0.924 total time=   0.0s
    [CV 1/5] END .................C=1, gamma=0.0001;, score=0.950 total time=   0.0s
    [CV 2/5] END .................C=1, gamma=0.0001;, score=0.938 total time=   0.0s
    [CV 3/5] END .................C=1, gamma=0.0001;, score=0.963 total time=   0.0s
    [CV 4/5] END .................C=1, gamma=0.0001;, score=0.949 total time=   0.0s
    [CV 5/5] END .................C=1, gamma=0.0001;, score=0.937 total time=   0.0s
    [CV 1/5] END .....................C=10, gamma=1;, score=0.637 total time=   0.0s
    [CV 2/5] END .....................C=10, gamma=1;, score=0.625 total time=   0.0s
    [CV 3/5] END .....................C=10, gamma=1;, score=0.625 total time=   0.0s
    [CV 4/5] END .....................C=10, gamma=1;, score=0.633 total time=   0.0s
    [CV 5/5] END .....................C=10, gamma=1;, score=0.633 total time=   0.0s
    [CV 1/5] END ...................C=10, gamma=0.1;, score=0.637 total time=   0.0s
    [CV 2/5] END ...................C=10, gamma=0.1;, score=0.625 total time=   0.0s
    [CV 3/5] END ...................C=10, gamma=0.1;, score=0.625 total time=   0.0s
    [CV 4/5] END ...................C=10, gamma=0.1;, score=0.633 total time=   0.0s
    [CV 5/5] END ...................C=10, gamma=0.1;, score=0.633 total time=   0.0s
    [CV 1/5] END ..................C=10, gamma=0.01;, score=0.650 total time=   0.0s
    [CV 2/5] END ..................C=10, gamma=0.01;, score=0.625 total time=   0.0s
    [CV 3/5] END ..................C=10, gamma=0.01;, score=0.637 total time=   0.0s
    [CV 4/5] END ..................C=10, gamma=0.01;, score=0.633 total time=   0.0s
    [CV 5/5] END ..................C=10, gamma=0.01;, score=0.633 total time=   0.0s
    [CV 1/5] END .................C=10, gamma=0.001;, score=0.938 total time=   0.0s
    [CV 2/5] END .................C=10, gamma=0.001;, score=0.912 total time=   0.0s
    [CV 3/5] END .................C=10, gamma=0.001;, score=0.875 total time=   0.0s
    [CV 4/5] END .................C=10, gamma=0.001;, score=0.937 total time=   0.0s
    [CV 5/5] END .................C=10, gamma=0.001;, score=0.911 total time=   0.0s
    [CV 1/5] END ................C=10, gamma=0.0001;, score=0.938 total time=   0.0s
    [CV 2/5] END ................C=10, gamma=0.0001;, score=0.938 total time=   0.0s
    [CV 3/5] END ................C=10, gamma=0.0001;, score=0.950 total time=   0.0s
    [CV 4/5] END ................C=10, gamma=0.0001;, score=0.924 total time=   0.0s
    [CV 5/5] END ................C=10, gamma=0.0001;, score=0.937 total time=   0.0s
    [CV 1/5] END ....................C=100, gamma=1;, score=0.637 total time=   0.0s
    [CV 2/5] END ....................C=100, gamma=1;, score=0.625 total time=   0.0s
    [CV 3/5] END ....................C=100, gamma=1;, score=0.625 total time=   0.0s
    [CV 4/5] END ....................C=100, gamma=1;, score=0.633 total time=   0.0s
    [CV 5/5] END ....................C=100, gamma=1;, score=0.633 total time=   0.0s
    [CV 1/5] END ..................C=100, gamma=0.1;, score=0.637 total time=   0.0s
    [CV 2/5] END ..................C=100, gamma=0.1;, score=0.625 total time=   0.0s
    [CV 3/5] END ..................C=100, gamma=0.1;, score=0.625 total time=   0.0s
    [CV 4/5] END ..................C=100, gamma=0.1;, score=0.633 total time=   0.0s
    [CV 5/5] END ..................C=100, gamma=0.1;, score=0.633 total time=   0.0s
    [CV 1/5] END .................C=100, gamma=0.01;, score=0.650 total time=   0.0s
    [CV 2/5] END .................C=100, gamma=0.01;, score=0.625 total time=   0.0s
    [CV 3/5] END .................C=100, gamma=0.01;, score=0.637 total time=   0.0s
    [CV 4/5] END .................C=100, gamma=0.01;, score=0.633 total time=   0.0s
    [CV 5/5] END .................C=100, gamma=0.01;, score=0.633 total time=   0.0s
    [CV 1/5] END ................C=100, gamma=0.001;, score=0.938 total time=   0.0s
    [CV 2/5] END ................C=100, gamma=0.001;, score=0.912 total time=   0.0s
    [CV 3/5] END ................C=100, gamma=0.001;, score=0.875 total time=   0.0s
    [CV 4/5] END ................C=100, gamma=0.001;, score=0.937 total time=   0.0s
    [CV 5/5] END ................C=100, gamma=0.001;, score=0.911 total time=   0.0s
    [CV 1/5] END ...............C=100, gamma=0.0001;, score=0.950 total time=   0.0s
    [CV 2/5] END ...............C=100, gamma=0.0001;, score=0.925 total time=   0.0s
    [CV 3/5] END ...............C=100, gamma=0.0001;, score=0.925 total time=   0.0s
    [CV 4/5] END ...............C=100, gamma=0.0001;, score=0.949 total time=   0.0s
    [CV 5/5] END ...............C=100, gamma=0.0001;, score=0.911 total time=   0.0s
    [CV 1/5] END ...................C=1000, gamma=1;, score=0.637 total time=   0.0s
    [CV 2/5] END ...................C=1000, gamma=1;, score=0.625 total time=   0.0s
    [CV 3/5] END ...................C=1000, gamma=1;, score=0.625 total time=   0.0s
    [CV 4/5] END ...................C=1000, gamma=1;, score=0.633 total time=   0.0s
    [CV 5/5] END ...................C=1000, gamma=1;, score=0.633 total time=   0.0s
    [CV 1/5] END .................C=1000, gamma=0.1;, score=0.637 total time=   0.0s
    [CV 2/5] END .................C=1000, gamma=0.1;, score=0.625 total time=   0.0s
    [CV 3/5] END .................C=1000, gamma=0.1;, score=0.625 total time=   0.0s
    [CV 4/5] END .................C=1000, gamma=0.1;, score=0.633 total time=   0.0s
    [CV 5/5] END .................C=1000, gamma=0.1;, score=0.633 total time=   0.0s
    [CV 1/5] END ................C=1000, gamma=0.01;, score=0.650 total time=   0.0s
    [CV 2/5] END ................C=1000, gamma=0.01;, score=0.625 total time=   0.0s
    [CV 3/5] END ................C=1000, gamma=0.01;, score=0.637 total time=   0.0s
    [CV 4/5] END ................C=1000, gamma=0.01;, score=0.633 total time=   0.0s
    [CV 5/5] END ................C=1000, gamma=0.01;, score=0.633 total time=   0.0s
    [CV 1/5] END ...............C=1000, gamma=0.001;, score=0.938 total time=   0.0s
    [CV 2/5] END ...............C=1000, gamma=0.001;, score=0.912 total time=   0.0s
    [CV 3/5] END ...............C=1000, gamma=0.001;, score=0.875 total time=   0.0s
    [CV 4/5] END ...............C=1000, gamma=0.001;, score=0.937 total time=   0.0s
    [CV 5/5] END ...............C=1000, gamma=0.001;, score=0.911 total time=   0.0s
    [CV 1/5] END ..............C=1000, gamma=0.0001;, score=0.938 total time=   0.0s
    [CV 2/5] END ..............C=1000, gamma=0.0001;, score=0.900 total time=   0.0s
    [CV 3/5] END ..............C=1000, gamma=0.0001;, score=0.900 total time=   0.0s
    [CV 4/5] END ..............C=1000, gamma=0.0001;, score=0.937 total time=   0.0s
    [CV 5/5] END ..............C=1000, gamma=0.0001;, score=0.949 total time=   0.0s
    




    GridSearchCV(estimator=SVC(),
                 param_grid={'C': [0.1, 1, 10, 100, 1000],
                             'gamma': [1, 0.1, 0.01, 0.001, 0.0001]},
                 verbose=3)




```python
grid.best_params_
```




    {'C': 1, 'gamma': 0.0001}




```python
grid.best_estimator_
```




    SVC(C=1, gamma=0.0001)




```python
grid_predictions = grid.predict(X_test)
```


```python
print(confusion_matrix(y_test, grid_predictions))
print('\n')
print(classification_report(y_test, grid_predictions))
```

    [[ 58   7]
     [  4 102]]
    
    
                  precision    recall  f1-score   support
    
               0       0.94      0.89      0.91        65
               1       0.94      0.96      0.95       106
    
        accuracy                           0.94       171
       macro avg       0.94      0.93      0.93       171
    weighted avg       0.94      0.94      0.94       171
    
    


```python

```
