___

<a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
___
# Support Vector Machines Project 

Welcome to your Support Vector Machine Project! Just follow along with the notebook and instructions below. We will be analyzing the famous iris data set!

## The Data
For this series of lectures, we will be using the famous [Iris flower data set](http://en.wikipedia.org/wiki/Iris_flower_data_set). 

The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by Sir Ronald Fisher in the 1936 as an example of discriminant analysis. 

The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor), so 150 total samples. Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

Here's a picture of the three different Iris types:


```python
# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)
```




    
![jpeg](output_1_0.jpg)
    




```python
# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)
```




    
![jpeg](output_2_0.jpg)
    




```python
# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)
```




    
![jpeg](output_3_0.jpg)
    



The iris dataset contains measurements for 150 iris flowers from three different species.

The three classes in the Iris dataset:

    Iris-setosa (n=50)
    Iris-versicolor (n=50)
    Iris-virginica (n=50)

The four features of the Iris dataset:

    sepal length in cm
    sepal width in cm
    petal length in cm
    petal width in cm

## Get the data

**Use seaborn to get the iris data by using: iris = sns.load_dataset('iris') **


```python
import numpy as np
import pandas as pd
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
iris = sns.load_dataset('iris')
```


```python
iris.head()
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>



Let's visualize the data and get you started!

## Exploratory Data Analysis

Time to put your data viz skills to the test! Try to recreate the following plots, make sure to import the libraries you'll need!

**Import some libraries you think you'll need.**


```python

```

** Create a pairplot of the data set. Which flower species seems to be the most separable?**


```python
sns.pairplot(iris, hue='species')
```




    <seaborn.axisgrid.PairGrid at 0x2273fe78340>




    
![png](output_12_1.png)
    



```python

```




    <seaborn.axisgrid.PairGrid at 0x12afb9cc0>




    
![png](output_13_1.png)
    


**Create a kde plot of sepal_length versus sepal width for setosa species of flower.**


```python
sns.kdeplot(x='sepal_width', data=iris.loc[iris['species'] == 'setosa'], y='sepal_length', fill=True)
```




    <AxesSubplot:xlabel='sepal_width', ylabel='sepal_length'>




    
![png](output_15_1.png)
    



```python

```




    <matplotlib.axes._subplots.AxesSubplot at 0x12f102080>




    
![png](output_16_1.png)
    


# Train Test Split

** Split your data into a training set and a testing set.**


```python
from sklearn.model_selection import train_test_split
```


```python
X = iris.drop('species', axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

# Train a Model

Now its time to train a Support Vector Machine Classifier. 

**Call the SVC() model from sklearn and fit the model to the training data.**


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

```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)



## Model Evaluation

**Now get predictions from the model and create a confusion matrix and a classification report.**


```python
predictions = model.predict(X_test)
```


```python
from sklearn.metrics import confusion_matrix, classification_report
```


```python
print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))
```

    [[19  0  0]
     [ 0 15  0]
     [ 0  0 16]]
    
    
                  precision    recall  f1-score   support
    
          setosa       1.00      1.00      1.00        19
      versicolor       1.00      1.00      1.00        15
       virginica       1.00      1.00      1.00        16
    
        accuracy                           1.00        50
       macro avg       1.00      1.00      1.00        50
    weighted avg       1.00      1.00      1.00        50
    
    


```python

```

    [[15  0  0]
     [ 0 13  1]
     [ 0  0 16]]
    


```python

```

                 precision    recall  f1-score   support
    
         setosa       1.00      1.00      1.00        15
     versicolor       1.00      0.93      0.96        14
      virginica       0.94      1.00      0.97        16
    
    avg / total       0.98      0.98      0.98        45
    
    

Wow! You should have noticed that your model was pretty good! Let's see if we can tune the parameters to try to get even better (unlikely, and you probably would be satisfied with these results in real like because the data set is quite small, but I just want you to practice using GridSearch.

## Gridsearch Practice

** Import GridsearchCV from SciKit Learn.**


```python
from sklearn.model_selection import GridSearchCV
```

**Create a dictionary called param_grid and fill out some parameters for C and gamma.**


```python
param_grid={'C':[0.01, 0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
```

** Create a GridSearchCV object and fit it to the training data.**


```python
grid = GridSearchCV(SVC(), param_grid, verbose=3)
```


```python
grid.fit(X_train, y_train)
```

    Fitting 5 folds for each of 30 candidates, totalling 150 fits
    [CV 1/5] END ...................C=0.01, gamma=1;, score=0.350 total time=   0.0s
    [CV 2/5] END ...................C=0.01, gamma=1;, score=0.350 total time=   0.0s
    [CV 3/5] END ...................C=0.01, gamma=1;, score=0.350 total time=   0.0s
    [CV 4/5] END ...................C=0.01, gamma=1;, score=0.350 total time=   0.0s
    [CV 5/5] END ...................C=0.01, gamma=1;, score=0.550 total time=   0.0s
    [CV 1/5] END .................C=0.01, gamma=0.1;, score=0.350 total time=   0.0s
    [CV 2/5] END .................C=0.01, gamma=0.1;, score=0.350 total time=   0.0s
    [CV 3/5] END .................C=0.01, gamma=0.1;, score=0.350 total time=   0.0s
    [CV 4/5] END .................C=0.01, gamma=0.1;, score=0.350 total time=   0.0s
    [CV 5/5] END .................C=0.01, gamma=0.1;, score=0.500 total time=   0.0s
    [CV 1/5] END ................C=0.01, gamma=0.01;, score=0.350 total time=   0.0s
    [CV 2/5] END ................C=0.01, gamma=0.01;, score=0.350 total time=   0.0s
    [CV 3/5] END ................C=0.01, gamma=0.01;, score=0.350 total time=   0.0s
    [CV 4/5] END ................C=0.01, gamma=0.01;, score=0.350 total time=   0.0s
    [CV 5/5] END ................C=0.01, gamma=0.01;, score=0.500 total time=   0.0s
    [CV 1/5] END ...............C=0.01, gamma=0.001;, score=0.350 total time=   0.0s
    [CV 2/5] END ...............C=0.01, gamma=0.001;, score=0.350 total time=   0.0s
    [CV 3/5] END ...............C=0.01, gamma=0.001;, score=0.350 total time=   0.0s
    [CV 4/5] END ...............C=0.01, gamma=0.001;, score=0.350 total time=   0.0s
    [CV 5/5] END ...............C=0.01, gamma=0.001;, score=0.450 total time=   0.0s
    [CV 1/5] END ..............C=0.01, gamma=0.0001;, score=0.350 total time=   0.0s
    [CV 2/5] END ..............C=0.01, gamma=0.0001;, score=0.350 total time=   0.0s
    [CV 3/5] END ..............C=0.01, gamma=0.0001;, score=0.350 total time=   0.0s
    [CV 4/5] END ..............C=0.01, gamma=0.0001;, score=0.350 total time=   0.0s
    [CV 5/5] END ..............C=0.01, gamma=0.0001;, score=0.450 total time=   0.0s
    [CV 1/5] END ....................C=0.1, gamma=1;, score=0.900 total time=   0.0s
    [CV 2/5] END ....................C=0.1, gamma=1;, score=0.850 total time=   0.0s
    [CV 3/5] END ....................C=0.1, gamma=1;, score=0.900 total time=   0.0s
    [CV 4/5] END ....................C=0.1, gamma=1;, score=1.000 total time=   0.0s
    [CV 5/5] END ....................C=0.1, gamma=1;, score=0.900 total time=   0.0s
    [CV 1/5] END ..................C=0.1, gamma=0.1;, score=0.850 total time=   0.0s
    [CV 2/5] END ..................C=0.1, gamma=0.1;, score=0.900 total time=   0.0s
    [CV 3/5] END ..................C=0.1, gamma=0.1;, score=0.900 total time=   0.0s
    [CV 4/5] END ..................C=0.1, gamma=0.1;, score=1.000 total time=   0.0s
    [CV 5/5] END ..................C=0.1, gamma=0.1;, score=0.850 total time=   0.0s
    [CV 1/5] END .................C=0.1, gamma=0.01;, score=0.350 total time=   0.0s
    [CV 2/5] END .................C=0.1, gamma=0.01;, score=0.350 total time=   0.0s
    [CV 3/5] END .................C=0.1, gamma=0.01;, score=0.350 total time=   0.0s
    [CV 4/5] END .................C=0.1, gamma=0.01;, score=0.350 total time=   0.0s
    [CV 5/5] END .................C=0.1, gamma=0.01;, score=0.500 total time=   0.0s
    [CV 1/5] END ................C=0.1, gamma=0.001;, score=0.350 total time=   0.0s
    [CV 2/5] END ................C=0.1, gamma=0.001;, score=0.350 total time=   0.0s
    [CV 3/5] END ................C=0.1, gamma=0.001;, score=0.350 total time=   0.0s
    [CV 4/5] END ................C=0.1, gamma=0.001;, score=0.350 total time=   0.0s
    [CV 5/5] END ................C=0.1, gamma=0.001;, score=0.450 total time=   0.0s
    [CV 1/5] END ...............C=0.1, gamma=0.0001;, score=0.350 total time=   0.0s
    [CV 2/5] END ...............C=0.1, gamma=0.0001;, score=0.350 total time=   0.0s
    [CV 3/5] END ...............C=0.1, gamma=0.0001;, score=0.350 total time=   0.0s
    [CV 4/5] END ...............C=0.1, gamma=0.0001;, score=0.350 total time=   0.0s
    [CV 5/5] END ...............C=0.1, gamma=0.0001;, score=0.450 total time=   0.0s
    [CV 1/5] END ......................C=1, gamma=1;, score=0.950 total time=   0.0s
    [CV 2/5] END ......................C=1, gamma=1;, score=0.900 total time=   0.0s
    [CV 3/5] END ......................C=1, gamma=1;, score=0.900 total time=   0.0s
    [CV 4/5] END ......................C=1, gamma=1;, score=1.000 total time=   0.0s
    [CV 5/5] END ......................C=1, gamma=1;, score=0.950 total time=   0.0s
    [CV 1/5] END ....................C=1, gamma=0.1;, score=0.950 total time=   0.0s
    [CV 2/5] END ....................C=1, gamma=0.1;, score=0.850 total time=   0.0s
    [CV 3/5] END ....................C=1, gamma=0.1;, score=0.900 total time=   0.0s
    [CV 4/5] END ....................C=1, gamma=0.1;, score=1.000 total time=   0.0s
    [CV 5/5] END ....................C=1, gamma=0.1;, score=0.950 total time=   0.0s
    [CV 1/5] END ...................C=1, gamma=0.01;, score=1.000 total time=   0.0s
    [CV 2/5] END ...................C=1, gamma=0.01;, score=0.900 total time=   0.0s
    [CV 3/5] END ...................C=1, gamma=0.01;, score=0.950 total time=   0.0s
    [CV 4/5] END ...................C=1, gamma=0.01;, score=1.000 total time=   0.0s
    [CV 5/5] END ...................C=1, gamma=0.01;, score=0.850 total time=   0.0s
    [CV 1/5] END ..................C=1, gamma=0.001;, score=0.350 total time=   0.0s
    [CV 2/5] END ..................C=1, gamma=0.001;, score=0.350 total time=   0.0s
    [CV 3/5] END ..................C=1, gamma=0.001;, score=0.350 total time=   0.0s
    [CV 4/5] END ..................C=1, gamma=0.001;, score=0.350 total time=   0.0s
    [CV 5/5] END ..................C=1, gamma=0.001;, score=0.450 total time=   0.0s
    [CV 1/5] END .................C=1, gamma=0.0001;, score=0.350 total time=   0.0s
    [CV 2/5] END .................C=1, gamma=0.0001;, score=0.350 total time=   0.0s
    [CV 3/5] END .................C=1, gamma=0.0001;, score=0.350 total time=   0.0s
    [CV 4/5] END .................C=1, gamma=0.0001;, score=0.350 total time=   0.0s
    [CV 5/5] END .................C=1, gamma=0.0001;, score=0.450 total time=   0.0s
    [CV 1/5] END .....................C=10, gamma=1;, score=0.950 total time=   0.0s
    [CV 2/5] END .....................C=10, gamma=1;, score=0.800 total time=   0.0s
    [CV 3/5] END .....................C=10, gamma=1;, score=0.900 total time=   0.0s
    [CV 4/5] END .....................C=10, gamma=1;, score=1.000 total time=   0.0s
    [CV 5/5] END .....................C=10, gamma=1;, score=0.950 total time=   0.0s
    [CV 1/5] END ...................C=10, gamma=0.1;, score=0.950 total time=   0.0s
    [CV 2/5] END ...................C=10, gamma=0.1;, score=0.850 total time=   0.0s
    [CV 3/5] END ...................C=10, gamma=0.1;, score=0.900 total time=   0.0s
    [CV 4/5] END ...................C=10, gamma=0.1;, score=1.000 total time=   0.0s
    [CV 5/5] END ...................C=10, gamma=0.1;, score=1.000 total time=   0.0s
    [CV 1/5] END ..................C=10, gamma=0.01;, score=0.950 total time=   0.0s
    [CV 2/5] END ..................C=10, gamma=0.01;, score=0.900 total time=   0.0s
    [CV 3/5] END ..................C=10, gamma=0.01;, score=0.900 total time=   0.0s
    [CV 4/5] END ..................C=10, gamma=0.01;, score=1.000 total time=   0.0s
    [CV 5/5] END ..................C=10, gamma=0.01;, score=0.950 total time=   0.0s
    [CV 1/5] END .................C=10, gamma=0.001;, score=1.000 total time=   0.0s
    [CV 2/5] END .................C=10, gamma=0.001;, score=0.900 total time=   0.0s
    [CV 3/5] END .................C=10, gamma=0.001;, score=0.950 total time=   0.0s
    [CV 4/5] END .................C=10, gamma=0.001;, score=1.000 total time=   0.0s
    [CV 5/5] END .................C=10, gamma=0.001;, score=0.850 total time=   0.0s
    [CV 1/5] END ................C=10, gamma=0.0001;, score=0.350 total time=   0.0s
    [CV 2/5] END ................C=10, gamma=0.0001;, score=0.350 total time=   0.0s
    [CV 3/5] END ................C=10, gamma=0.0001;, score=0.350 total time=   0.0s
    [CV 4/5] END ................C=10, gamma=0.0001;, score=0.350 total time=   0.0s
    [CV 5/5] END ................C=10, gamma=0.0001;, score=0.450 total time=   0.0s
    [CV 1/5] END ....................C=100, gamma=1;, score=0.950 total time=   0.0s
    [CV 2/5] END ....................C=100, gamma=1;, score=0.800 total time=   0.0s
    [CV 3/5] END ....................C=100, gamma=1;, score=0.900 total time=   0.0s
    [CV 4/5] END ....................C=100, gamma=1;, score=1.000 total time=   0.0s
    [CV 5/5] END ....................C=100, gamma=1;, score=0.900 total time=   0.0s
    [CV 1/5] END ..................C=100, gamma=0.1;, score=0.950 total time=   0.0s
    [CV 2/5] END ..................C=100, gamma=0.1;, score=0.800 total time=   0.0s
    [CV 3/5] END ..................C=100, gamma=0.1;, score=0.900 total time=   0.0s
    [CV 4/5] END ..................C=100, gamma=0.1;, score=0.950 total time=   0.0s
    [CV 5/5] END ..................C=100, gamma=0.1;, score=1.000 total time=   0.0s
    [CV 1/5] END .................C=100, gamma=0.01;, score=0.950 total time=   0.0s
    [CV 2/5] END .................C=100, gamma=0.01;, score=0.950 total time=   0.0s
    [CV 3/5] END .................C=100, gamma=0.01;, score=0.900 total time=   0.0s
    [CV 4/5] END .................C=100, gamma=0.01;, score=1.000 total time=   0.0s
    [CV 5/5] END .................C=100, gamma=0.01;, score=1.000 total time=   0.0s
    [CV 1/5] END ................C=100, gamma=0.001;, score=0.950 total time=   0.0s
    [CV 2/5] END ................C=100, gamma=0.001;, score=0.900 total time=   0.0s
    [CV 3/5] END ................C=100, gamma=0.001;, score=0.900 total time=   0.0s
    [CV 4/5] END ................C=100, gamma=0.001;, score=1.000 total time=   0.0s
    [CV 5/5] END ................C=100, gamma=0.001;, score=0.950 total time=   0.0s
    [CV 1/5] END ...............C=100, gamma=0.0001;, score=1.000 total time=   0.0s
    [CV 2/5] END ...............C=100, gamma=0.0001;, score=0.900 total time=   0.0s
    [CV 3/5] END ...............C=100, gamma=0.0001;, score=0.950 total time=   0.0s
    [CV 4/5] END ...............C=100, gamma=0.0001;, score=1.000 total time=   0.0s
    [CV 5/5] END ...............C=100, gamma=0.0001;, score=0.850 total time=   0.0s
    [CV 1/5] END ...................C=1000, gamma=1;, score=0.950 total time=   0.0s
    [CV 2/5] END ...................C=1000, gamma=1;, score=0.800 total time=   0.0s
    [CV 3/5] END ...................C=1000, gamma=1;, score=0.900 total time=   0.0s
    [CV 4/5] END ...................C=1000, gamma=1;, score=1.000 total time=   0.0s
    [CV 5/5] END ...................C=1000, gamma=1;, score=0.900 total time=   0.0s
    [CV 1/5] END .................C=1000, gamma=0.1;, score=0.950 total time=   0.0s
    [CV 2/5] END .................C=1000, gamma=0.1;, score=0.800 total time=   0.0s
    [CV 3/5] END .................C=1000, gamma=0.1;, score=0.900 total time=   0.0s
    [CV 4/5] END .................C=1000, gamma=0.1;, score=1.000 total time=   0.0s
    [CV 5/5] END .................C=1000, gamma=0.1;, score=0.950 total time=   0.0s
    [CV 1/5] END ................C=1000, gamma=0.01;, score=0.950 total time=   0.0s
    [CV 2/5] END ................C=1000, gamma=0.01;, score=0.800 total time=   0.0s
    [CV 3/5] END ................C=1000, gamma=0.01;, score=0.900 total time=   0.0s
    [CV 4/5] END ................C=1000, gamma=0.01;, score=0.950 total time=   0.0s
    [CV 5/5] END ................C=1000, gamma=0.01;, score=1.000 total time=   0.0s
    [CV 1/5] END ...............C=1000, gamma=0.001;, score=0.950 total time=   0.0s
    [CV 2/5] END ...............C=1000, gamma=0.001;, score=0.950 total time=   0.0s
    [CV 3/5] END ...............C=1000, gamma=0.001;, score=0.900 total time=   0.0s
    [CV 4/5] END ...............C=1000, gamma=0.001;, score=1.000 total time=   0.0s
    [CV 5/5] END ...............C=1000, gamma=0.001;, score=0.950 total time=   0.0s
    [CV 1/5] END ..............C=1000, gamma=0.0001;, score=0.950 total time=   0.0s
    [CV 2/5] END ..............C=1000, gamma=0.0001;, score=0.900 total time=   0.0s
    [CV 3/5] END ..............C=1000, gamma=0.0001;, score=0.900 total time=   0.0s
    [CV 4/5] END ..............C=1000, gamma=0.0001;, score=1.000 total time=   0.0s
    [CV 5/5] END ..............C=1000, gamma=0.0001;, score=0.950 total time=   0.0s
    




    GridSearchCV(estimator=SVC(),
                 param_grid={'C': [0.01, 0.1, 1, 10, 100, 1000],
                             'gamma': [1, 0.1, 0.01, 0.001, 0.0001]},
                 verbose=3)




```python

```

    Fitting 3 folds for each of 16 candidates, totalling 48 fits
    [CV] gamma=1, C=0.1 ..................................................
    [CV] ......................................... gamma=1, C=0.1 -   0.0s
    [CV] gamma=1, C=0.1 ..................................................
    [CV] ......................................... gamma=1, C=0.1 -   0.0s
    [CV] gamma=1, C=0.1 ..................................................
    [CV] ......................................... gamma=1, C=0.1 -   0.0s
    [CV] gamma=0.1, C=0.1 ................................................
    [CV] ....................................... gamma=0.1, C=0.1 -   0.0s
    [CV] gamma=0.1, C=0.1 ................................................
    [CV] ....................................... gamma=0.1, C=0.1 -   0.0s
    [CV] gamma=0.1, C=0.1 ................................................
    [CV] ....................................... gamma=0.1, C=0.1 -   0.0s
    [CV] gamma=0.01, C=0.1 ...............................................
    [CV] ...................................... gamma=0.01, C=0.1 -   0.0s
    [CV] gamma=0.01, C=0.1 ...............................................
    [CV] ...................................... gamma=0.01, C=0.1 -   0.0s
    [CV] gamma=0.01, C=0.1 ...............................................
    [CV] ...................................... gamma=0.01, C=0.1 -   0.0s
    [CV] gamma=0.001, C=0.1 ..............................................
    [CV] ..................................... gamma=0.001, C=0.1 -   0.0s
    [CV] gamma=0.001, C=0.1 ..............................................
    [CV] ..................................... gamma=0.001, C=0.1 -   0.0s
    [CV] gamma=0.001, C=0.1 ..............................................
    [CV] ..................................... gamma=0.001, C=0.1 -   0.0s
    [CV] gamma=1, C=1 ....................................................
    [CV] ........................................... gamma=1, C=1 -   0.0s
    [CV] gamma=1, C=1 ....................................................
    [CV] ........................................... gamma=1, C=1 -   0.0s
    [CV] gamma=1, C=1 ....................................................
    [CV] ........................................... gamma=1, C=1 -   0.0s
    [CV] gamma=0.1, C=1 ..................................................
    [CV] ......................................... gamma=0.1, C=1 -   0.0s
    [CV] gamma=0.1, C=1 ..................................................
    [CV] ......................................... gamma=0.1, C=1 -   0.0s
    [CV] gamma=0.1, C=1 ..................................................
    [CV] ......................................... gamma=0.1, C=1 -   0.0s
    [CV] gamma=0.01, C=1 .................................................
    [CV] ........................................ gamma=0.01, C=1 -   0.0s
    [CV] gamma=0.01, C=1 .................................................
    [CV] ........................................ gamma=0.01, C=1 -   0.0s
    [CV] gamma=0.01, C=1 .................................................
    [CV] ........................................ gamma=0.01, C=1 -   0.0s
    [CV] gamma=0.001, C=1 ................................................
    [CV] ....................................... gamma=0.001, C=1 -   0.0s
    [CV] gamma=0.001, C=1 ................................................
    [CV] ....................................... gamma=0.001, C=1 -   0.0s
    [CV] gamma=0.001, C=1 ................................................
    [CV] ....................................... gamma=0.001, C=1 -   0.0s
    [CV] gamma=1, C=10 ...................................................
    [CV] .......................................... gamma=1, C=10 -   0.0s
    [CV] gamma=1, C=10 ...................................................
    [CV] .......................................... gamma=1, C=10 -   0.0s
    [CV] gamma=1, C=10 ...................................................
    [CV] .......................................... gamma=1, C=10 -   0.0s
    [CV] gamma=0.1, C=10 .................................................
    [CV] ........................................ gamma=0.1, C=10 -   0.0s
    [CV] gamma=0.1, C=10 .................................................
    [CV] ........................................ gamma=0.1, C=10 -   0.0s
    [CV] gamma=0.1, C=10 .................................................
    [CV] ........................................ gamma=0.1, C=10 -   0.0s
    [CV] gamma=0.01, C=10 ................................................
    [CV] ....................................... gamma=0.01, C=10 -   0.0s
    [CV] gamma=0.01, C=10 ................................................
    [CV] ....................................... gamma=0.01, C=10 -   0.0s
    [CV] gamma=0.01, C=10 ................................................
    [CV] ....................................... gamma=0.01, C=10 -   0.0s
    [CV] gamma=0.001, C=10 ...............................................
    [CV] ...................................... gamma=0.001, C=10 -   0.0s
    [CV] gamma=0.001, C=10 ...............................................
    [CV] ...................................... gamma=0.001, C=10 -   0.0s
    [CV] gamma=0.001, C=10 ...............................................
    [CV] ...................................... gamma=0.001, C=10 -   0.0s
    [CV] gamma=1, C=100 ..................................................
    [CV] ......................................... gamma=1, C=100 -   0.0s
    [CV] gamma=1, C=100 ..................................................
    [CV] ......................................... gamma=1, C=100 -   0.0s
    [CV] gamma=1, C=100 ..................................................
    [CV] ......................................... gamma=1, C=100 -   0.0s
    [CV] gamma=0.1, C=100 ................................................
    [CV] ....................................... gamma=0.1, C=100 -   0.0s
    [CV] gamma=0.1, C=100 ................................................
    [CV] ....................................... gamma=0.1, C=100 -   0.0s
    [CV] gamma=0.1, C=100 ................................................
    [CV] ....................................... gamma=0.1, C=100 -   0.0s
    [CV] gamma=0.01, C=100 ...............................................
    [CV] ...................................... gamma=0.01, C=100 -   0.0s
    [CV] gamma=0.01, C=100 ...............................................
    [CV] ...................................... gamma=0.01, C=100 -   0.0s
    [CV] gamma=0.01, C=100 ...............................................
    [CV] ...................................... gamma=0.01, C=100 -   0.0s
    [CV] gamma=0.001, C=100 ..............................................
    [CV] ..................................... gamma=0.001, C=100 -   0.0s
    [CV] gamma=0.001, C=100 ..............................................
    [CV] ..................................... gamma=0.001, C=100 -   0.0s
    [CV] gamma=0.001, C=100 ..............................................
    [CV] ..................................... gamma=0.001, C=100 -   0.0s
    

    [Parallel(n_jobs=1)]: Done  40 tasks       | elapsed:    0.2s
    [Parallel(n_jobs=1)]: Done  48 out of  48 | elapsed:    0.2s finished
    




    GridSearchCV(cv=None, error_score='raise',
           estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'gamma': [1, 0.1, 0.01, 0.001], 'C': [0.1, 1, 10, 100]},
           pre_dispatch='2*n_jobs', refit=True, scoring=None, verbose=2)



** Now take that grid model and create some predictions using the test set and create classification reports and confusion matrices for them. Were you able to improve?**


```python
grid.best_params_
```




    {'C': 100, 'gamma': 0.01}




```python
grid.best_estimator_
```




    SVC(C=100, gamma=0.01)




```python
grid_predictions = grid.predict(X_test)
```


```python
print(confusion_matrix(y_test, grid_predictions))
print('\n')
print(classification_report(y_test, grid_predictions))
```

    [[19  0  0]
     [ 0 15  0]
     [ 0  0 16]]
    
    
                  precision    recall  f1-score   support
    
          setosa       1.00      1.00      1.00        19
      versicolor       1.00      1.00      1.00        15
       virginica       1.00      1.00      1.00        16
    
        accuracy                           1.00        50
       macro avg       1.00      1.00      1.00        50
    weighted avg       1.00      1.00      1.00        50
    
    


```python

```

    [[15  0  0]
     [ 0 13  1]
     [ 0  0 16]]
    


```python

```

                 precision    recall  f1-score   support
    
         setosa       1.00      1.00      1.00        15
     versicolor       1.00      0.93      0.96        14
      virginica       0.94      1.00      0.97        16
    
    avg / total       0.98      0.98      0.98        45
    
    

You should have done about the same or exactly the same, this makes sense, there is basically just one point that is too noisey to grab, which makes sense, we don't want to have an overfit model that would be able to grab that.

## Great Job!
