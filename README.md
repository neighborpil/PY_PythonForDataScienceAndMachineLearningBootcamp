# PY_PythonForDataScienceAndMachineLearningBootcamp
example codes


## Installing venv
```
> conda create --name venvtest numpy
> conda activate venvtest
(venvtest) conda install pandas # installing additional libraries
> conda deactivate

# show virtual evnviroment list
> conda env list

# create virtual evnviroment with python version 3.5 and 
> condal create --name python35withnumpy python=3.5 numpy
```

## Starting Jupyter Notebook
```
> cd C:\Work\GitRepository\PY_PythonForDataScienceAndMachineLearningBootcamp
> jupyter notebook
```

# Python basics
```
for x in range(0,5):
    print(x);
    
list(range(0,5))


list(range(10))

for num in x:
    out.append(num**2)
print(out)

[num**2 for num in x]

def my_func(name='Default Name'):
    print('Hello ' + name)
    
my_func(name='joy')
my_func()

def square(num):
    """
    독타입 정의하면 다른데서 볼 수 있다
    This is a docstring
    can go multiple lines
    """
    return num**2

def times2(var):
    return var*2
    
seq = list(range(1,6))
 
list(map(times2, seq))

list(map(lambda num: num*2, seq))

list(filter(lambda num: num%2 == 0, seq))

s.lower()

s.upper()

s.split()

tweet = 'Go Sports! #Sports'
tweet.split('#')

d = {'k1': 1, 'k2':2}
d.keys()
d.items()
d.values()

lst = [1,2,3]
item = lst.pop()
lst.pop(0)

'x' in [1,2,3]
'x' in ['x','y','z']

x = [(1,2),(3,4),(5,6)]
for item in x:
    print(item)
    
for a,b in x:
    print(a)
    print(b)
    

```

## Numpy
```
import numpy as np
my_list = [1,2,3]
arr = np.array(my_list)

np.arange(0,10)

np.arange(0,10,2)

np.zeros(3)
# array([0., 0., 0.])

np.zeros((5,5))

# array([[0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.]])

np.zeros((2, 3))
# array([[0., 0., 0.],
       [0., 0., 0.]])

np.ones((3,4))
# array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]])

np.linspace(0, 5, 10)
# array([0.        , 0.55555556, 1.11111111, 1.66666667, 2.22222222,
       2.77777778, 3.33333333, 3.88888889, 4.44444444, 5.        ])
       
np.eye(4)
# array([[1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.]])
       
np.random.rand(5)
# array([0.65324063, 0.97146608, 0.5783551 , 0.29142837, 0.59190609])

np.random.rand(5,5)
# array([[0.32527526, 0.43247911, 0.93320166, 0.45385394, 0.6500155 ],
       [0.81880138, 0.77408162, 0.50863177, 0.0569942 , 0.97081251],
       [0.727519  , 0.85341521, 0.04767051, 0.21483402, 0.80151898],
       [0.84347208, 0.7102733 , 0.44963776, 0.56807216, 0.89593188],
       [0.05739072, 0.11054024, 0.6482786 , 0.46502734, 0.05662895]])
       
np.random.randn(2)
# array([ 0.32629206, -1.08274094])

np.random.randn(4,4)
# array([[ 0.88691527,  0.89580122,  0.80593658, -0.53526618],
       [-0.94725438, -0.22023542, -0.68925382,  0.06029622],
       [ 0.88640697, -0.24179796,  0.72470006,  0.71540645],
       [ 0.68034124, -0.43906192,  0.12198041, -1.5851285 ]])
       
np.random.randint(1, 100)
# 75

np.random.randint(1, 100, 10)
# array([13, 91,  1,  4, 50, 74, 42, 54, 69,  2])

randarr = np.random.randint(0, 50, 10)
# array([ 7, 27, 23, 45, 11, 36, 38, 14, 42, 10])

arr = np.arange(25)
# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24])
       
arr.reshape(5,5)
# array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24]])



randarr.max()

randarr.min()

randarr.argmin() # show index of min value

randarr.argmax()

arr
# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24])
    
arr = arr.reshape(5,5)
arr.shape
# (5, 5)



# show datatype
arr.dtype
# arr.dtype


```

### numpy array 복사
1. 참조복사
 - sub array를 만들어 값을 바꾸면 기존의 array의 값도 바뀐다
```
arr = np.arange(0, 11)
slice_of_array = arr[0:6]
```
2. 깊은 복사
 - .copy()로 명시를 해줘야 한다
```
arr = np.arange(0, 11)
deep_copied = arr.copy()
```

### numpy array indexing and selection examples
```
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


arr = np.arange(0, 11)


# In[3]:


arr


# In[4]:


arr[8]


# In[5]:


arr[1:5]


# In[6]:


arr[0:5]


# In[7]:


arr[0:6]


# In[8]:


arr[5:]


# In[10]:


arr[0:5] = 100


# In[11]:


arr


# In[13]:


arr = np.arange(0, 11)


# In[14]:


arr


# In[15]:


slice_of_arr = arr[0:6]


# In[16]:


slice_of_arr


# In[19]:


slice_of_arr[:] = 99


# In[20]:


slice_of_arr


# In[22]:


arr


# In[23]:


arr_copy = arr.copy()


# In[24]:


arr_copy


# In[25]:


arr_copy[:] = 100


# In[26]:


arr_copy


# In[27]:


arr


# In[28]:


arr_2d = np.array([[5,10,15],[20,25,30],[35,40,45]])


# In[29]:


arr_2d


# In[30]:


arr_2d[0][0]


# In[31]:


arr_2d[0]


# In[32]:


arr_2d[1]


# In[33]:


arr_2d[1][1]


# In[34]:


arr_2d[2,1]


# In[35]:


arr_2d[:2,1:]


# In[36]:


arr_2d[:2]

```

#### numpy divide by 0
 - 그냥 파이썬 array는 에러 띄운다
 - numpy는 nan or inf으로 표시하고 에러는 띄우지 않는다
 - 대신 경고는 띄운다

#### numpy universal functions
 - https://docs.scipy.org/doc/numpy/reference/ufuncs.html

#### numpy operations
```
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


arr = np.arange(0, 11)


# In[3]:


arr


# In[4]:


arr + arr


# In[5]:


arr


# In[6]:


arr - arr


# In[7]:


arr * arr


# In[8]:


arr + 100


# In[9]:


arr * 100


# In[10]:


arr -100


# In[11]:


1 / 0


# In[12]:


arr /arr


# In[13]:


1 / arr


# In[14]:


arr ** 2


# In[15]:


np.sqrt(arr)


# In[16]:


np.exp(arr)


# In[17]:


np.max(arr)


# In[18]:


arr.max()


# In[19]:


np.sin(arr)


# In[20]:


np.log(arr)
```

