```python
import numpy as np
```


```python
import pandas as pd
```


```python
labels = ['a','b','c']
my_data = [10, 20, 30]
arr = np.array(my_data)
d = {'a': 10, 'b':20, 'c':30}
```


```python
arr
```




    array([10, 20, 30])




```python
labels
```




    ['a', 'b', 'c']




```python
my_data
```




    [10, 20, 30]




```python
d
```




    {'a': 10, 'b': 20, 'c': 30}




```python
pd.Series(data= my_data)
```




    0    10
    1    20
    2    30
    dtype: int64




```python
pd.Series(data=my_data, index=labels)
```




    a    10
    b    20
    c    30
    dtype: int64




```python
pd.Series(my_data, labels)
```




    a    10
    b    20
    c    30
    dtype: int64




```python
pd.Series(arr)
```




    0    10
    1    20
    2    30
    dtype: int32




```python
pd.Series(arr, labels)
```




    a    10
    b    20
    c    30
    dtype: int32




```python
pd.Series(d)
```




    a    10
    b    20
    c    30
    dtype: int64




```python
pd.Series(data=labels)
```




    0    a
    1    b
    2    c
    dtype: object




```python
pd.Series(data=[sum, print, len])
```




    0      <built-in function sum>
    1    <built-in function print>
    2      <built-in function len>
    dtype: object




```python
ser1 = pd.Series([1,2,3,4],['USA','Germany','USSR','Korea'])
```


```python
ser1
```




    USA        1
    Germany    2
    USSR       3
    Korea      4
    dtype: int64




```python
ser2 = pd.Series([1,2,5,4],['USA','Germany','Italy','Korea'])
```


```python
ser2
```




    USA        1
    Germany    2
    Italy      5
    Korea      4
    dtype: int64




```python
ser1['USA']
```




    1




```python
ser3 = pd.Series(data=labels)
```


```python
ser3
```




    0    a
    1    b
    2    c
    dtype: object




```python
ser3[0]
```




    'a'




```python
ser1
```




    USA        1
    Germany    2
    USSR       3
    Korea      4
    dtype: int64




```python
ser2
```




    USA        1
    Germany    2
    Italy      5
    Korea      4
    dtype: int64




```python
ser1 + ser2
```




    Germany    4.0
    Italy      NaN
    Korea      8.0
    USA        2.0
    USSR       NaN
    dtype: float64




```python

```
