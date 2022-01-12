```python
import numpy as np
import pandas as pd
from numpy.random import rand
```


```python
# Create dataframe
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}
```


```python
df = pd.DataFrame(data)
```


```python
df
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
      <th>Company</th>
      <th>Person</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GOOG</td>
      <td>Sam</td>
      <td>200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GOOG</td>
      <td>Charlie</td>
      <td>120</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MSFT</td>
      <td>Amy</td>
      <td>340</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MSFT</td>
      <td>Vanessa</td>
      <td>124</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FB</td>
      <td>Carl</td>
      <td>243</td>
    </tr>
    <tr>
      <th>5</th>
      <td>FB</td>
      <td>Sarah</td>
      <td>350</td>
    </tr>
  </tbody>
</table>
</div>




```python
byComp = df.groupby('Company')
```


```python
byComp.mean()
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
      <th>Sales</th>
    </tr>
    <tr>
      <th>Company</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FB</th>
      <td>296.5</td>
    </tr>
    <tr>
      <th>GOOG</th>
      <td>160.0</td>
    </tr>
    <tr>
      <th>MSFT</th>
      <td>232.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
byComp.sum()
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
      <th>Sales</th>
    </tr>
    <tr>
      <th>Company</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FB</th>
      <td>593</td>
    </tr>
    <tr>
      <th>GOOG</th>
      <td>320</td>
    </tr>
    <tr>
      <th>MSFT</th>
      <td>464</td>
    </tr>
  </tbody>
</table>
</div>




```python
byComp.std()
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
      <th>Sales</th>
    </tr>
    <tr>
      <th>Company</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FB</th>
      <td>75.660426</td>
    </tr>
    <tr>
      <th>GOOG</th>
      <td>56.568542</td>
    </tr>
    <tr>
      <th>MSFT</th>
      <td>152.735065</td>
    </tr>
  </tbody>
</table>
</div>




```python
byComp.sum().loc['FB']
```




    Sales    593
    Name: FB, dtype: int64




```python
df.groupby('Company').sum().loc['FB']
```




    Sales    593
    Name: FB, dtype: int64




```python
df.groupby('Company').count()
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
      <th>Person</th>
      <th>Sales</th>
    </tr>
    <tr>
      <th>Company</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FB</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>GOOG</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>MSFT</th>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df
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
      <th>Company</th>
      <th>Person</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GOOG</td>
      <td>Sam</td>
      <td>200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GOOG</td>
      <td>Charlie</td>
      <td>120</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MSFT</td>
      <td>Amy</td>
      <td>340</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MSFT</td>
      <td>Vanessa</td>
      <td>124</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FB</td>
      <td>Carl</td>
      <td>243</td>
    </tr>
    <tr>
      <th>5</th>
      <td>FB</td>
      <td>Sarah</td>
      <td>350</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby('Company').max()
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
      <th>Person</th>
      <th>Sales</th>
    </tr>
    <tr>
      <th>Company</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FB</th>
      <td>Sarah</td>
      <td>350</td>
    </tr>
    <tr>
      <th>GOOG</th>
      <td>Sam</td>
      <td>200</td>
    </tr>
    <tr>
      <th>MSFT</th>
      <td>Vanessa</td>
      <td>340</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby('Company').min()
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
      <th>Person</th>
      <th>Sales</th>
    </tr>
    <tr>
      <th>Company</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FB</th>
      <td>Carl</td>
      <td>243</td>
    </tr>
    <tr>
      <th>GOOG</th>
      <td>Charlie</td>
      <td>120</td>
    </tr>
    <tr>
      <th>MSFT</th>
      <td>Amy</td>
      <td>124</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby('Company').describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">Sales</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Company</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FB</th>
      <td>2.0</td>
      <td>296.5</td>
      <td>75.660426</td>
      <td>243.0</td>
      <td>269.75</td>
      <td>296.5</td>
      <td>323.25</td>
      <td>350.0</td>
    </tr>
    <tr>
      <th>GOOG</th>
      <td>2.0</td>
      <td>160.0</td>
      <td>56.568542</td>
      <td>120.0</td>
      <td>140.00</td>
      <td>160.0</td>
      <td>180.00</td>
      <td>200.0</td>
    </tr>
    <tr>
      <th>MSFT</th>
      <td>2.0</td>
      <td>232.0</td>
      <td>152.735065</td>
      <td>124.0</td>
      <td>178.00</td>
      <td>232.0</td>
      <td>286.00</td>
      <td>340.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby('Company').describe().transpose()
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
      <th>Company</th>
      <th>FB</th>
      <th>GOOG</th>
      <th>MSFT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="8" valign="top">Sales</th>
      <th>count</th>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>296.500000</td>
      <td>160.000000</td>
      <td>232.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>75.660426</td>
      <td>56.568542</td>
      <td>152.735065</td>
    </tr>
    <tr>
      <th>min</th>
      <td>243.000000</td>
      <td>120.000000</td>
      <td>124.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>269.750000</td>
      <td>140.000000</td>
      <td>178.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>296.500000</td>
      <td>160.000000</td>
      <td>232.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>323.250000</td>
      <td>180.000000</td>
      <td>286.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>350.000000</td>
      <td>200.000000</td>
      <td>340.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
