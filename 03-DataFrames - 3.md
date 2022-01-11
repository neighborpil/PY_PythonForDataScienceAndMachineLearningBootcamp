```python
import numpy as np
import pandas as pd
from numpy.random import randn
```


```python
outside = ['G1', 'G1', 'G1', 'G2', 'G2', 'G2']
inside = [1,2,3,1,2,3,]
hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)
```


```python
outside
```




    ['G1', 'G1', 'G1', 'G2', 'G2', 'G2']




```python
inside
```




    [1, 2, 3, 1, 2, 3]




```python
list(zip(outside,inside))
```




    [('G1', 1), ('G1', 2), ('G1', 3), ('G2', 1), ('G2', 2), ('G2', 3)]




```python
hier_index
```




    MultiIndex([('G1', 1),
                ('G1', 2),
                ('G1', 3),
                ('G2', 1),
                ('G2', 2),
                ('G2', 3)],
               )




```python
df = pd.DataFrame(randn(6,2), hier_index, ['A','B'])
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
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">G1</th>
      <th>1</th>
      <td>-1.468706</td>
      <td>-2.001623</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.793780</td>
      <td>0.688697</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.097501</td>
      <td>-1.222164</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">G2</th>
      <th>1</th>
      <td>0.709204</td>
      <td>0.437456</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.493907</td>
      <td>-0.216717</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.259546</td>
      <td>-1.032613</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc['G1']
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-1.468706</td>
      <td>-2.001623</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.793780</td>
      <td>0.688697</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.097501</td>
      <td>-1.222164</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc['G1'].loc[1]
```




    A   -1.468706
    B   -2.001623
    Name: 1, dtype: float64




```python
df.index.names
```




    FrozenList([None, None])




```python
df.index.names = ['Groups', 'Num']
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
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>Groups</th>
      <th>Num</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">G1</th>
      <th>1</th>
      <td>-1.468706</td>
      <td>-2.001623</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.793780</td>
      <td>0.688697</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.097501</td>
      <td>-1.222164</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">G2</th>
      <th>1</th>
      <td>0.709204</td>
      <td>0.437456</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.493907</td>
      <td>-0.216717</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.259546</td>
      <td>-1.032613</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc['G2'].loc[2]['B']
```




    -0.216716778762725




```python
df.loc['G1']
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
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>Num</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-1.468706</td>
      <td>-2.001623</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.793780</td>
      <td>0.688697</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.097501</td>
      <td>-1.222164</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.xs('G1')
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
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>Num</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-1.468706</td>
      <td>-2.001623</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.793780</td>
      <td>0.688697</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.097501</td>
      <td>-1.222164</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.xs(1, level='Num')
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
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>Groups</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>G1</th>
      <td>-1.468706</td>
      <td>-2.001623</td>
    </tr>
    <tr>
      <th>G2</th>
      <td>0.709204</td>
      <td>0.437456</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
