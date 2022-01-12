```python
import numpy as np
import pandas as pd
```


```python
df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})
df.head()
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>444</td>
      <td>abc</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>555</td>
      <td>def</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>666</td>
      <td>ghi</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>444</td>
      <td>xyz</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['col2'].unique() # distinct
```




    array([444, 555, 666], dtype=int64)




```python
len(df['col2'].unique())
```




    3




```python
df['col2'].value_counts()
```




    444    2
    555    1
    666    1
    Name: col2, dtype: int64




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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>444</td>
      <td>abc</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>555</td>
      <td>def</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>666</td>
      <td>ghi</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>444</td>
      <td>xyz</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['col1']>2]
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>666</td>
      <td>ghi</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>444</td>
      <td>xyz</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[(df['col1']>2) & (df['col2']==444)]
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>444</td>
      <td>xyz</td>
    </tr>
  </tbody>
</table>
</div>




```python
def times2(x):
    return x*2
```


```python
df['col1'].sum()
```




    10




```python
df['col1'].apply(times2)
```




    0    2
    1    4
    2    6
    3    8
    Name: col1, dtype: int64




```python
df['col3'].apply(len)
```




    0    3
    1    3
    2    3
    3    3
    Name: col3, dtype: int64




```python
df['col2'].apply(lambda x: x*2)
```




    0     888
    1    1110
    2    1332
    3     888
    Name: col2, dtype: int64




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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>444</td>
      <td>abc</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>555</td>
      <td>def</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>666</td>
      <td>ghi</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>444</td>
      <td>xyz</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop('col1', axis=1, inplace=True)
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
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>444</td>
      <td>abc</td>
    </tr>
    <tr>
      <th>1</th>
      <td>555</td>
      <td>def</td>
    </tr>
    <tr>
      <th>2</th>
      <td>666</td>
      <td>ghi</td>
    </tr>
    <tr>
      <th>3</th>
      <td>444</td>
      <td>xyz</td>
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>444</td>
      <td>abc</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>555</td>
      <td>def</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>666</td>
      <td>ghi</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>444</td>
      <td>xyz</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```




    Index(['col1', 'col2', 'col3'], dtype='object')




```python
df.index
```




    RangeIndex(start=0, stop=4, step=1)




```python
df.sort_values('col2')
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>444</td>
      <td>abc</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>444</td>
      <td>xyz</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>555</td>
      <td>def</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>666</td>
      <td>ghi</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isnull()
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = {'A':['foo','foo','foo','bar','bar','bar'],
     'B':['one','one','two','two','one','one'],
       'C':['x','y','x','y','x','y'],
       'D':[1,3,2,5,4,1]}

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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>foo</td>
      <td>one</td>
      <td>x</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>foo</td>
      <td>one</td>
      <td>y</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>foo</td>
      <td>two</td>
      <td>x</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bar</td>
      <td>two</td>
      <td>y</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bar</td>
      <td>one</td>
      <td>x</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>bar</td>
      <td>one</td>
      <td>y</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.pivot_table(values='D', index=['A', 'B'], columns=['C'])
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
      <th>C</th>
      <th>x</th>
      <th>y</th>
    </tr>
    <tr>
      <th>A</th>
      <th>B</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">bar</th>
      <th>one</th>
      <td>4.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>two</th>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">foo</th>
      <th>one</th>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
