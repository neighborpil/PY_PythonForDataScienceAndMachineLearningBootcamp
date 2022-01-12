```python
pwd
```




    'C:\\Work\\GitRepository\\PY_PythonForDataScienceAndMachineLearningBootcamp\\Py_DS_ML_Bootcamp-master\\03-Python-for-Data-Analysis-Pandas'




```python
import pandas as pd
```


```python
pd.read_csv('example.csv')
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.read_csv('example.csv')
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.to_csv('My_output.csv', index=False)
```


```python
pd.read_csv('My_output.csv')
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.read_excel('Excel_Sample.xlsx', sheet_name='Sheet1')
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
      <th>Unnamed: 0</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.to_excel('Excel_Sample2.xlsx', sheet_name='New Sheet')
```


```python
pd.read_excel('Excel_Sample2.xlsx', sheet_name='New Sheet')
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
      <th>Unnamed: 0</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = pd.read_html('https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/')
```


```python
type(data)
```




    list




```python
data
```




    [                         Bank NameBank           CityCity StateSt  CertCert  \
     0                    Almena State Bank             Almena      KS     15426   
     1           First City Bank of Florida  Fort Walton Beach      FL     16748   
     2                 The First State Bank      Barboursville      WV     14361   
     3                   Ericson State Bank            Ericson      NE     18265   
     4     City National Bank of New Jersey             Newark      NJ     21111   
     ..                                 ...                ...     ...       ...   
     558                 Superior Bank, FSB           Hinsdale      IL     32646   
     559                Malta National Bank              Malta      OH      6629   
     560    First Alliance Bank & Trust Co.         Manchester      NH     34264   
     561  National State Bank of Metropolis         Metropolis      IL      3815   
     562                   Bank of Honolulu           Honolulu      HI     21029   
     
                      Acquiring InstitutionAI Closing DateClosing  FundFund  
     0                            Equity Bank    October 23, 2020     10538  
     1              United Fidelity Bank, fsb    October 16, 2020     10537  
     2                         MVB Bank, Inc.       April 3, 2020     10536  
     3             Farmers and Merchants Bank   February 14, 2020     10535  
     4                        Industrial Bank    November 1, 2019     10534  
     ..                                   ...                 ...       ...  
     558                Superior Federal, FSB       July 27, 2001      6004  
     559                    North Valley Bank         May 3, 2001      4648  
     560  Southern New Hampshire Bank & Trust    February 2, 2001      4647  
     561              Banterra Bank of Marion   December 14, 2000      4646  
     562                   Bank of the Orient    October 13, 2000      4645  
     
     [563 rows x 7 columns]]




```python
data[0]
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
      <th>Bank NameBank</th>
      <th>CityCity</th>
      <th>StateSt</th>
      <th>CertCert</th>
      <th>Acquiring InstitutionAI</th>
      <th>Closing DateClosing</th>
      <th>FundFund</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Almena State Bank</td>
      <td>Almena</td>
      <td>KS</td>
      <td>15426</td>
      <td>Equity Bank</td>
      <td>October 23, 2020</td>
      <td>10538</td>
    </tr>
    <tr>
      <th>1</th>
      <td>First City Bank of Florida</td>
      <td>Fort Walton Beach</td>
      <td>FL</td>
      <td>16748</td>
      <td>United Fidelity Bank, fsb</td>
      <td>October 16, 2020</td>
      <td>10537</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The First State Bank</td>
      <td>Barboursville</td>
      <td>WV</td>
      <td>14361</td>
      <td>MVB Bank, Inc.</td>
      <td>April 3, 2020</td>
      <td>10536</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ericson State Bank</td>
      <td>Ericson</td>
      <td>NE</td>
      <td>18265</td>
      <td>Farmers and Merchants Bank</td>
      <td>February 14, 2020</td>
      <td>10535</td>
    </tr>
    <tr>
      <th>4</th>
      <td>City National Bank of New Jersey</td>
      <td>Newark</td>
      <td>NJ</td>
      <td>21111</td>
      <td>Industrial Bank</td>
      <td>November 1, 2019</td>
      <td>10534</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>558</th>
      <td>Superior Bank, FSB</td>
      <td>Hinsdale</td>
      <td>IL</td>
      <td>32646</td>
      <td>Superior Federal, FSB</td>
      <td>July 27, 2001</td>
      <td>6004</td>
    </tr>
    <tr>
      <th>559</th>
      <td>Malta National Bank</td>
      <td>Malta</td>
      <td>OH</td>
      <td>6629</td>
      <td>North Valley Bank</td>
      <td>May 3, 2001</td>
      <td>4648</td>
    </tr>
    <tr>
      <th>560</th>
      <td>First Alliance Bank &amp; Trust Co.</td>
      <td>Manchester</td>
      <td>NH</td>
      <td>34264</td>
      <td>Southern New Hampshire Bank &amp; Trust</td>
      <td>February 2, 2001</td>
      <td>4647</td>
    </tr>
    <tr>
      <th>561</th>
      <td>National State Bank of Metropolis</td>
      <td>Metropolis</td>
      <td>IL</td>
      <td>3815</td>
      <td>Banterra Bank of Marion</td>
      <td>December 14, 2000</td>
      <td>4646</td>
    </tr>
    <tr>
      <th>562</th>
      <td>Bank of Honolulu</td>
      <td>Honolulu</td>
      <td>HI</td>
      <td>21029</td>
      <td>Bank of the Orient</td>
      <td>October 13, 2000</td>
      <td>4645</td>
    </tr>
  </tbody>
</table>
<p>563 rows × 7 columns</p>
</div>




```python
data[0].head()
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
      <th>Bank NameBank</th>
      <th>CityCity</th>
      <th>StateSt</th>
      <th>CertCert</th>
      <th>Acquiring InstitutionAI</th>
      <th>Closing DateClosing</th>
      <th>FundFund</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Almena State Bank</td>
      <td>Almena</td>
      <td>KS</td>
      <td>15426</td>
      <td>Equity Bank</td>
      <td>October 23, 2020</td>
      <td>10538</td>
    </tr>
    <tr>
      <th>1</th>
      <td>First City Bank of Florida</td>
      <td>Fort Walton Beach</td>
      <td>FL</td>
      <td>16748</td>
      <td>United Fidelity Bank, fsb</td>
      <td>October 16, 2020</td>
      <td>10537</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The First State Bank</td>
      <td>Barboursville</td>
      <td>WV</td>
      <td>14361</td>
      <td>MVB Bank, Inc.</td>
      <td>April 3, 2020</td>
      <td>10536</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ericson State Bank</td>
      <td>Ericson</td>
      <td>NE</td>
      <td>18265</td>
      <td>Farmers and Merchants Bank</td>
      <td>February 14, 2020</td>
      <td>10535</td>
    </tr>
    <tr>
      <th>4</th>
      <td>City National Bank of New Jersey</td>
      <td>Newark</td>
      <td>NJ</td>
      <td>21111</td>
      <td>Industrial Bank</td>
      <td>November 1, 2019</td>
      <td>10534</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sqlalchemy import create_engine
```


```python
engine = create_engine('sqlite:///:memory:')
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.to_sql('my_table', engine)
```


```python
sqldf = pd.read_sql('my_table', con=engine)
```


```python
sqldf
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
      <th>index</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
