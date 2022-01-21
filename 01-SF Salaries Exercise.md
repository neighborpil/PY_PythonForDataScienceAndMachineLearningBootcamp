___

<a href='http://www.pieriandata.com'> <img src='../../Pierian_Data_Logo.png' /></a>
___

# SF Salaries Exercise 

Welcome to a quick exercise for you to practice your pandas skills! We will be using the [SF Salaries Dataset](https://www.kaggle.com/kaggle/sf-salaries) from Kaggle! Just follow along and complete the tasks outlined in bold below. The tasks will get harder and harder as you go along.

** Import pandas as pd.**


```python
import pandas as pd
```

** Read Salaries.csv as a dataframe called sal.**


```python
df = pd.read_csv("Salaries.csv")
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
      <th>Id</th>
      <th>EmployeeName</th>
      <th>JobTitle</th>
      <th>BasePay</th>
      <th>OvertimePay</th>
      <th>OtherPay</th>
      <th>Benefits</th>
      <th>TotalPay</th>
      <th>TotalPayBenefits</th>
      <th>Year</th>
      <th>Notes</th>
      <th>Agency</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NATHANIEL FORD</td>
      <td>GENERAL MANAGER-METROPOLITAN TRANSIT AUTHORITY</td>
      <td>167411.18</td>
      <td>0.00</td>
      <td>400184.25</td>
      <td>NaN</td>
      <td>567595.43</td>
      <td>567595.43</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>GARY JIMENEZ</td>
      <td>CAPTAIN III (POLICE DEPARTMENT)</td>
      <td>155966.02</td>
      <td>245131.88</td>
      <td>137811.38</td>
      <td>NaN</td>
      <td>538909.28</td>
      <td>538909.28</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>ALBERT PARDINI</td>
      <td>CAPTAIN III (POLICE DEPARTMENT)</td>
      <td>212739.13</td>
      <td>106088.18</td>
      <td>16452.60</td>
      <td>NaN</td>
      <td>335279.91</td>
      <td>335279.91</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>CHRISTOPHER CHONG</td>
      <td>WIRE ROPE CABLE MAINTENANCE MECHANIC</td>
      <td>77916.00</td>
      <td>56120.71</td>
      <td>198306.90</td>
      <td>NaN</td>
      <td>332343.61</td>
      <td>332343.61</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>PATRICK GARDNER</td>
      <td>DEPUTY CHIEF OF DEPARTMENT,(FIRE DEPARTMENT)</td>
      <td>134401.60</td>
      <td>9737.00</td>
      <td>182234.59</td>
      <td>NaN</td>
      <td>326373.19</td>
      <td>326373.19</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>148649</th>
      <td>148650</td>
      <td>Roy I Tillery</td>
      <td>Custodian</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2014</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>148650</th>
      <td>148651</td>
      <td>Not provided</td>
      <td>Not provided</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2014</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>148651</th>
      <td>148652</td>
      <td>Not provided</td>
      <td>Not provided</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2014</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>148652</th>
      <td>148653</td>
      <td>Not provided</td>
      <td>Not provided</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2014</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>148653</th>
      <td>148654</td>
      <td>Joe Lopez</td>
      <td>Counselor, Log Cabin Ranch</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>-618.13</td>
      <td>0.0</td>
      <td>-618.13</td>
      <td>-618.13</td>
      <td>2014</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>148654 rows × 13 columns</p>
</div>



** Check the head of the DataFrame. **


```python
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
      <th>Id</th>
      <th>EmployeeName</th>
      <th>JobTitle</th>
      <th>BasePay</th>
      <th>OvertimePay</th>
      <th>OtherPay</th>
      <th>Benefits</th>
      <th>TotalPay</th>
      <th>TotalPayBenefits</th>
      <th>Year</th>
      <th>Notes</th>
      <th>Agency</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NATHANIEL FORD</td>
      <td>GENERAL MANAGER-METROPOLITAN TRANSIT AUTHORITY</td>
      <td>167411.18</td>
      <td>0.00</td>
      <td>400184.25</td>
      <td>NaN</td>
      <td>567595.43</td>
      <td>567595.43</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>GARY JIMENEZ</td>
      <td>CAPTAIN III (POLICE DEPARTMENT)</td>
      <td>155966.02</td>
      <td>245131.88</td>
      <td>137811.38</td>
      <td>NaN</td>
      <td>538909.28</td>
      <td>538909.28</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>ALBERT PARDINI</td>
      <td>CAPTAIN III (POLICE DEPARTMENT)</td>
      <td>212739.13</td>
      <td>106088.18</td>
      <td>16452.60</td>
      <td>NaN</td>
      <td>335279.91</td>
      <td>335279.91</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>CHRISTOPHER CHONG</td>
      <td>WIRE ROPE CABLE MAINTENANCE MECHANIC</td>
      <td>77916.00</td>
      <td>56120.71</td>
      <td>198306.90</td>
      <td>NaN</td>
      <td>332343.61</td>
      <td>332343.61</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>PATRICK GARDNER</td>
      <td>DEPUTY CHIEF OF DEPARTMENT,(FIRE DEPARTMENT)</td>
      <td>134401.60</td>
      <td>9737.00</td>
      <td>182234.59</td>
      <td>NaN</td>
      <td>326373.19</td>
      <td>326373.19</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```

** Use the .info() method to find out how many entries there are.**


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 148654 entries, 0 to 148653
    Data columns (total 13 columns):
     #   Column            Non-Null Count   Dtype  
    ---  ------            --------------   -----  
     0   Id                148654 non-null  int64  
     1   EmployeeName      148654 non-null  object 
     2   JobTitle          148654 non-null  object 
     3   BasePay           148045 non-null  float64
     4   OvertimePay       148650 non-null  float64
     5   OtherPay          148650 non-null  float64
     6   Benefits          112491 non-null  float64
     7   TotalPay          148654 non-null  float64
     8   TotalPayBenefits  148654 non-null  float64
     9   Year              148654 non-null  int64  
     10  Notes             0 non-null       float64
     11  Agency            148654 non-null  object 
     12  Status            0 non-null       float64
    dtypes: float64(8), int64(2), object(3)
    memory usage: 14.7+ MB
    


```python

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 148654 entries, 0 to 148653
    Data columns (total 13 columns):
    Id                  148654 non-null int64
    EmployeeName        148654 non-null object
    JobTitle            148654 non-null object
    BasePay             148045 non-null float64
    OvertimePay         148650 non-null float64
    OtherPay            148650 non-null float64
    Benefits            112491 non-null float64
    TotalPay            148654 non-null float64
    TotalPayBenefits    148654 non-null float64
    Year                148654 non-null int64
    Notes               0 non-null float64
    Agency              148654 non-null object
    Status              0 non-null float64
    dtypes: float64(8), int64(2), object(3)
    memory usage: 14.7+ MB
    

**What is the average BasePay ?**


```python
df['BasePay']
```




    0         167411.18
    1         155966.02
    2         212739.13
    3          77916.00
    4         134401.60
                ...    
    148649         0.00
    148650          NaN
    148651          NaN
    148652          NaN
    148653         0.00
    Name: BasePay, Length: 148654, dtype: float64




```python
df['BasePay'].mean()
```




    66325.44884050643




```python

```




    66325.44884050643



** What is the highest amount of OvertimePay in the dataset ? **


```python
df['OvertimePay'].max()
```




    245131.88




```python

```




    245131.88



** What is the job title of  JOSEPH DRISCOLL ? Note: Use all caps, otherwise you may get an answer that doesn't match up (there is also a lowercase Joseph Driscoll). **


```python
df['EmployeeName'] == 'JOSEPH DRISCOLL'
```




    0         False
    1         False
    2         False
    3         False
    4         False
              ...  
    148649    False
    148650    False
    148651    False
    148652    False
    148653    False
    Name: EmployeeName, Length: 148654, dtype: bool




```python
df[df['EmployeeName'] == 'JOSEPH DRISCOLL']['JobTitle']
```




    24    CAPTAIN, FIRE SUPPRESSION
    Name: JobTitle, dtype: object




```python

```




    24    CAPTAIN, FIRE SUPPRESSION
    Name: JobTitle, dtype: object



** How much does JOSEPH DRISCOLL make (including benefits)? **


```python
df[df['EmployeeName'] == 'JOSEPH DRISCOLL']['TotalPayBenefits']
```




    24    270324.91
    Name: TotalPayBenefits, dtype: float64




```python

```




    24    270324.91
    Name: TotalPayBenefits, dtype: float64



** What is the name of highest paid person (including benefits)?**


```python
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
      <th>Id</th>
      <th>EmployeeName</th>
      <th>JobTitle</th>
      <th>BasePay</th>
      <th>OvertimePay</th>
      <th>OtherPay</th>
      <th>Benefits</th>
      <th>TotalPay</th>
      <th>TotalPayBenefits</th>
      <th>Year</th>
      <th>Notes</th>
      <th>Agency</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NATHANIEL FORD</td>
      <td>GENERAL MANAGER-METROPOLITAN TRANSIT AUTHORITY</td>
      <td>167411.18</td>
      <td>0.00</td>
      <td>400184.25</td>
      <td>NaN</td>
      <td>567595.43</td>
      <td>567595.43</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>GARY JIMENEZ</td>
      <td>CAPTAIN III (POLICE DEPARTMENT)</td>
      <td>155966.02</td>
      <td>245131.88</td>
      <td>137811.38</td>
      <td>NaN</td>
      <td>538909.28</td>
      <td>538909.28</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>ALBERT PARDINI</td>
      <td>CAPTAIN III (POLICE DEPARTMENT)</td>
      <td>212739.13</td>
      <td>106088.18</td>
      <td>16452.60</td>
      <td>NaN</td>
      <td>335279.91</td>
      <td>335279.91</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>CHRISTOPHER CHONG</td>
      <td>WIRE ROPE CABLE MAINTENANCE MECHANIC</td>
      <td>77916.00</td>
      <td>56120.71</td>
      <td>198306.90</td>
      <td>NaN</td>
      <td>332343.61</td>
      <td>332343.61</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>PATRICK GARDNER</td>
      <td>DEPUTY CHIEF OF DEPARTMENT,(FIRE DEPARTMENT)</td>
      <td>134401.60</td>
      <td>9737.00</td>
      <td>182234.59</td>
      <td>NaN</td>
      <td>326373.19</td>
      <td>326373.19</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['TotalPayBenefits'] == df['TotalPayBenefits'].max()]
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
      <th>Id</th>
      <th>EmployeeName</th>
      <th>JobTitle</th>
      <th>BasePay</th>
      <th>OvertimePay</th>
      <th>OtherPay</th>
      <th>Benefits</th>
      <th>TotalPay</th>
      <th>TotalPayBenefits</th>
      <th>Year</th>
      <th>Notes</th>
      <th>Agency</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NATHANIEL FORD</td>
      <td>GENERAL MANAGER-METROPOLITAN TRANSIT AUTHORITY</td>
      <td>167411.18</td>
      <td>0.0</td>
      <td>400184.25</td>
      <td>NaN</td>
      <td>567595.43</td>
      <td>567595.43</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>EmployeeName</th>
      <th>JobTitle</th>
      <th>BasePay</th>
      <th>OvertimePay</th>
      <th>OtherPay</th>
      <th>Benefits</th>
      <th>TotalPay</th>
      <th>TotalPayBenefits</th>
      <th>Year</th>
      <th>Notes</th>
      <th>Agency</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NATHANIEL FORD</td>
      <td>GENERAL MANAGER-METROPOLITAN TRANSIT AUTHORITY</td>
      <td>167411.18</td>
      <td>0.0</td>
      <td>400184.25</td>
      <td>NaN</td>
      <td>567595.43</td>
      <td>567595.43</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



** What is the name of lowest paid person (including benefits)? Do you notice something strange about how much he or she is paid?**


```python
df[df['TotalPayBenefits'] == df['TotalPayBenefits'].min()]
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
      <th>Id</th>
      <th>EmployeeName</th>
      <th>JobTitle</th>
      <th>BasePay</th>
      <th>OvertimePay</th>
      <th>OtherPay</th>
      <th>Benefits</th>
      <th>TotalPay</th>
      <th>TotalPayBenefits</th>
      <th>Year</th>
      <th>Notes</th>
      <th>Agency</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>148653</th>
      <td>148654</td>
      <td>Joe Lopez</td>
      <td>Counselor, Log Cabin Ranch</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-618.13</td>
      <td>0.0</td>
      <td>-618.13</td>
      <td>-618.13</td>
      <td>2014</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>EmployeeName</th>
      <th>JobTitle</th>
      <th>BasePay</th>
      <th>OvertimePay</th>
      <th>OtherPay</th>
      <th>Benefits</th>
      <th>TotalPay</th>
      <th>TotalPayBenefits</th>
      <th>Year</th>
      <th>Notes</th>
      <th>Agency</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>148653</th>
      <td>148654</td>
      <td>Joe Lopez</td>
      <td>Counselor, Log Cabin Ranch</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-618.13</td>
      <td>0.0</td>
      <td>-618.13</td>
      <td>-618.13</td>
      <td>2014</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



** What was the average (mean) BasePay of all employees per year? (2011-2014) ? **


```python
dfYear = df[(df['Year'] >= 2011) & (df['Year'] < 2015)]
```


```python
dfYear
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
      <th>Id</th>
      <th>EmployeeName</th>
      <th>JobTitle</th>
      <th>BasePay</th>
      <th>OvertimePay</th>
      <th>OtherPay</th>
      <th>Benefits</th>
      <th>TotalPay</th>
      <th>TotalPayBenefits</th>
      <th>Year</th>
      <th>Notes</th>
      <th>Agency</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NATHANIEL FORD</td>
      <td>GENERAL MANAGER-METROPOLITAN TRANSIT AUTHORITY</td>
      <td>167411.18</td>
      <td>0.00</td>
      <td>400184.25</td>
      <td>NaN</td>
      <td>567595.43</td>
      <td>567595.43</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>GARY JIMENEZ</td>
      <td>CAPTAIN III (POLICE DEPARTMENT)</td>
      <td>155966.02</td>
      <td>245131.88</td>
      <td>137811.38</td>
      <td>NaN</td>
      <td>538909.28</td>
      <td>538909.28</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>ALBERT PARDINI</td>
      <td>CAPTAIN III (POLICE DEPARTMENT)</td>
      <td>212739.13</td>
      <td>106088.18</td>
      <td>16452.60</td>
      <td>NaN</td>
      <td>335279.91</td>
      <td>335279.91</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>CHRISTOPHER CHONG</td>
      <td>WIRE ROPE CABLE MAINTENANCE MECHANIC</td>
      <td>77916.00</td>
      <td>56120.71</td>
      <td>198306.90</td>
      <td>NaN</td>
      <td>332343.61</td>
      <td>332343.61</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>PATRICK GARDNER</td>
      <td>DEPUTY CHIEF OF DEPARTMENT,(FIRE DEPARTMENT)</td>
      <td>134401.60</td>
      <td>9737.00</td>
      <td>182234.59</td>
      <td>NaN</td>
      <td>326373.19</td>
      <td>326373.19</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>148649</th>
      <td>148650</td>
      <td>Roy I Tillery</td>
      <td>Custodian</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2014</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>148650</th>
      <td>148651</td>
      <td>Not provided</td>
      <td>Not provided</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2014</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>148651</th>
      <td>148652</td>
      <td>Not provided</td>
      <td>Not provided</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2014</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>148652</th>
      <td>148653</td>
      <td>Not provided</td>
      <td>Not provided</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2014</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>148653</th>
      <td>148654</td>
      <td>Joe Lopez</td>
      <td>Counselor, Log Cabin Ranch</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>-618.13</td>
      <td>0.0</td>
      <td>-618.13</td>
      <td>-618.13</td>
      <td>2014</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>148654 rows × 13 columns</p>
</div>




```python
dfYear.groupby('Year').mean()['BasePay']
```




    Year
    2011    63595.956517
    2012    65436.406857
    2013    69630.030216
    2014    66564.421924
    Name: BasePay, dtype: float64




```python

```




    Year
    2011    63595.956517
    2012    65436.406857
    2013    69630.030216
    2014    66564.421924
    Name: BasePay, dtype: float64



** How many unique job titles are there? **


```python
df['JobTitle'].drop_duplicates().count()
```




    2159




```python

```




    2159



    ** What are the top 5 most common jobs? **


```python
df['JobTitle'].value_counts().head(5)
```




    Transit Operator                7036
    Special Nurse                   4389
    Registered Nurse                3736
    Public Svc Aide-Public Works    2518
    Police Officer 3                2421
    Name: JobTitle, dtype: int64




```python

```




    Transit Operator                7036
    Special Nurse                   4389
    Registered Nurse                3736
    Public Svc Aide-Public Works    2518
    Police Officer 3                2421
    Name: JobTitle, dtype: int64



** How many Job Titles were represented by only one person in 2013? (e.g. Job Titles with only one occurence in 2013?) **


```python
(df[df['Year'] == 2013]['JobTitle'].value_counts() == 1)
```




    Transit Operator                  False
    Special Nurse                     False
    Registered Nurse                  False
    Public Svc Aide-Public Works      False
    Custodian                         False
                                      ...  
    Sr General Utility Mechanic        True
    Sprv Adult Prob Ofc (SFERS)        True
    Mayoral Staff XII                  True
    Pr Investigator, Tax Collector     True
    Senior Assessment Clerk            True
    Name: JobTitle, Length: 1051, dtype: bool




```python
dfYear2013 = df[(df['Year'] == 2013)]
```


```python
(dfYear2013['JobTitle'].value_counts() == 1).sum()
```




    202




```python
dfResult = dfYear2013.groupby('JobTitle')
```

    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x0000021AFD444160>
    


```python
dfResult
```




    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x0000021AFD444E80>




```python

```


```python

```




    202



** How many people have the word Chief in their job title? (This is pretty tricky) **


```python
df[df['JobTitle'].str.contains('Chief')].count()
```




    Id                  423
    EmployeeName        423
    JobTitle            423
    BasePay             420
    OvertimePay         423
    OtherPay            423
    Benefits            423
    TotalPay            423
    TotalPayBenefits    423
    Year                423
    Notes                 0
    Agency              423
    Status                0
    dtype: int64




```python
df['JobTitle'].apply(lambda x: True if 'chief' in x.lower().split() else False).sum()
```




    477




```python

```




    477



** Bonus: Is there a correlation between length of the Job Title string and Salary? **


```python
df['title_len'] = df['JobTitle'].apply(len)
```


```python
df[['TotalPayBenefits', 'title_len']].corr()
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
      <th>TotalPayBenefits</th>
      <th>title_len</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TotalPayBenefits</th>
      <td>1.000000</td>
      <td>-0.036878</td>
    </tr>
    <tr>
      <th>title_len</th>
      <td>-0.036878</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title_len</th>
      <th>TotalPayBenefits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>title_len</th>
      <td>1.000000</td>
      <td>-0.036878</td>
    </tr>
    <tr>
      <th>TotalPayBenefits</th>
      <td>-0.036878</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



# Great Job!
