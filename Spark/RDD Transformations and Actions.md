```python
# from pyspark import SparkContext
'''
import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext(appName="myAppName")
'''
```


      File "<ipython-input-7-c68727a8c088>", line 2
        ```
        ^
    SyntaxError: invalid syntax




```python
from pyspark import SparkContext
sc = SparkContext()
```


```python
%%writefile example2.txt
first
second line
the third line
then a fourth line
```

    Overwriting example2.txt



```python
sc.textFile('example2.txt')
```




    example2.txt MapPartitionsRDD[1] at textFile at NativeMethodAccessorImpl.java:0




```python
text_rdd = sc.textFile('example2.txt')
```


```python
words = text_rdd.map(lambda line: line.split())
```


```python
words.collect()
```




    [['first'],
     ['second', 'line'],
     ['the', 'third', 'line'],
     ['then', 'a', 'fourth', 'line']]




```python
text_rdd.collect()
```




    ['first', 'second line', 'the third line', 'then a fourth line']




```python
text_rdd.flatMap(lambda line: line.split()).collect()
```




    ['first',
     'second',
     'line',
     'the',
     'third',
     'line',
     'then',
     'a',
     'fourth',
     'line']




```python
%%writefile services.txt
#EventId    Timestamp    Customer   State    ServiceID    Amount
201       10/13/2017      100       NY       131          100.00
204       10/18/2017      700       TX       129          450.00
202       10/15/2017      203       CA       121          200.00
206       10/19/2017      202       CA       131          500.00
203       10/17/2017      101       NY       173          750.00
205       10/19/2017      202       TX       121          200.00
```

    Writing services.txt



```python
services = sc.textFile('services.txt')
```


```python
services.take(2)
```




    ['#EventId    Timestamp    Customer   State    ServiceID    Amount',
     '201       10/13/2017      100       NY       131          100.00']




```python
services.map(lambda line: line.split()).take(3)
```




    [['#EventId', 'Timestamp', 'Customer', 'State', 'ServiceID', 'Amount'],
     ['201', '10/13/2017', '100', 'NY', '131', '100.00'],
     ['204', '10/18/2017', '700', 'TX', '129', '450.00']]




```python
clean = services.map(lambda line: line[1:] if line[0] == '#' else line)
```


```python
clean = clean.map(lambda line: line.split())
```


```python
clean.collect()
```




    [['EventId', 'Timestamp', 'Customer', 'State', 'ServiceID', 'Amount'],
     ['201', '10/13/2017', '100', 'NY', '131', '100.00'],
     ['204', '10/18/2017', '700', 'TX', '129', '450.00'],
     ['202', '10/15/2017', '203', 'CA', '121', '200.00'],
     ['206', '10/19/2017', '202', 'CA', '131', '500.00'],
     ['203', '10/17/2017', '101', 'NY', '173', '750.00'],
     ['205', '10/19/2017', '202', 'TX', '121', '200.00']]




```python
pairs = clean.map(lambda lst: (lst[3], lst[-1]))
```


```python
pairs.collect()
```




    [('State', 'Amount'),
     ('NY', '100.00'),
     ('TX', '450.00'),
     ('CA', '200.00'),
     ('CA', '500.00'),
     ('NY', '750.00'),
     ('TX', '200.00')]




```python
rekey = pairs.reduceByKey(lambda amt1, amt2: float(amt1) + float(amt2))
```


```python
rekey.collect()
```




    [('State', 'Amount'), ('NY', 850.0), ('TX', 650.0), ('CA', 700.0)]




```python
clean.collect()
```




    [['EventId', 'Timestamp', 'Customer', 'State', 'ServiceID', 'Amount'],
     ['201', '10/13/2017', '100', 'NY', '131', '100.00'],
     ['204', '10/18/2017', '700', 'TX', '129', '450.00'],
     ['202', '10/15/2017', '203', 'CA', '121', '200.00'],
     ['206', '10/19/2017', '202', 'CA', '131', '500.00'],
     ['203', '10/17/2017', '101', 'NY', '173', '750.00'],
     ['205', '10/19/2017', '202', 'TX', '121', '200.00']]




```python
# Grab (State, Amount)
step1 = clean.map(lambda lst: (lst[3], lst[-1]))
# Reduce by Key
step2 = step1.reduceByKey(lambda amt1, amt2: float(amt1) + float(amt2))
# Get rid of State, Amount titles
step3 = step2.filter(lambda x: not x[0] == 'State')
# Sort Results by Amount
step4 = step3.sortBy(lambda stAmount: stAmount[1], ascending=False)
# Action
step4.collect()
```




    [('NY', 850.0), ('CA', 700.0), ('TX', 650.0)]




```python
x = ['ID', 'State', 'Amount']
```


```python
def func1(lst):
    return lst[-1]
```


```python
def func2(id_st_amt):
    # unpack values
    (id, st, amt) = id_st_amt
    return amt
```


```python
func1(x)
```




    'Amount'




```python
func2(x)
```




    'Amount'




```python

```
