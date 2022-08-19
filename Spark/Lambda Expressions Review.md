```python
def square(num):
    result = num**2
    return result
```


```python
square(4)
```




    16




```python
def square(num):
    return num**2
```


```python
square(4)
```




    16




```python
lambda num: num**2
```




    <function __main__.<lambda>(num)>




```python
sq = lambda num: num**2
```


```python
sq(5)
```




    25




```python
even = lambda num: num%2 == 0
```


```python
even(3)
```




    False




```python
even(4)
```




    True




```python
first = lambda s: s[0]
```


```python
first('asdfasef')
```




    'a'




```python
rev = lambda s: s[::-1] # reverse text
```


```python
rev('asdfasef')
```




    'fesafdsa'




```python
def adder(x,y):
    return x+y
```


```python
adder(3,4)
```




    7




```python
adder = lambda x,y: x+y
```


```python
adder(4,5)
```




    9




```python

```
