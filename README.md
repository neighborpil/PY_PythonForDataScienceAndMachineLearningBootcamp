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
