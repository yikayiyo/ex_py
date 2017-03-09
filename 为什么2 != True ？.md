
##真值测试
``` python
True==0		# False
True==1		# True
True==2		# False
False==0		# True
False==1		# False
False==2		# False
```

####python中的任何对象都有真值，以下对象的真值为**false**：
	- None
	- False
	- zero of any numeric type,for example 0,0.0,0j.
	- any empty sequence,for example,'',(),[].
	- any empty mapping,for example,{}
	- instances of user-defined classes, if the class defines a __bool__() or __len__() method, when that method returns the integer zero or bool value False. 

其它所有对象的真值都被视为 **true** 。一些返回布尔值的操作和内置函数，总是返回0或者False来表示真值为**false**，返回1或者True来表示真值为**true**. (例外: **布尔操作** or 和 and 总是返回其操作数)
##布尔操作

```python
True and 0		# 0	
0 and True		# 0
True and 1		# 1
1 and True		# True
True and 2		# 2
2 and True		# True
False and 0		# False
0 and False		# 0
False and 1		# False
1 and False		# False
False and 2		# False
2 and False		# False
```
```
True or 0		# True
0 or True		# True
True or 1		# True
1 or True		# 1
True or 2		# True
2 or True		# 2
False or 0		# 0
0 or False		# False
False or 1		# 1
1 or False		# 1
False or 2		# 2
2 or False		# 2
```
####python中的布尔操作，有如下规则：
| Operations | Result | Notes |
|--|-|-|
|x or y|if x is false,then y,else x|(1)|
|x and y|if x is false,then x,else y|(2)|
|not x|if x is false,then True,else False|(3)|

Notes:

(1) This is a short-circuit operator, so it only evaluates the second argument if the first one is false.
(2) This is a short-circuit operator, so it only evaluates the second argument if the first one is true.
(3) not has a lower priority than non-Boolean operators, so **not a == b** is interpreted as **not (a == b)**, and **a == not b** is a syntax error.


##布尔值
```python
isinstance(True,int)	# True
isinstance(False,int)	# True
True is 1		# False
False is 0		# False
int(True)		# 1
int(False)		# 0
bool(2) 		# True
bool(0) 		# False
```
	布尔值包括两个常量对象，分别是True和False，来表示真值；
	在数值上下文中，布尔值True和False等同于1和0，例如：5+True，返回了6 ；
	内置函数bool()可以将任何值转换为布尔值，前提是该值可以被解释为真值。

 结论：2 != True ，但是 bool(2) == True 。 
 

```python
bool(2) == True		#True
2 != True			#True
```