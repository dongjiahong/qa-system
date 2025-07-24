# Python编程基础

## 什么是Python？

Python是一种高级编程语言，由Guido van Rossum在1989年发明。它具有简洁、易读的语法，广泛应用于Web开发、数据科学、人工智能等领域。

## Python的特点

1. **简洁易读**：Python的语法接近自然语言，容易理解和学习
2. **跨平台**：可以在Windows、macOS、Linux等多种操作系统上运行
3. **丰富的库**：拥有大量的第三方库和框架
4. **解释型语言**：无需编译，可以直接运行

## Python的应用领域

- **Web开发**：使用Django、Flask等框架
- **数据科学**：使用NumPy、Pandas、Matplotlib等库
- **人工智能**：使用TensorFlow、PyTorch等框架
- **自动化脚本**：系统管理、文件处理等

## 基本语法示例

### 变量定义
```python
name = "Python"
version = 3.12
is_popular = True
```

### 函数定义
```python
def greet(name):
    return f"Hello, {name}!"

def calculate_area(length, width):
    return length * width
```

### 条件语句
```python
if version >= 3:
    print("使用Python 3")
else:
    print("使用Python 2")
```

### 循环语句
```python
# for循环
for i in range(5):
    print(f"数字: {i}")

# while循环
count = 0
while count < 3:
    print(f"计数: {count}")
    count += 1
```

## 数据结构

### 列表（List）
```python
fruits = ["苹果", "香蕉", "橙子"]
fruits.append("葡萄")
print(fruits[0])  # 输出: 苹果
```

### 字典（Dictionary）
```python
person = {
    "name": "张三",
    "age": 25,
    "city": "北京"
}
print(person["name"])  # 输出: 张三
```

### 元组（Tuple）
```python
coordinates = (10, 20)
x, y = coordinates
```

## 面向对象编程

```python
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def study(self, subject):
        return f"{self.name}正在学习{subject}"

# 创建对象
student = Student("李四", 20)
print(student.study("Python"))
```

## 异常处理

```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("不能除以零")
except Exception as e:
    print(f"发生错误: {e}")
finally:
    print("清理资源")
```

## 文件操作

```python
# 读取文件
with open("data.txt", "r", encoding="utf-8") as file:
    content = file.read()

# 写入文件
with open("output.txt", "w", encoding="utf-8") as file:
    file.write("Hello, Python!")
```

## 模块和包

```python
# 导入标准库
import os
import sys
from datetime import datetime

# 导入第三方库
import requests
import pandas as pd

# 使用模块
current_time = datetime.now()
print(f"当前时间: {current_time}")
```

## 总结

Python是一门功能强大且易于学习的编程语言，适合初学者入门，也能满足专业开发的需求。通过掌握基本语法、数据结构、面向对象编程等核心概念，可以开始Python编程之旅。