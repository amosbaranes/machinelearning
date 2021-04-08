"""
This is a practice file of very basic python knowledge

"""

# ----- simple Hello world ----
print("Hello World")
print('-----------')
print('Amos')

# ------- Conditions -----
# if 1 > 2:
#     print("Five is greater than two!")
# else:
#     print("no it is not")

# ------- Variables ------
# x = 5
# y = "Hello, World!"
# z = x * 20
#
# print(x)
# print(y)
# print(z)
#
# print(y*10)
# print('-'*50)

# x, y, z = "Orange", "Banana", "Cherry"
# print(x)
# print(y)
# print(z)
# print(x + ' ' + y)

# x = y = z = "Orange"
# print(x)
# print(y)
# print(z)

# -----
# x = 5
# b = type(x)
# print(b)
# print('==')
# x = 5.2456
# b = type(x)
# print(b)

# print('--')
# x = 'amos'
# b = type(x)
# print(b)
# print('--')

# ----- range ----
# x = range(6)
# print(x)
# for i in range(6):
#     print(i)

# dict ------
# x = {"name" : "John", "age" : 36}
# print(x)
# print(x["name"])
# print(x["age"])
# x["country"] = "Ivory Coast"
# print(x)
# --

x = {"apple", "banana", "cherry"}
# print(x)
# #
# # x = float(1)     # x will be 1.0
# # print(x)
# # y = float(2.8)   # y will be 2.8
# # print(y)
# # z = float("3")   # z will be 3.0
# # print(z)
# # w = float("4.2") # w will be 4.2
# # print(w)


# a = "Hello, World!, israel"
# b = a.split(",")
# c = type(b)
# print(b)
# print(c)
# print(b[0])
# print(b[2])
# print(b[3])

# for loop ---
# fruits = ["apple", "banana", "cherry"]
# for x in fruits:
#     print(x)


# break ---
# fruits = ["apple", "banana", "cherry"]
# for x in fruits:
#     if x == "banana":
#         break
#     print(x)
#
# print('---')
#
# fruits = ["apple", "banana", "cherry"]
# for x in fruits:
#     print(x)
#     if x == "banana":
#         break

# fruits = ["apple", "banana", "cherry"]
# for x in fruits:
#     print(x)
#     if x == "banana":
#         continue
#     print(x)

# functions
# def my_function():
#     print("Hello from a function")
#     print('----')
#     print('amos')
#
# my_function()
# def my_function(fname):
#     print(fname + " Refsnes")
#
#
# my_function(fname = "Emil")
# my_function(fname = "Tobias")
# my_function(fname = "Linus")

# default declaration
# def my_function(fname = "Amos"):
#     print(fname + " Refsnes")
#
#
# my_function(fname = "Emil")
# my_function(fname = "Tobias")
# my_function()

# ----
# def ProcessFile(s):
#     print('------')
#     print(s)
#     print('------')
#
#
# def my_function(food = []):
#     for x in food:
#         ProcessFile('Year_' + str(x) + '.xlsx')
#
#
# fruits = [2015, 2017, 2018, 2019]
# my_function(fruits)
# --

# # -- Classes ---
# class Animal:
#     def __init__(self, t, h, l):
#         self.type = t
#         self.hands = h
#         self.legs = l
#
#     def set_legs(self, k):
#         self.legs=k
#
#
# class Cat(Animal):
#     def __init__(self):
#         Animal.__init__(self, t='cat', h=0, l=4)
#
#
# class Human(Animal):
#     def __init__(self, cat=None):
#         Animal.__init__(self, t='human', h=2, l=2)
#         self.cat = cat
#
#
# c = Cat()
# print(c.type + ' : ' + str(c.hands) + ' : ' + str(c.legs))
# print('--------')
# c.set_legs(k=3)
# print(c.type + ' : ' + str(c.hands) + ' : ' + str(c.legs))
# print('--------')
#
# h = Human()
# hc = Human(cat=c)
# print(h.type + ' : ' + str(h.hands) + ' : ' + str(h.legs))
# print('--')
# hc.cat.set_legs(k=4)
# print('--------')
# print(hc.cat.type + ' : ' + str(hc.cat.hands) + ' : ' + str(hc.cat.legs))
# # End classes

