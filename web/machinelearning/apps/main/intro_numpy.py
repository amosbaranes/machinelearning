import pandas as pd
import numpy as np

# # print(' -- 1 ---')
# # print('basic array')
# # print('-'*10)
# # a = np.array([1, 2, 3])   # Create a rank 1 array
# # print(type(a))            # Prints "<class 'numpy.ndarray'>"
# # print(a.shape)            # Prints "(3,)"
# # print(a[0], a[1], a[2])   # Prints "1 2 3"
# # a[0] = 5                  # Change an element of the array
# # print(a)                  # Prints "[5, 2, 3]"
# # # ---
#
# # print(' -- 2 ---')
# # b = np.array([[1, 2, 3], [4, 5, 6]])    # Create a rank 2 array
# # print(b.shape)                     # Prints "(2, 3)"
# # print(b[0, 0], b[0, 1], b[1, 0])   # Prints "1 2 4"
# #
# # print('-'*30)
# # print('Numpy also provides many functions to create arrays:')
# # print('-'*30)
# # a = np.zeros((2, 2))   # Create an array of all zeros
# # print(a)
# # print('-'*10)
# # b = np.ones((1, 2))    # Create an array of all ones
# # print(b)
# # print('-'*10)
# # c = np.full((2, 2), 7)  # Create a constant array
# # print(c)
# # print('-'*10)
# # # --
#
# # print(' -- 3 ---')
# # d = np.eye(2)         # Create a 2x2 identity matrix
# # print(d)
# # print('-'*10)
# #
# # e = np.random.random((2, 2))  # Create an array filled with random values
# # print(e)
# #
#
# print(' -- 4 ---')
# print('-'*30)
# print('Array indexing')
# print('Slicing')
# print('-'*30)
# # Create the following rank 2 array with shape (3, 4)
# # [[ 1  2  3  4]
# #  [ 5  6  7  8]
# #  [ 9 10 11 12]]
# a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# print(a)
# # # #
# # #
# # Use slicing to pull out the subarray consisting of the first 2 rows
# # and columns 1 and 2; b is the following array of shape (2, 2):
# # [[2 3]
# #  [6 7]]
# b = a[:2, 1:3]
# print(b)
# # print('-'*10)
# # A slice of an array is a view into the same data, so modifying it
# # will modify the original array.
# print(a[0, 1])   # Prints "2"
# b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
# print(a[0, 1])   # Prints "77"
# print('-'*10)
# print('-4.1')
# print('-'*10)
#
# # Two ways of accessing the data in the middle row of the array.
# # Mixing integer indexing with slices yields an array of lower rank,
# # while using only slices yields an array of the same rank as the
# # original array:
# row_r1 = a[1, :]    # Rank 1 view of the second row of a
# row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
# print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
# print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"
#
# # We can make the same distinction when accessing columns of an array:
# col_r1 = a[:, 1]
# col_r2 = a[:, 1:2]
# print(col_r1, col_r1.shape)  # Prints "[ 2  6 10] (3,)"
# print(col_r2, col_r2.shape)  # Prints "[[ 2]
#                              #          [ 6]
#                              #          [10]] (3, 1)"
#
# print(' -- - ---')
# print(' -- 5 ---')
# print(' -- - ---')
# print('Integer array indexing:')
# print('-'*30)
# print('a = np.array([[1,2], [3, 4], [5, 6]])')
# a = np.array([[1, 2], [3, 4], [5, 6]])
# print('-- a --')
# print(a)
# print('--')
# # An example of integer array indexing.
# # The returned array will have shape (3,) and
# print('a[[0, 1, 2], [0, 1, 0]])')
# print('--')
# print(a[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"
# print('--')
#
# # The above example of integer array indexing is equivalent to this:
# print(np.array([a[0, 0], a[1, 1], a[2, 0]]))  # Prints "[1 4 5]"
# print('--')
#
# # When using integer array indexing, you can reuse the same
# # element from the source array:
# print(a[[0, 0], [1, 1]])  # Prints "[2 2]"
#
# # Equivalent to the previous integer array indexing example
# print(np.array([a[0, 1], a[0, 1]]))  # Prints "[2 2]"
#
# print(' -- 6 ---')
# print('One useful trick with integer array indexing is selecting or mutating one element from each row of a matrix:')
# print('-'*30)
# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
#
# print(a)  # prints "array([[ 1,  2,  3],
#           #                [ 4,  5,  6],
#           #                [ 7,  8,  9],
#           #                [10, 11, 12]])"
#
# print(' Create an array of indices')
# print(' b = np.array([0, 2, 0, 1])')
# b = np.array([0, 2, 0, 1])
#
# print(' Select one element from each row of a using the indices in b')
# print('-'*40)
# print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"
# print('--')
# print('Mutate one element from each row of a using the indices in b')
# a[np.arange(4), b] += 10
# print(a)  # prints "array([[11,  2,  3],
#           #                [ 4,  5, 16],
#           #                [17,  8,  9],
#           #                [10, 21, 12]])
# print('-'*30)
#
#
# print('-'*20)
# print(' -- 7 ---')
# print('-'*20)
# print('Boolean array indexing')
# print('-'*30)
# a = np.array([[1, 2], [3, 4], [5, 6]])
# print(a)
# print('--')
# bool_idx = (a > 2)   # Find the elements of a that are bigger than 2;
#                      # this returns a numpy array of Booleans of the same
#                      # shape as a, where each slot of bool_idx tells
#                      # whether that element of a is > 2.
#
# print(bool_idx)      # Prints "[[False False]
#                      #          [ True  True]
#                      #          [ True  True]]"
#
# # We use boolean array indexing to construct a rank 1 array
# # consisting of the elements of a corresponding to the True values
# # of bool_idx
# print(a[bool_idx])  # Prints "[3 4 5 6]"
# print('--')
# # We can do all of the above in a single concise statement:
# print(a[a > 2])     # Prints "[3 4 5 6]"
#
#
# print('-'*20)
# print(' -- 8 ---')
# print('-'*20)
# print('-'*30)
# print('Datatypes')
# print('-'*30)
# x = np.array([1, 2])   # Let numpy choose the datatype
# print(x.dtype)         # Prints "int64"
# print('--')
# x = np.array([1.0, 2.0])   # Let numpy choose the datatype
# print(x.dtype)             # Prints "float64"
#
# x = np.array([1, 2], dtype=np.int64)   # Force a particular datatype
# print(x.dtype)
#
#
# print('-'*20)
# print(' -- 9 ---')
# print('-'*20)
# print('basic math')
# print('-'*30)
# print('Array math')
# print('-'*30)
# x = np.array([[1, 2], [3, 4]], dtype=np.float64)
# y = np.array([[5, 6], [7, 8]], dtype=np.float64)
# print('x')
# print(x)
# print('y')
# print(y)
# print('--')
# print(' Elementwise sum; both produce the array')
# print('add matrix')
#                 # [[ 6.0  8.0]
#                 #  [10.0 12.0]]
# print(x + y)
# print('--')
# print(np.add(x, y))
# print('-----')
#
# # Elementwise difference; both produce the array
#                 # [[-4.0 -4.0]
#                 #  [-4.0 -4.0]]
# print('-- subtract')
# print(x - y)
# print('--')
# print(np.subtract(x, y))
#
# print('Elementwise product; both produce the array')
# # [[ 5.0 12.0]
# #  [21.0 32.0]]
# print(x * y)
# print('--')
# print(np.multiply(x, y))
#
# print('Elementwise division; both produce the array')
# # [[ 0.2         0.33333333]
# #  [ 0.42857143  0.5       ]]
# print(x / y)
# print(np.divide(x, y))
#
# print('Elementwise square root; produces the array')
#         # [[ 1.          1.41421356]
#         #  [ 1.73205081  2.        ]]
# print(np.sqrt(x))
#
#
# print(' -- 10 ---')
# print('inner product vectors and matrixes')
# v = np.array([9, 10])
# w = np.array([11, 12])
# x = np.array([[1, 2], [3, 4]], dtype=np.float64)
# y = np.array([[5, 6], [7, 8]], dtype=np.float64)
# print('x')
# print(x)
# print('y')
# print(y)
# # Inner product of vectors; both produce 219
# print(v.dot(w))
# print(np.dot(v, w))
# print('--')
# # Matrix / vector product; both produce the rank 1 array [29 67]
# print(w.dot(v))
# print(np.dot(w, v))
# print('----')
#
# print('Matrix / matrix product; both produce the rank 2 array')
#                 # [[19 22]
#                 #  [43 50]]
# print(x.dot(y))
# print('--')
# print(np.dot(x, y))
#
# print('-'*30)
# print('Numpy provides many useful functions ')
# print('-'*30)
# x = np.array([[1, 2], [3, 4]])
# print(x)
# print('--')
# print(np.sum(x))  # Compute sum of all elements; prints "10"
# print('--')
# print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
# print('--')
# print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"
#
# print('-'*30)
# print('transpose a matrix, use the T attribute of an array object')
# print('-'*30)
# x = np.array([[1, 2], [3, 4]])
# print(x)    # Prints "[[1 2]
#             #          [3 4]]"
# print('--')
# print(x.T)  # Prints "[[1 3]
#             #          [2 4]]"
#
# print('-'*30)
# print('Note that taking the transpose of a rank 1 array does nothing:')
# print('-'*30)
# v = np.array([1, 2, 3])
# print(v)    # Prints "[1 2 3]"
# print('--')
# print(v.T)  # Prints "[1 2 3]"


# print(' -- 11 ---')
# print('-'*30)
# print('Broadcasting')
# print('-'*30)
# # We will add the vector v to each row of the matrix x,
# # storing the result in the matrix y
# x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# v = np.array([1, 0, 1])
# y = np.empty_like(x)   # Create an empty matrix with the same shape as x
# print('x')
# print(x)
# print('v')
# print(v)
# print('y')
# print(y)
# print('--')
#
# print('Add the vector v to each row of the matrix x with an explicit loop')
# for i in range(4):
#     y[i, :] = x[i, :] + v
#
# print('Now y is the following')
#     # [[ 2  2  4]
#     #  [ 5  5  7]
#     #  [ 8  8 10]
#     #  [11 11 13]]
# print(y)
# print('--')
#
# print('-'*30)
# print('We will add the vector v to each row of the matrix x,')
# print('storing the result in the matrix y')
# print('-'*30)
# x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# v = np.array([1, 0, 1])
# vv = np.tile(v, (4, 1))   # Stack 4 copies of v on top of each other
# print('--')
# print(vv)                 # Prints "[[1 0 1]
#                           #          [1 0 1]
#                           #          [1 0 1]
#                           #          [1 0 1]]"
# print('--')
# y = x + vv  # Add x and vv elementwise
# print(y)  # Prints "[[ 2  2  4
#           #          [ 5  5  7]
#           #          [ 8  8 10]
#           #          [11 11 13]]"
#
# print('Numpy broadcasting allows us to perform this computation without')
# print('actually creating multiple copies of v. Consider this version, using broadcasting')
# x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# v = np.array([1, 0, 1])
# y = x + v  # Add v to each row of x using broadcasting
# print('x')
# print(x)
# print('v')
# print(v)
# print('y = x + v')
# print(y)  # Prints "[[ 2  2  4]
#           #          [ 5  5  7]
#           #          [ 8  8 10]
#           #          [11 11 13]]"

# print('------12--------')
# print('To compute an outer product, we first reshape v to be a column')
# print('vector of shape (3, 1); we can then broadcast it against w to yield')
# print('an output of shape (3, 2), which is the outer product of v and w:')
#
# v = np.array([1, 2, 3])  # v has shape (3,)
# w = np.array([4, 5])    # w has shape (2,)
# print('v')
# print(v)
# print('w')
# print(w)
#                 # [[ 4  5]
#                 #  [ 8 10]
#                 #  [12 15]]
# print('np.reshape(v, (3, 1))')
# print(np.reshape(v, (3, 1)))
# print('--')
# print('np.reshape(v, (3, 1)) * w')
# print(np.reshape(v, (3, 1)) * w)
#
# print('Add a vector to each row of a matrix')
# x = np.array([[1, 2, 3], [4, 5, 6]])
# # x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# # giving the following matrix:
# # [[2 4 6]
# #  [5 7 9]]
# print('x')
# print(x)
# print('v')
# print(v)
# print('x + v')
# print(x + v)
#
# print('Add a vector to each column of a matrix')
# print('x has shape (2, 3) and w has shape (2,).')
# print('If we transpose x then it has shape (3, 2) and can be broadcast')
# print("int('yields the final result of shape (2, 3) which is the matrix x with")
# print('the vector w added to each column. Gives the following matrix:')
#     # [[ 5  6  7]
#     #  [ 9 10 11]]
# print('print((x.T + w).T)')
# print('Another solution is to reshape w to be a column vector of shape (2, 1);')
# print('we can then broadcast it directly against x to produce the same')
# print('output.')
# print(x + np.reshape(w, (2, 1)))
#
# print('Multiply a matrix by a constant:')
# print('x has shape (2, 3). Numpy treats scalars as arrays of shape ();')
# print('these can be broadcast together to shape (2, 3), producing the')
# print('following array:')
#         # [[ 2  4  6]
#         #  [ 8 10 12]]
#
# print('x')
# print(x)
# print('x * 2')
# print(x * 2)
#
# # print('End')

# -------
# SciPy
# -------
# See the code in the reference.
# We will use similar code during the ML course.
# print('End')