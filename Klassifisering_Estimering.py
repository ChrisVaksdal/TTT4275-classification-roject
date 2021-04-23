from sklearn.datasets import load_iris
from pathlib import Path
import numpy as np

iris = load_iris
# Sette et 1 tall bak alt
#Dataen X har en form på 1x4, men man har lyst til å legge til et 1-tall på slutten slik at vi får en matrise på formen 1x5.
# D = fetures
# C = classer
# Make the dataOpen dir and r

# heyehye


# def add_one(filepath):
#     entries = Path(filepath)
#     i = 1
#     for entry in entries.iterdir():
#         if entry.name == f'class_{i}':
#             data = np.genfromtxt(entry, delimiter=',')
#             print(data[0][0])
#             # np.append(data, [[1.0]], axis=0)
#             print(data[0])
#             # i += 1
#     # print(data)

# one = []
# for i in range(len(data)):  # array with ones, sa long as data
#     one.append(1.0)
#
# d = np.append(data, one, axis=1)
# print(d)
#
# print(one)
# print(len(one))
# print(len(data))
# # print(data)

# data1 = data[0]
# print(data1)
#
# data = [1, 2, 3, 4]
# data.insert(len(data), 1.0)
#
# print(data)
# data = np.fromfile(filename, dtype=float, count=-1, sep=",")
# print(data)

# vektor med enere


filename = 'C:\\Users\\amand\Documents\\NTNU studier\\6. semester øvinger\\Estimering\\Iris_TTT4275\\class_1'

data = np.genfromtxt('C:\\Users\\amand\Documents\\NTNU studier\\6. semester øvinger\\Estimering\\Iris_TTT4275\\class_1', dtype='float_',delimiter=',')
data1 = np.loadtxt(filename, dtype='float_', delimiter=',')
print(data1)

