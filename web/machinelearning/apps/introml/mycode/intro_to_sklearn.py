from .algos import Perceptron, AdalineSGD, MLData, plot_decision_regions
import numpy as np
import matplotlib.pyplot as plt

# Get data
data = MLData(file_name='iris')
df = data.get_data()
print(df.tail())