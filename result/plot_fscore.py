import pandas as pd
from matplotlib import pyplot as plt

# Set the figure size
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

# Make a list of columns
columns = ['JOY', 'ANGER', 'SHAME', 'DISGUST', 'SADNESS', 'FEAR', 'GUILT']

# Read a CSV file
df = pd.read_csv("tf-idf-learningrate.csv", usecols=columns)

# Plot the lines
df.plot()
plt.title('Tf-idf with Different Learning Rates')
plt.xlabel('Learning Rates')
plt.ylabel('F1 Score')

values = ['0.01', '0.02', '0.03', '0.04', '0.05']
plt.xticks([0, 1, 2, 3, 4], values)

plt.show()