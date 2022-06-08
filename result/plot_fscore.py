import pandas as pd
from matplotlib import pyplot as plt

# Set the figure size
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

# Make a list of columns
columns = ['JOY', 'ANGER', 'SHAME', 'DISGUST', 'SADNESS', 'FEAR', 'GUILT']

# Read a CSV file from the 'result' folder
df = pd.read_csv("result/BOW-niter.csv", usecols=columns)

# Plot the lines
df.plot()

# Set the title, labels and the position of the legend
plt.title('BOW with Different Iterations')
plt.xlabel('Iterations')
plt.ylabel('F1 Score')
plt.legend(loc='upper right')

# Define values of x-axis of learning rates
# Comment this part when plotting iterations
#values = ['0.01', '0.02', '0.03', '0.04', '0.05']
#plt.xticks([0, 1, 2, 3, 4], values)

plt.show()