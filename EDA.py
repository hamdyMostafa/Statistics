# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:46:49 2018

@author: hamdymostafa

"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

######### Exporatory Data Analysis ##############


# download the data from 
# "https://www.datacamp.com/courses/statistical-thinking-in-python-part-1"

## 1. Graphical exploratory data analysis
#  Histogram, bee swarm plots, ECDF plots , box plots , Scatter plots


## Histogram ##

# Set default Seaborn style (it will adjust the bin automatically)
sns.set()

# Compute number of data points: n_data
n_data = len(versicolor_petal_length)

# Number of bins is the square root of number of data points: n_bins
n_bins = np.sqrt(n_data)

# Convert number of bins to integer: n_bins
n_bins = int(n_bins)

# Plot the histogram
plt.hist(versicolor_petal_length, bins = n_bins)

# Label axes
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('count')

# Show histogram
plt.show()



## Bee swarm plots ##

# Create bee swarm plot with Seaborn's default settings
_ = sns.swarmplot(x = 'species' , y ='petal length (cm)' ,data = df )

# Label the axes
_ = plt.xlabel('species')
_ = plt.ylabel('petal length (cm)')

# Show the plot

plt.show()




## ECDF plots ##


def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y


# Compute ECDF for versicolor data: x_vers, y_vers
x_vers, y_vers = ecdf(versicolor_petal_length)

# Generate plot
plt.plot(x_vers,y_vers,marker = '.',linestyle = 'none')

# Make the margins nice
plt.margins(0.02)

# Label the axes
_ = plt.xlabel('versicolor_petal_length')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()


# Comparison of ECDFs



# Compute ECDFs
x_set, y_set = ecdf(setosa_petal_length)
x_vers, y_vers = ecdf(versicolor_petal_length)
x_virg, y_virg = ecdf(virginica_petal_length)



# Plot all ECDFs on the same plot
plt.plot(x_set, y_set , marker = '.', linestyle = 'none')
plt.plot(x_vers, y_vers , marker = '.', linestyle = 'none')
plt.plot(x_virg, y_virg, marker = '.', linestyle = 'none')



# Make nice margins
plt.margins(0.02)

# Annotate the plot
plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()



# Box-and-whisker plot

# Create box plot with Seaborn's default settings
sns.boxplot(x = 'species' ,y = 'petal length (cm)', data = df )

# Label the axes
_ = plt.xlabel('species')
_ = plt.ylabel('petal length (cm)')

# Show the plot

plt.show()




#  scatter plot

_ = plt.plot(versicolor_petal_length,versicolor_petal_width , marker = '.', linestyle = 'none')

# Set margins
plt.margins(0.02)

# Label the axes

_ = plt.xlabel('versicolor petal length')
_ = plt.ylabel('versus petal width')

# Show the result
plt.show()




## 2. Quantitative exploratory data analysis
# mean , median , percentiles , variance & standard deviation , covariance , correlation



# mean

mean_length_vers = np.mean(versicolor_petal_length)

# percentiles

percentiles = np.array([2.5 , 25,50, 75, 97.5])

# Compute percentiles: ptiles_vers
ptiles_vers = np.percentile(versicolor_petal_length,percentiles )


# Comparing percentiles to ECDF
# Plot the ECDF
_ = plt.plot(x_vers, y_vers, '.')
plt.margins(0.02)
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Overlay percentiles as red diamonds.
_ = plt.plot(ptiles_vers, percentiles/100, marker='D', color='red',
         linestyle='none')

# Show the plot
plt.show()


# variance & std

variance_np = np.var(versicolor_petal_length)

std_np = np.std(versicolor_petal_length)



#  covariance  

covariance_matrix = np.cov(versicolor_petal_length,versicolor_petal_width )
petal_cov = covariance_matrix[0,1]

# correlation
corr_mat = np.corrcoef(x,y)
r = corr_mat[0,1]










