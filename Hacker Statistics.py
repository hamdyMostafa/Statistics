# -*- coding: utf-8 -*-
"""
@author: hamdymostafa
"""
### Hacker stats probabilities "wear your hacker statistician hat : ) " 
# 1. Determine how to simulate data
# 2. Simulate many many times
# 3. Probability is approximately fraction of trails with the outcome of interest

# You have the power of a computer. If you can simulate a story, you can get its distribution

# Course: Statistical Thinking in python


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##### Question: What is the probability of getting 4 heads in flipping a coin 4 times ? 

n_all_4heads = 0

for _ in range(10000):
    heads = np.random.random(size = 4) > 0.5
    n_heads = np.sum(heads)
    
    if n_heads == 4:
        n_all_4heads += 1

prob = n_all_4heads / 10000

print(prob)

# The number r of sucesses in n Bernoulli trials with p probability of sucess, is binomially distributed
# Our question distribution is the Binomial distribution 
# Sampling from the binomial Distribution

#another solution to the question above
samples = np.random.binomial(4,0.5,size = 10000)
event = np.sum(samples == 4 )
prob = event / 10000

print(prob)




###### Question : How many defaults might we expect? ..
#  Let's say a bank made 100 mortgage loans .. given that the probability of a default is p = 0.05. 

sns.set()
np.random.seed(42)

# Initialize the number of defaults: n_defaults


def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    # Initialize number of successes: n_success
    n_success = 0
    
    # Perform trials
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = np.random.random()

        # If less than p, it's a success so add one to n_success
        if random_number < p:
            n_success +=1

    return n_success


n_defaults = np.empty(1000)
# Compute the number of defaults
for i in range(1000):
    n_defaults[i] = perform_bernoulli_trials(100,0.05)


# Plot the histogram with default number of bins; label your axes
_ = plt.hist(n_defaults, normed=True)
_ = plt.xlabel('number of defaults out of 100 loans')
_ = plt.ylabel('probability')

# Show the plot
plt.show()


####### Question : Will the bank fail?
# the bank will lose money if 10 or more of its loans are defaulted upon,
#  what is the probability that the bank will lose money?



def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

# Compute ECDF: x, y
x,y = ecdf(n_defaults)
# Plot the ECDF with labeled axes

plt.plot(x,y,marker = '.', linestyle = 'none')
_ = plt.xlabel('Number of defaults')
_ = plt.ylabel('Probability')


# Show the plot
plt.show()


# Compute the number of 100-loan simulations with 10 or more defaults: n_lose_money

n_lose_money = np.sum(n_defaults >= 10)
# Compute and print probability of losing money
print('Probability of losing money =', n_lose_money / len(n_defaults))



# Faster solution using the underlying distribution of the process

# Take 10,000 samples out of the binomial distribution: n_defaults
n_defaults = np.random.binomial(100,0.05,size = 10000)

# Compute CDF: x, y
x,y = ecdf(n_defaults)

# Plot the CDF with axis labels
plt.plot(x,y,marker = '.', linestyle = 'none')
_ = plt.xlabel('Number of defaults')
_ = plt.ylabel('Probability')

# Show the plot
plt.show()




# Plotting the Binomial PMF as Hist


# Compute bin edges: bins
bins = np.arange(min(n_defaults), max(n_defaults) + 1.5) - 0.5

# Generate histogram
plt.hist(n_defaults, bins = bins,normed=True)

# Set margins
plt.margins(0.02)

# Label axes

_ = plt.xlabel('Number of defaults')
_ = plt.ylabel('Probability')

# Show the plot
plt.show()





# Question: what is the relation between poisson and binomial distributions

# the Poisson distribution is a limit of the Binomial distribution for rare events
# i.e : large n & small p

# Draw 10,000 samples out of Poisson distribution: samples_poisson

samples_poisson = np.random.poisson(10,size = 10000)
# Print the mean and standard deviation
print('Poisson:     ', np.mean(samples_poisson),
                       np.std(samples_poisson))

# Specify values of n and p to consider for Binomial: n, p
n = [20,100,1000]
p = [0.5,0.1,0.01]

# Draw 10,000 samples for each n,p pair: samples_binomial
for i in range(3):
    samples_binomial = np.random.binomial(n[i],p[i],size = 10000)

    # Print results
    print('n =', n[i], 'Binom:', np.mean(samples_binomial),
                                 np.std(samples_binomial))



#### Question
# 1990 and 2015 featured the most no-hitters of any season of baseball
# (there were seven). Given that there are on average 251/115 no-hitters per season,
# what is the probability of having seven or more in a season?    
    
# Draw 10,000 samples out of Poisson distribution: n_nohitters
n_nohitters = np.random.poisson(251/115, size = 10000)

# Compute number of samples that are seven or greater: n_large
n_large = np.sum(n_nohitters >= 7)

# Compute probability of getting seven or more: p_large
p_large = n_large / 10000

# Print the result
print('Probability of seven or more no-hitters:', p_large)



## The normal Distribution


### Question
# Are the Belmont Stakes results Normally distributed?

# Compute mean and standard deviation: mu, sigma

mu = np.mean(belmont_no_outliers)
sigma = np.std(belmont_no_outliers)

# Sample out of a normal distribution with this mu and sigma: samples
samples = np.random.normal(mu,sigma, size = 10000)

# Get the CDF of the samples and of the data
x,y = ecdf(belmont_no_outliers)
x_theor, y_theor = ecdf(samples)

# Plot the CDFs and show the plot
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('Belmont winning time (sec.)')
_ = plt.ylabel('CDF')
plt.show()






## The Exponential Distribution
# The waiting time between arrivals of a poisson process is Exponentially distributed

# Question

# Now, you'll use your sampling function to compute 
# the waiting time to observe a no-hitter and hitting of the cycle

def successive_poisson(tau1, tau2, size=1):
    # Draw samples out of first exponential distribution: t1
    t1 = np.random.exponential(tau1, size = size)

    # Draw samples out of second exponential distribution: t2
    t2 = np.random.exponential(tau2, size = size)

    return t1 + t2



# Draw samples of waiting times: waiting_times
waiting_times = successive_poisson(764 ,715,size = 100000)

# Make the histogram
plt.hist(waiting_times,bins=100, normed=True, histtype='step')


# Label axes
_ = plt.xlabel('Waiting time')
_ = plt.ylabel('Probability')


# Show the plot
plt.show()



