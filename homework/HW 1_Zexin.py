#!/usr/bin/env python
# coding: utf-8

# In this homework, I will use 11 assests data to implement Mean-Variance Optimization.

# Data:
# The time-series data gives monthly returns for the 11 asset classes and a short-term Treasury-bill fund return, ("SHV",) which we consider as the risk-free rate
# 
# The data is provided in total returns
# 
# These are nominal returns (no need for inflation adjustment)
# 
# Model:
# I will use the excess data for this model

# ## Part 2: Summary Statistics

# In[330]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[331]:


def tangency_portfolio(mean, cov_matrix):
    inverted_cov = np.linalg.inv(cov_matrix)
    ones = np.ones(len(cov_matrix.index))
    scaler = ones @ inverted_cov @ mean
    w_tan = 1/scaler * (inverted_cov @ mean)
    return w_tan


# In[332]:


def portfolio_performance(weighted, mean, cov_matrix):
# Compute the mean, volatility, and Sharpe ratio for the tangency portfolio
    port_mu = np.dot(weighted, mean)
    port_sd = np.sqrt(weighted @ cov_matrix @ weighted.T) * np.sqrt(12)
    port_sharp = port_mu/port_sd 
    sums = {"Return":port_mu,
            "Volatility": port_sd,
            "Sharp_ratio": port_sharp}
    return sums


# In[333]:


df1 = pd.read_excel(xls, 'descriptions')
df1.head(11)


# In[334]:


# load data
# There are in total 4 sheets in the excel file. In this model, I only use excess 
# return to build the model.
xls = pd.ExcelFile("/Users/lori/Desktop/multi_asset_etf_data.xlsx")
# df1 = pd.read_excel(xls, 'descriptions')
# df2 = pd.read_excel(xls, 'prices')
# df3 = pd.read_excel(xls, 'total returns')
df4 = pd.read_excel(xls, 'excess returns', parse_dates=True, index_col=0)


# In[335]:


df4.head()


# In[336]:


df4.info() # 173 rows, 12 rows


# In[337]:


# Calculate and display the mean and
# volatility of each asset’s excess return.
mu = df4.mean() * 12
sigma = df4.std() * np.sqrt(12)


# In[338]:


# Find the best and worst Sharpe ratios
sharp_ratio = mu/sigma
sharp_table = pd.DataFrame(sharp_ratio, columns=['Sharp_ratio'])
sharp_table.index = list(df4.columns)
sharp_table = sharp_table.sort_values(by=['Sharp_ratio'])
print(sharp_table)


# We got the best sharpe ratio is SPY which is 0.973245.

# We got the worst sharpe ratio is BWX which is -0.022112.

# ## Descriptive Analysis

# In[339]:


# Calculate the correlation matrix of the returns.
corrM = df4.corr()
corrM


# In[340]:


plt.matshow(corrM)
plt.show()


# The correlation between PSP and SPY is the highest(lighest color) equals to 0.895729. And the lowest correlation (deepest color) is the pair of DBC and IEF which equals to -0.321738.

# In[242]:


# How well have TIPS done in our sample? Have they outperformed domestic bonds? Foreign bonds?



# ## The MV Frontier

# In[341]:


# Compute and display the weights of the tangency portfolios: w_tan
covM = df4.cov()
w_tan = tangency_portfolio(mu, covM)


# In[342]:


wt_table = pd.DataFrame(w_tan, columns=['Tangency Portfolio Weight'])
wt_table.index = list(covM.index)
wt_table = wt_table.sort_values(by=['Tangency Portfolio Weight'])
print(wt_table)


# In[343]:


# Compare with Sharp ratio
print(sharp_table)


# Does not align.

# In[344]:


# Portfolio Performance
portfolio_performance(w_tan, mu, covM)


# ##  TIPS

# In[345]:


# Dropped TIP
df4_new = df4.drop(columns=['TIP'])
mu_new = df4_new.mean() * 12
covM_new = df4_new.cov()


# In[346]:


w_tan_new = tangency_portfolio(mu_new, covM_new)


# In[347]:


wt_table_new = pd.DataFrame(w_tan_new, columns=['Tangency Portfolio Weight'])
wt_table_new.index = list(covM_new.index)
wt_table_new = wt_table_new.sort_values(by=['Tangency Portfolio Weight'])
print(wt_table_new)


# In[361]:


portfolio_performance(w_tan_new, mu_new, covM_new)


# In[363]:


sharp_ratio_new = 0.3862908162901172/0.200110989376447
sharp_ratio_new 


# In[348]:


# The expected excess return to TIPS is adjusted to be 0.0012 higher than what the historic sample shows.
mu_adj = df4.mean()*12
mu_adj['TIP'] = mu_adj['TIP'] + (0.0012*12)
covM_adj = df4.cov()


# In[349]:


w_tan_adj = tangency_portfolio(mu_adj, covM_adj)


# In[351]:


wt_table_adj = pd.DataFrame(w_tan_adj, columns=['Tangency Portfolio Weight'])
wt_table_adj.index = list(covM_adj.index)
wt_table_adj = wt_table_adj.sort_values(by=['Tangency Portfolio Weight'])
print(wt_table_adj)


# In[364]:


portfolio_performance(w_tan_adj, mu_adj, covM_adj)


# In[366]:


sharp_ratio_adj = 0.3289084045946173/0.16194669293062275
print(sharp_ratio_adj )


# We got the highest sharp ratio with TIP adjusted to be 0.0012 higher. And the other two are approximately equal but the one without TIP is slightly low. 

# ## Part 3: Allocations
# 

# ### Eqaul Weight Portfolio

# In[352]:


# set the targted mean excess return is 0.01 per month.
target_mu = 0.01 * 12
w_equal = np.array(len(corrM.index)*[1/len(corrM.index)]) # w_i = 1/11
factor_equal = target_mu/(w_equal.T @ mu) ############
w_equal_adj = w_equal * factor_equal


# In[353]:


portfolio_performance(w_equal_adj, mu, covM)


# ### Risky-Parity

# In[354]:


w_risky =np.asarray(1/(sigma**2))
factor_risky = target_mu/(w_risky.T @ mu)
w_risky_adj = w_risky * factor_risky


# In[355]:


portfolio_performance(w_risky_adj, mu, covM)


# ### Regularized

# In[399]:


# Regularized Covariance Matrix
diag_matrix = np.diag(df4.var())
reg_matrix = (covM + diag_matrix )/2
w_reg = np.linalg.inv(reg_matrix) @ mu
factor_reg = target_mu/(w_reg.T @ mu)
w_reg_adj = w_reg * factor_reg


# In[400]:


portfolio_performance(w_reg_adj, mu, covM)


# ### Tangency

# In[389]:


factor_tan = target_mu/(w_tan.T @ mu)
w_tan_adj = w_tan * factor_tan


# In[390]:


portfolio_performance(w_tan_adj, mu, covM)


# In[ ]:




