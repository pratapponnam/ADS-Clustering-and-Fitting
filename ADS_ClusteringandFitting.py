# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 19:58:07 2024

@author: anude
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import requests
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler
from scipy.optimize import curve_fit
import scipy.stats as ss


def download_data(url):
    """
    Function to downlaod the data from the url provided and 
    returns the data frame which has the downloaded data 

    """
    while(1):
        try:
        #fetch the data from url 
            response = requests.get(url)
            #check if we got a response from the url
            if response.status_code == 200:
            # Save the content of the response to a local CSV file
                with open("downloaded_data.csv", "wb") as f:
                    f.write(response.content)
                break
            else:
                print("Failed to download CSV file. Status code:", response.status_code)
        #if exception is raised,continuing the loop
        except requests.exceptions.HTTPError :
            continue
        except requests.exceptions.ConnectionError :
            continue
        except requests.exceptions.Timeout :
            continue
        except requests.exceptions.RequestException :
            continue
    #moving data to dataframe from the downladed data
    df = pd.read_csv("downloaded_data.csv")
    return df


def PB_Process_Data(df):
    df = df.drop(['Country Name','Country Code','Series Code'],axis =1)
    df.set_index('Series Name', inplace=True)
    df= df.T
    df.index = df.index.str.split().str.get(0)
    df.columns.name = ''
    df = df.rename(columns=
        {"GDP per capita (current US$)":
                                    "GDP per capita",
         "Employment to population ratio, 15+, total (%) (modeled ILO estimate)": 
                                    "Employment to Pop ratio",
         "Population density (people per sq. km of land area)":
                                    "Population density",
         "Foreign direct investment, net inflows (% of GDP)"
                                   :"Foreign direct investment, net inflows",
         "Foreign direct investment, net outflows (% of GDP)"
                                   :"Foreign direct investment, net outflows",
         "Expense (% of GDP)"
                                   :"Expense",
         "Exports of goods and services (% of GDP)"
                                   :"Exports of Goods and Services"})
    return df
 

def plot_Line_Graph(*df):
    """
    Defining a function to create a Line plot 
    to identify the relation between Forest Area and Population Density
    """
    plt.figure(figsize=(7, 5))
    #plotting the global temp change data
    cmap = ['red','skyblue','orange','green']
    
    for i, df in enumerate(df):
        sns.lineplot( data = df['Foreign direct investment, net inflows'], color = cmap[i],marker ='o',label = x[i])
        
    
    #set the titles, labels, limits and grid values
    plt.title('Relation between Foreign Investments across countries')
    plt.xlabel('Years')
    plt.ylabel('Foreign Investments')
    plt.grid(True)
    #plt.xscale(5)
    #plt.xlim(1990,2020)
    plt.xticks(rotation=45)
    # Save the plot as Linegraph.png
    plt.savefig('Linegraph.png')
    # Show the plot
    plt.show()
    return

    
def PB_Plot_Histogram(*df):
    """
    Defining a function to create a histogram 
    to understand the frequency of temperature 
    anomalies for different zones across the years
    """
    plt.figure(figsize=(7, 5))
    
    # plotting an overlapped histogram to observe the frequency of the temp.
    for i, df in enumerate(df):
        sns.histplot(df['Employment to Pop ratio'], kde=True, stat="density",bins=10, 
                     linewidth=0, label=x[i],alpha=0.5)
    
    #set the titles, legend, labels and grid 
    plt.title('Distribution of Employment to Pop ratio')
    plt.xlabel('Employment to Population ratio')
    plt.ylabel('Frequency')
    #plt.xlim(0.4,1.4)
    plt.grid(axis='y')
    plt.legend()
    # Save the plot as histogram.png
    plt.savefig('histogram.png')
    # Show the plot
    plt.show()
    return



def plot_heatmap_correlation(df):
    """
    Defining a function to create a Heatmap to plot
    correlation between different factors 
    """
    # creating an upper triangular data with 0's and 1's for masking
    mask = np.triu(np.ones_like(df.corr()))
    plt.figure(figsize=(7, 5))
    # plotting a heatmap
    sns.heatmap(df.corr(), annot=True, mask=mask,
                cmap='BuPu', linewidths=.5)
    
    #set the title
    plt.title('Correlation between various factors')
    # Save the plot as Heatmap.png
    plt.savefig('Heatmap.png')
    # Show the plot
    plt.show()
    return


def one_silhoutte_inertia(n, xy):
    """ 
    Calculates the silhoutte score and WCSS for n clusters 
    """
    # set up the clusterer with the number of expected clusters
    kmeans = KMeans(n_clusters=n, n_init=20)
    # Fit the data
    kmeans.fit(xy)
    labels = kmeans.labels_
    
    # calculate the silhoutte score
    score = silhouette_score(xy, labels)
    inertia = kmeans.inertia_

    return score, inertia


def PB_Plot_Fitted_GDP_Expense(labels, xy, xkmeans, ykmeans, centre_labels):
    """
    Plots clustered data as a scatter plot with determined centres shown
    """
    colours = plt.cm.Set1(np.linspace(0, 1, len(np.unique(labels))))
    cmap = ListedColormap(colours)
    
    fig, ax = plt.subplots(dpi=144)
    s = ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap=cmap, marker='o', label='Data')

    ax.scatter(xkmeans, ykmeans, c=centre_labels, cmap=cmap, marker='x', s=100, label='Estimated Centres')

    cbar = fig.colorbar(s, ax=ax)
    cbar.set_ticks(np.unique(labels))
    ax.legend()
    ax.set_xlabel('GDP per capita')
    ax.set_ylabel('Expense')
    plt.show()
    return


def PB_Plot_Elbow_Method(min_k, max_k, wcss, best_n):
    """
    Plots the elbow method between min_k and max_k
    """
    fig, ax = plt.subplots(dpi=144)
    ax.plot(range(min_k, max_k + 1), wcss, 'kx-')
    ax.scatter(best_n, wcss[best_n-min_k], marker='o', color='red', facecolors='none', s=50)
    ax.set_xlabel('k')
    ax.set_xlim(min_k, max_k)
    ax.set_ylabel('WCSS')
    plt.show()
    return


def logistic(t, n0, g, t0):
    """
    Calculates the logistic function with scale factor n0 and growth rate g
    """
    
    f = n0 / (1 + np.exp(-g*(t - t0)))
    
    return f


#storing the filelinks in variables
url1 = 'https://github.com/pratapponnam/ADS-Clustering-and-Fitting/blob/main/GDP_data.csv?raw=True'


df_GDP =  download_data(url1)
df_France = df_GDP[df_GDP['Country Name'].isin(['France'])]
df_Swedon = df_GDP[df_GDP['Country Name'].isin(['Sweden'])]
df_Italy = df_GDP[df_GDP['Country Name'].isin(['Italy'])]
df_Bulgaria = df_GDP[df_GDP['Country Name'].isin(['Bulgaria'])]


df_France = PB_Process_Data(df_France)
df_Swedon = PB_Process_Data(df_Swedon)
df_Bulgaria = PB_Process_Data(df_Bulgaria)
df_Italy = PB_Process_Data(df_Italy)


x = ['France','Italy','Swedon','Bulgaria']

plot_Line_Graph(df_France,df_Italy,df_Swedon,df_Bulgaria)

PB_Plot_Histogram(df_France,df_Italy,df_Swedon,df_Bulgaria)

df= pd.concat([df_France,df_Italy,df_Swedon,df_Bulgaria])
plot_heatmap_correlation(df)

df_clust = df[['GDP per capita','Expense']].copy()
scaler = RobustScaler()
norm = scaler.fit_transform(df_clust)



colours = plt.cm.Set1(np.linspace(0, 1, 5))
cmap = ListedColormap(colours)

wcss = []
best_n, best_score = None, -np.inf
for n in range(2, 11):  # 2 to 10 clusters
    score, inertia = one_silhoutte_inertia(n, norm)
    wcss.append(inertia)
    if score > best_score:
        best_n = n
        best_score = score
    #print(f"{n:2g} clusters silhoutte score = {score:0.2f}")

print(f"Best number of clusters = {best_n:2g}")
PB_Plot_Elbow_Method(2, 10, wcss, best_n)


inv_norm = scaler.inverse_transform(norm)  # this is important for plotting data accurately
for k in range(3, 5):
    kmeans = KMeans(n_clusters=k, n_init=20)
    kmeans.fit(norm)     # fit done on x,y pairs
    labels = kmeans.labels_
    
    # the estimated cluster centres
    cen = scaler.inverse_transform(kmeans.cluster_centers_)
    xkmeans = cen[:, 0]
    ykmeans = cen[:, 1]
    cenlabels = kmeans.predict(kmeans.cluster_centers_)
    PB_Plot_Fitted_GDP_Expense(labels, inv_norm, xkmeans, ykmeans, cenlabels)
    



df_Bulgaria_fit = df_Bulgaria[['GDP per capita']]
df_Bulgaria_fit.index = df_Bulgaria_fit.index.astype(int)


# let's normalise the time frame, quite important for exponentials
numeric_index = (df_Bulgaria_fit.index - 2013).values

# give some initial guesses of N0 and growth
p, cov = curve_fit(logistic, numeric_index, df_Bulgaria_fit['GDP per capita'],
                  p0=(1.2e12, 0.03, 10))

# get uncertainties on each parameter
sigma = np.sqrt(np.diag(cov))

print(f"N0 = {p[0]:g} +/- {sigma[0]:g}")
print(f"g = {p[1]:.2f} +/- {sigma[1]:.2f}")
print(f"t0 = {p[2] + 1990:.2f} +/- {sigma[2]:.2f}")


fig, ax = plt.subplots(dpi=144)
df_Bulgaria_fit= df_Bulgaria_fit.assign(Logistic_Fit = logistic(numeric_index, *p))
df_Bulgaria_fit.plot(ax=ax, ylabel='GDP £')
#ax.set_yscale('log')
plt.show()

numeric_index = (df_Bulgaria_fit.index - 2005).values
p, cov = curve_fit(logistic, numeric_index, df_Bulgaria_fit['GDP per capita'],
                  p0=(1.2e12, 0.03, 10))
gdp_2050 = logistic(2050 - 2005, *p)  # remember to subtract the 1990 as we did when 'training'
print(f"GDP in 2050: {gdp_2050:g}")

# take 1000 normal random samples for each parameter
sample_params = ss.multivariate_normal.rvs(mean=p, cov=cov, size=1000)

# standard deviation of all possible parameter sampling
gdp_unc_2050 = np.std(logistic(2050 - 2005, *sample_params.T))  # note the transpose

print(f"GDP in 2050: {gdp_2050:g} +/- {gdp_unc_2050:g}")

fig, ax = plt.subplots(dpi=144)
# create array of values within data, and beyond
time_predictions = np.arange(1990, 2050, 1)
# determine predictions for each of those times
gdp_predictions = logistic(time_predictions - 2005, *p)
# determine uncertainty at each prediction
gdp_uncertainties = [np.std(logistic(future_time - 2005, *sample_params.T)) for future_time in time_predictions]

ax.plot(df_Bulgaria_fit.index, df_Bulgaria_fit['GDP per capita'], 'b-', label='Data')
ax.plot(time_predictions, gdp_predictions, 'k-', label='Logistic Fit')
ax.fill_between(time_predictions, gdp_predictions - gdp_uncertainties, gdp_predictions + gdp_uncertainties, 
                color='gray', alpha=0.5)
ax.set_xlabel('Time')
ax.set_ylabel('GDP')
ax.grid()
ax.legend()
plt.show()