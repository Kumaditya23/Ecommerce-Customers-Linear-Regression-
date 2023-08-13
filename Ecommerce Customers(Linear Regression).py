#!/usr/bin/env python
# coding: utf-8

# # Import libray

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split, cross_val_score, KFold,LeaveOneOut, ShuffleSplit, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score


# ### 1. Get Data :-

# In[6]:


df = pd.read_csv("E:\Ecommerce Customers.csv")


# ### Show head of data :

# In[7]:


df.head()


# ### Show column of data :-

# In[8]:


df.columns


# ### Show data types of each columns:-

# In[9]:


df.info()


# ### 2. Descriptive Statistic :-

# ### 2.1 descriptive statistic for numerical column :-

# ### check if NaN in our data set :-

# In[10]:


df.isna().sum()


# ### There is no na value in our data .

# In[11]:


df.describe()


# ### 2.2 Correlation Analysis :-

# In[12]:


# Create correlation matrix :

correlation_matrix = df.corr()

# Show the correlation matrix :-

correlation_matrix


# In[ ]:


### There is a significant relationship between the length of membership 
#   and the yearly amount spent.


# # 2.3 Exploratory Data Analysis :

# ### 2.3.1 Using seaborn to create a jointplot to compare the Time on Website and  Yearly Ampount Spent columns .  Does the correlation make sense ?
# 

# In[15]:


sns.set(style= 'whitegrid')

# Create the joint plot eith all possible arguments 
joint_plot = sns.jointplot(x= 'Time on Website', y='Yearly Amount Spent',
            data= df, kind = 'scatter', height = 7, ratio =6, space=0,
            marginal_kws={'color': 'purple'},palette='viridis', edgecolor='w',
            linewidth= 1.5)

# Calculate the correlation coefficient
correlation_coefficient = df['Time on Website'].corr(df['Yearly Amount Spent'])

# Set plot labels and title
joint_plot.set_axis_labels('Time on Website', 'Yearly Amount Spent',fontsize =14)
joint_plot.fig.suptitle(f'correlation: {correlation_coefficient:.2f}',fontsize=16
                       , y = 1.02)

# Customize the plot aesthetics 

plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
joint_plot.ax_marg_x.set_facecolor('white')
joint_plot.ax_marg_y.set_facecolor('white')
joint_plot.ax_joint.collections[0].set_edgecolor('w')
joint_plot.ax_joint.collections[0].set_linewidth(1.5)
plt.grid(visible=False)
joint_plot.ax_joint.grid(True)

# Show the plot 
plt.show()


# #### 2.3.2 Using seaborn to create a jointplot to compare the Time on App and Yearly Amount Spent columns. Does the correlation make sense ?

# In[32]:


sns.set(style='whitegrid')

# Create the joint with all possible arguments 
joint_plot = sns.jointplot(x='Time on App', y = 'Yearly Amount Spent', data =df
            , kind='scatter', height=7, ratio=6,space=0, 
            marginal_kws={'color':'purple'}, palette='viridis', edgecolor='w',
                            linewidth=1.5)

# Calculate the correlation coefficient
correlation_coefficient = df['Time on App']. corr(df['Yearly Amount Spent'])


# Set plot labels and title
joint_plot.set_axis_labels('Time on App', 'Yearly Amount Spent', fontsize =14)
joint_plot.fig.suptitle(f'Correlation: {correlation_coefficient:.2f}',
                       fontsize=16, y =1.02)

#Customize the plot awsthetics 
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
joint_plot.ax_marg_x.set_facecolor('white')
joint_plot.ax_marg_y.set_facecolor('white')
joint_plot.ax_joint.collections[0].set_edgecolor('w')
joint_plot.ax_joint.collections[0].set_linewidth(1.5)
joint_plot.ax_joint.grid(True)

# Show the plot
plt.show()


# ### 2.3.3 Lets explore these types of relationships across the entire data set. Using pairplot to recreate the plot below.

# In[37]:


# Assuming df is your data frame with thr data 
sns.set(style='whitegrid')


# Create the pairplot with all possible arguments

pair_plot = sns.pairplot(df, palette='set1', diag_kind='kde', markers='o')

# set title for the entire pair plot
pair_plot.fig.suptitle('Pair Plot of DataFrame', fontsize=18, y=1.02)

# Customize the plot aesthetic
pair_plot.map_upper(sns.scatterplot,edgecolor='w',linewidth=0.5)
pair_plot.map_lower(sns.kdeplot, cmap='Blues')
pair_plot.map_diag(sns.histplot, kde=True, color='purple', edgecolor='w',
                   linewidth=0.5)

# Adjust tick font size for x and y axes on individual axes
for ax in pair_plot.axes.flat:
    ax.tick_params(axis='both', labelsize=12)
    
# Add grid lines to all subplots
for ax in pair_plot.axes.flat:
    ax.grid(True, linewidth=0.5,linestyle='--', color='lightgray')
    
# Remove top and right spines from all subplots 
sns.despine(top=True, right=True)

# Supress specific warnings
warnings.filterwarnings('ignore', message=".*palette.*")
warnings.filterwarnings("ignore")

# Show the Plot
plt.show()


# ### 2.3.4 Creating a linear model plot (using implot) of yearly amount spent Vs length of membership

# Pairplot arguments :
# 
# 1- hue : The hue argument is set to 'Avatar', which will color the data points based on the different avatar categories.
# 
# 2- Palette : The palette argument is set to "Set1" , which is a color palette from seaborn used for coloring the data poits.
# 
# 3- markers: The markers argument is set to ['o,'s,'D'], which specifies different markers for the data points corresponding to different avatar categories.
# 
# 4- Ci : The ci arguments is set to 95, which addds a 95% confidence interval around the regression line.
# 
# 5- scatter_kws : The scatter_kws argument is set to {"lw":2} , which adjusts the line width of the regression line .

# In[41]:


sns.set(style ='whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Set a larger figure size 
plt.figure(figsize=(30,15))

# Create a linear model plot (lmplot) with adjusted arguments

lm_plot= sns.lmplot(x='Length of Membership', y = 'Yearly Amount Spent', 
                    data = df, palette='cool', ci=95, scatter_kws={"s":80},
                    line_kws={"lw":2})

# Setv  axis labels and title 
lm_plot.set_axis_labels('Length of Membership', 'Yearly Amount Spent', fontsize= 14)
lm_plot.fig.suptitle('Relationship between Length of Membership and Yearly Amount Spent', fontsize =18, y = 1.03)
                     
                     # Add grid lines to both axes
plt.grid(True , linewidth =0.5, linestyle='--', color = 'lightgray')
                     
                     # Remove top and right spines
sns.despine(top=True, right=True)
                     # Customize the plot aesthetics
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
                     
                     # show the plot
plt.show()


# ### 2.4  Training and Testing Data :

# In[43]:


# step 1 : Define the feature (X) and the target variable (Y)
X= df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]


# Features (input variables)
Y = df['Yearly Amount Spent']   # Target variable (output variable)


# We use train_test_split to split the data into training and testing sets.
# 
# We pass the features (X) and the target variable (Y) to the function as the first two arguments.
# 
# we set the test_size parameter to 0.3 , which means we want to allocate 30% oif the data for testing , and the remaining 70% for training .
# 
# we set the random_state parameter to 42, ensuring that the data will be split consistently every time we run the code.

# In[44]:


# Step 2 : split the data into training and testing sets (70 % for training , 30% for testing)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=42)


# ### We create a Linear Regression model using LinearRegression.

# ### We use the fit method to train the model using te training features ( X_train) and the corresponding target variable (Y-train)

# In[46]:


# Step 3 : Train the linear regression model on the training set 

model = LinearRegression()
model.fit(X_train, Y_train)


# ### We use the trained model to make prediction on the testing features (X_test).
# 
# #### The predicted valuesare stored in 'Y_pred'.

# In[47]:


# Step 4 : Test the model on the the testing set and make predictions

Y_pred = model.predict(X_test)


# ### We use the mean_squared_error function to calculate the mean squared error between the actual target values (Y_test) and the predicted values (Y_pred).
# 
# #### We use the r2_score function to calculate the R-squared (coefficient of determination) between Y-test and Y_pred, which indicated how well the model fits the data.
# 
# #### We print the mean squared error and the R-squared score to evaluate the model's performance .

# In[51]:


# Step 5 : Evaluate the model's performance 
mse = mean_squared_error(Y_test, Y_pred)
r_squared = r2_score(Y_test, Y_pred)
print("Mean Squared Error:",mse)
print("R-squared:", r_squared)


# ### 2.4.1  Printing out the coefficients of the model :

# In[53]:


coefficients_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
intercept_df = pd.DataFrame({'Intercept': [model.intercept_]})
print(coefficients_df)
print(intercept_df)


# ### 2.4.2 ** Predection Vizualization **

# In[54]:


sns.set(style = 'whitegrid')
sns.set_context("notebook", font_scale=1.2)


# Create a scatter plot with adjusted arguments 
plt.figure(figsize=(10,8))
sns.scatterplot(x=Y_test, y = Y_pred, alpha= 0.7, color ='blue',edgecolor='k',linewidth=1.5)

# Add a regression line to better visualize the relationship
sns.regplot(x=Y_test, y=Y_pred, scatter = False, color='red', line_kws={"linewidth": 2})
 
# Set axis labels and title
plt.xlabel('Actual Yearly Amount Spent' , fontsize =14)
plt.ylabel ('Predicted Yearly Amount Spent' , fontsize=14)
plt.title('Actual vs. Predicted Yearly Amount Spent', fontsize =16)

# Add grid lines 
plt.grid(True, linewidth =0.5, linestyle='--', color='lightgray')

# Remove top and right spines
sns.despine(top=True, right =True)

# Customize the plot aesthetics
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.show()


# ### 2.5 Evaluating the Model :

# In[57]:


from tabulate import tabulate


# Evaluate the model's performance 
mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
r_squared = r2_score(Y_test, Y_pred)
evs = explained_variance_score(Y_test, Y_pred)

# create a dataframe to hold the metrics

metrics_df = pd.DataFrame({
    'Metric' : ['Mean Squared Error', 'Mean Absolute Error', 'R-squared', 'Explained Variance Score'],
    'Value': [mse,mae, r_squared, evs]
})

# Convert the DataFrame to a tabular format

table = tabulate(metrics_df, headers='keys', tablefmt='fancy_grid', showindex= False)


# Display the table
print(table)


# 1. Mean Squared Error (MSE): The mean squared error measures the avergae squared difference between the actual target values and predicted values. In this case, the MSE is approximately 103.92. Lower values of MSE indicate that the model's prediction are closer to the actual values , indicating a better fit. Since the MSE is relatively small, it suggest that the model's prediction have a small overall error.
# 
# 

# 2. Mean Absolute Error : (MAE) The Mean absolute error computes the average absolute difference between the actual target values and the predicted values. The MAE is approximately 8.43, similar to MSE, lower values of MAE indicate that the model's pedictions are closer to the actual values. The MAE being relatively small suggest that the model's prediction are on average about 8.43 units away from the actual values.
# 
# 

# 3. R-Squared (R Square) : The R-squared value , also known as the coefficient of determination , measures the proportion of the variance in the target variable (Yearly Amount Spent) that is explained by the linear regression model . The R square value ranges from 0 to 1, with higher values indicating a better fit of the model to the data . In this case, the R square value is approximately 0.981 , which means that around 98.1% of the variance in the target variable is explained by the model. A high R square value indiactes that the models predictions are closely related to the actual values. 

# 4. Explained Variance Score : The explained variance score is another metric that indiacate the proportion of the variance in the target variable that is expalined by the model.The explained variance score also ranges from 0 to 1, and higher values indicate a better fit. Here , the explained variance score is approximately 0.981, which is consistent with the high R Square  value . It reinforces the notion that the model expalins about 98.1% of the variance in the Yearly Amount Spent.

# ### 2.6 Residuals 

# In[59]:


residuals = Y_test - Y_pred

# Set Seaborn style and font scale 
sns.set(style ='whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Create the histogram plot with adjusted arguments 
plt.figure(figsize=(10,6))
sns.histplot(residuals, bins=50, kde=True, color='skyblue', edgecolor='black', linewidth=1)

# Add a Vertical line at x=0 to indicate the mean of residuals
plt.axvline(x=0, color='red', linestyle='--')

# Set axis labels and title
plt.xlabel('Residuals', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.title('Distribution of Residuals', fontsize=16)

# Add grid lines to the plot
plt.grid(True, linewidth=0.5, linestyle ='--', color='lightgray')

# Remove top and right spines 
sns.despine(top=True, right=True)

# Show the plot
plt.show()


# ### 2.7 K-Fold Cross- Validation :

# K-Fold Cross Validation is a popular technique used for evaluating the performance of a machine learning model on a dataset. It involves dividing the dataset into K subsets or "folds" of approximately equal size. The process can be summarized in the following steps:

# In[61]:


# Perform K-Fold Cross - Validation (K=5)
kfold = KFold(n_splits=5)
cv_scores_kfold = cross_val_score(model, X, Y, cv=kfold)

# Print the cross-validation scores :
print("K-Fold Cross-Validation Scores:",cv_scores_kfold)
print("Mean CV Score (K-Fold):", cv_scores_kfold.mean())


# ### 2.8 Shuffle Split (Randomized Cross - Validation) : 

# Performing Shuffle Split, also known as Randomized Cross Validation, is another technique used for model eveluation and validation. Unlike K-Fold Cross- Validation, Shuffle Split randomly shuffles the data and split it into a specified number of train-test sets, without any overlap between the sets. its is especially useful when dealing with with large datasets for allows for multiple train-test splits with different random samples. 

# In[66]:


# Perform Shuffle Split ( Randomized Cross - Validation )

shuffle_split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
cv_scores_shuffle_split = cross_val_score(model, X ,Y, cv=shuffle_split)

print("Shuffle Split Cross-Validation Scores:", cv_scores_shuffle_split)
print("Mean CV Score (Shuffle Split):", cv_scores_shuffle_split.mean())


# In[ ]:





# In[ ]:




