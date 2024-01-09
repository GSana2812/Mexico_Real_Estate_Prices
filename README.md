## Mexico City House Prices

This project intends to predict the house prices in Mexico, given several features like Area, Location and Coordinates.
It's a regression task, where the target variable is given as a function y = f(x) of the features mentioned. Initially
in the dataset, there were way more features, but the analysis of data brought us to conclusions that several features had to 
be dropped, for several reasons:

1. High or Low Cardinality Features (Features which contain a lot of unique values or not)
2. High Leakage Features (Features that are similar in purpose or what they indicate with the target variable)
3. Multicollinearity (Features which are highly correlated to each other)

We created pipelines for either the data preprocessing also the modelling process. The preprocessor pipeline and the 
model pipeline are saved as pickle files, to be reused in inference level, where we take an example, and based on that
we can perform fresh predictions.

To be done later:
- We can either use SVD (Dimensionality Reduction Algorithm) or not, so we can have a look about the feature importances,
or features that have a high importance in our predictions.
- We can create a  local web app, using streamlit library for the front-end. 
- We can connect the ml-model with an API using FastAPI.
- We can use tools like MLFlow, to track and monitor our model's performance and progress.

