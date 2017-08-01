# Amazon_review_helpfulness_prediction
this is my repository for the Amazon Review Helpfulness prediction model project  
_last updated: 8/01/2017_  

## Repo Instructions

python folder:  
+ contains 1 python files  
   * data_prep_new.py contains all function used for this project  
   
Procedure.ipynb  
+ Jupyter notebook that runs python codes above. note that there is no data stored in this repo.

images  
+ jpeg images used in this readme markdown file.  

## Introduction:  

When you shop in Amazon, Do you notice that product reviews are in the order of helpfulness of reviews?  
This helpfulness is based on user's votes. If Amazon users find a product review helpful,  
users can simply leave positive votes on that review.  
the review with most positive votes gets placed as "Top Customer Reviews" on the product page by Amazon.  
  
Here is an example from actual Amazon website:  
  
-------------------
![example1](images/example_of_review.png =250x)  
-------------------
  
My question is:  
**Can Machine learning model learn characteristics of good customer reviews and predict helpfulness of reviews?**  
   
To answer this question, I used Amazon review dataset from [Julian McAuley's website](http://jmcauley.ucsd.edu/data/amazon/links.html)  
and created a XGboost binary classifier prediction model.  
  
2 label classes for my model:  
Highly helpful reviews = reviews with more than 75% of total votes were positive (positive = voted as helpful review)  
not helpful reviews = reviews with less than 75% of total votes were positive  
  
**With high prediction accuracy, I can determine helpful reviews regardless of user votes on Amazon.**  
  
## Results  
  
Dataset used:  
Home & Kitchen  
Training data: # of reviews = 10,000  
Test data: # of reviews = 3,823  
Prediction accuracy: 76.01%   

**Confusion Matrix**  

 |                 |       NOT HELPFUL TRUE       |        HIGHLY HELPFUL TRUE        |  
 |:--------------: | :-------------------:|:-----------------------:|  
 |       NOT HELPFUL PRED     |        1359.0        |           453.0        |  
 |        HIGHLY HELPFUL PRED     |        464.0        |           1547.0        |  

Not helpful prediction rate: 74.55%  
HIGH helpful prediction rate: 77.35%  
  
## Data used:

the original data and data unzipping function were created by Julian McAuley  
  
This dataset contains product reviews and metadata from Amazon,   
including 142.8 million reviews spanning May 1996 - July 2014.   
The dataset includes reviews (ratings, text, helpfulness votes),   
product metadata (descriptions, category information, price, brand, and image features),   
and links (also viewed/also bought graphs).  
  
I merged review dataset and metadata together and preprocessed data to run prediction model.  
  
------------
#### other info:


#### Data cleaning  
1. Null values on the price feature were filled with average price values for each category.  
2. Null values on the sales ranking value feature were filled with average sales ranks for Home & Kitchen category products.  
  
#### Filters applied to the model  
Filters were applied to 
1. product with null price values and null category sales ranking values  
2. product that belongs to categories that are less than 5000 when products data is groupby category.*

![example](images/category_list_homeandkitchen.png =250x)  
Here you are seeing a list of categories that Home & Kitchen products belongs to.  
As you can see, there are categories that are not related to Home & Kitchen product at all (for example, Video Games).  
my filter is applied to take out these minor categories.   

#### Feature engineering:  
There are over 1900 features at the end of preprocessing.  
During feature engineering, I added:  
1. NMF results in percentage (10 features where 10 = number of topics)  
2. Tfidf terms (1000 features where 1000 = number of tfidf terms)  
3. name of sub categories (100 to 1000 features depending on the main category and filtering parameters)  
4. review text length, rating that reviewer gave to a product, price, categories,  
  sales ranking, percentage of review helpfulness (label) etc.



#### Label:   
HIGH,LOW

HIGH = Highly helpful reviews  
LOW = not helpful reviews  

## MODELS 
1. XGboost model  
  * parameter:
    1. N estimators = 4000
    2. Learning rates = 0.15
    3. subsample = 0.8
    4. Max Depth = 6
  
2. Random Forest model  
  * parameter:  
    1. N estimators = 1000  
    2. Max Features = 50  
    
-----------------
#### 1. XGBOOST result


#### 2. Random Forest result 
Overall prediction accuracy: 76.82%  
Confusion Matrix:  
  
|                 |       NOT HELPFUL TRUE       |        HIGHLY HELPFUL TRUE        |  
|:--------------: | :-------------------:|:-----------------------:|  
|       NOT HELPFUL PRED     |        1492.0        |           555.0        |  
|        HIGHLY HELPFUL PRED     |        331.0        |           1445.0        |  
  
Not helpful review prediction rate: 81.84%  
highly helpful review prediction rate: 72.25%  
  
-------------------------
　　
## Findings:  
Random Forest model tends to have high accuracy in predicting note helpful reviews compare to highly helpful reviews.  

  
**XGBoost model's top15 most important features:**  
　　
## Other:  
  
## Resources:  
  
http://jmcauley.ucsd.edu/data/amazon/links.html  
J. McAuley, A. Yang. Addressing Complex and Subjective Product-Related Queries with Customer Reviews. WWW, 2016　
R. He, J. McAuley. Modeling the visual evolution of fashion trends with one-class collaborative filtering. WWW, 2016  
J. McAuley, C. Targett, J. Shi, A. van den Hengel. Image-based recommendations on styles and substitutes. SIGIR, 2015  
