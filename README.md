# Amazon_review_helpfulness_prediction
this is my repository for the Amazon Review Helpfulness prediction model project  
_last updated: 7/26/2017_  

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
This helpfulness is based on user votes. If Amazon users find review helpful, users can simply leave positive votes on that review.  
the review with most positive votes gets placed as "Top Customer Reviews" on the product page by Amazon.  
    
     
My question is:  
**Can Machine learning model learn characteristics of good customer reviews and predict helpfulness of reviews?**  
  
  
To answer this question, I am using Amazon review dataset from [Julian McAuley's website] (http://jmcauley.ucsd.edu/data/amazon/links.html)  
and created a XGboost binary classifier prediction model.  
  
2 label classes for my model:  
Highly helpful reviews = reviews with more than 75% of total votes were positive (positive = voted as helpful review)  
not helpful reviews = reviews with less than 75% of total votes were positive  
  
With high prediction accuracy, I can determine helpful reviews regardless of user votes on Amazon.  
  
## Results  
Dataset used:  
Home & Kitchen  
Prediction accuracy: 74.42%  

**Confusion Matrix**  

|                 |      LOW True       |        HIGH True       |
|:--------------: | :------------------:|:----------------------:|
|  LOW Predicted  |         1571        |            726         |
|  HIGH Predicted |          252        |           1274         |  

LOW prediction rate: 86.18%  
HIGH prediction rate: 63.7%  
  
## Data used:

the original data and data unzipping function was created by Julian McAuley  
  
This dataset contains product reviews and metadata from Amazon,   
including 142.8 million reviews spanning May 1996 - July 2014.   
The dataset includes reviews (ratings, text, helpfulness votes),   
product metadata (descriptions, category information, price, brand, and image features),   
and links (also viewed/also bought graphs).  
  
I merged review dataset and metadata together and preprocessed data to run prediction model.  
  
------------
#### other info:

## MODELS
#### Filters applied to both models  
  
**label:**  
HIGH,LOW

HIGH = Highly helpful reviews  
LOW = not helpful reviews  
-----------------
#### 1. XGBOOST
  
-------------------------
　　
## Findings:  
 
   
**XGBoost model's top10 most important features:**  
　　
## Other:  
  
## Resources:  

http://jmcauley.ucsd.edu/data/amazon/links.html  
J. McAuley, A. Yang. Addressing Complex and Subjective Product-Related Queries with Customer Reviews. WWW, 2016　
R. He, J. McAuley. Modeling the visual evolution of fashion trends with one-class collaborative filtering. WWW, 2016  
J. McAuley, C. Targett, J. Shi, A. van den Hengel. Image-based recommendations on styles and substitutes. SIGIR, 2015  
