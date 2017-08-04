# Amazon_review_helpfulness_prediction
this is my repository for the Amazon Review Helpfulness prediction model project  
_last updated: 8/04/2017_  

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
  
<img src="images/example_of_review.png" width="600" align="middle">  
  
-------------------
  
My question is:  
**Can Machine learning model learn characteristics of good customer reviews and predict helpfulness of reviews?**  
   
To answer this question, I used Amazon review dataset from [Julian McAuley's website](http://jmcauley.ucsd.edu/data/amazon/links.html)  
and built a xgboost ensemble method model that identify each review as highly helpful or not helpful.  
  
The definition of 2 label classes:    
Highly helpful reviews = Amazon reviews with more than 75% of votes being positive (= helpful)  
Not helpful reviews = Amazon reviews with less than 75% of votes being positive  
  
**With high prediction accuracy, I can determine helpful reviews regardless of user votes on Amazon.**  
  
## Results  
  
Dataset used:  
Home & Kitchen  
Training data: # of reviews = 20,000  
Test data: # of reviews = 3,823      
  
#### 1. XGBOOST result
Overall prediction accuracy: 76.72%     
Confusion Matrix:  
  
 |                 |       NOT HELPFUL TRUE       |        HIGHLY HELPFUL TRUE        |  
 |:--------------: | :-------------------:|:-----------------------:|  
 |       NOT HELPFUL PRED     |        1421.0        |           488.0        |  
 |        HIGHLY HELPFUL PRED     |        402.0        |           1512.0        |  
  
Not helpful review prediction rate: 77.95%  
Highly helpful review prediction rate: 75.6%  
  
#### 2. Random Forest result 
Overall prediction accuracy: 76.82%  
Confusion Matrix:  
  
|                 |       NOT HELPFUL TRUE       |        HIGHLY HELPFUL TRUE        |  
|:--------------: | :-------------------:|:-----------------------:|  
|       NOT HELPFUL PRED     |        1492.0        |           555.0        |  
|        HIGHLY HELPFUL PRED     |        331.0        |           1445.0        |  
  
Not helpful review prediction rate: 81.84%  
Highly helpful review prediction rate: 72.25%  
  
-------------------------
  
## Data used:

the original data and data unzipping function were created by Julian McAuley  
  
This dataset contains product reviews and metadata from Amazon,   
including 142.8 million reviews spanning May 1996 - July 2014.   
The dataset includes reviews (ratings, text, helpfulness votes),   
product metadata (descriptions, category information, price, brand, and image features),   
and links (also viewed/also bought graphs).  

For my prediction model listed here, I used Home & kitchen product dataset which contains:  
  1. 4,253,926 reviews  
  2. 436,988 products meta data  
  
I merged meta data and reviews and preprocessed data before running prediction model.  

  
## Preprocessing

#### Data cleaning  
1. Null values on the price feature were filled with average price values for each category.  
2. Null values on the sales ranking value feature were filled with average sales ranks for Home & Kitchen category products.  
  
#### Filters applied to the model  
<img src="images/category_list_homeandkitchen.png" width="150" align="right">   
  
Filters were applied to
1. product with null price values and null category sales ranking values  
2. product that belongs to categories that are less than 5000 when products data is groupby category.  

Left:  Here you are seeing a list of categories that Home & Kitchen products belongs to.  
The number represents total number of Home & Kitchen product in that category.  
As you can see, there are several categories that seems to be not related to  
Home & Kitchen products at all! (for example, Video Games)    
my filter is applied to take out these minor categories where total number is less than 5000.   
  
   
#### Feature engineering:  
There are over 1900 features at the end of preprocessing.  
During feature engineering, I added:  
1. NMF results in percentage (10 features where 10 = number of topics)  
<img src="images/elbowPlot_first12.png" width="200" align="middle">   
  * To select NMF N topics, I used elbow method. However, the dataset did not plot obvious elbow.  
    I decided # of N based on the change in error rate and I found that the change becumes insignificant  
    after 10th topic.  

NMF result:  
<img src="images/resultsofNMF.png" width="500" align="middle">  
values represents correlation between a review (row) and NMF groups.    

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
    1. N estimators = 2000
    2. Learning rates = 0.1
    3. subsample = 0.8
    4. Max Depth = 25
  
2. Random Forest model  
  * parameter:  
    1. N estimators = 1000  
    2. Max Features = 50  
    
-----------------
#### 1. XGBOOST result  
Overall prediction accuracy: 76.72%     
Confusion Matrix:  
  
 |                 |       NOT HELPFUL TRUE       |        HIGHLY HELPFUL TRUE        |  
 |:--------------: | :-------------------:|:-----------------------:|  
 |       NOT HELPFUL PRED     |        1421.0        |           488.0        |  
 |        HIGHLY HELPFUL PRED     |        402.0        |           1512.0        |  
  
Not helpful review prediction rate: 77.95%  
Highly helpful review prediction rate: 75.6%  
  
#### 2. Random Forest result 
Overall prediction accuracy: 76.82%  
Confusion Matrix:  
  
|                 |       NOT HELPFUL TRUE       |        HIGHLY HELPFUL TRUE        |  
|:--------------: | :-------------------:|:-----------------------:|  
|       NOT HELPFUL PRED     |        1492.0        |           555.0        |  
|        HIGHLY HELPFUL PRED     |        331.0        |           1445.0        |  
  
Not helpful review prediction rate: 81.84%  
Highly helpful review prediction rate: 72.25%  
  
-------------------------
　　
## Findings:  
Random Forest model tends to have high accuracy in predicting note helpful reviews compare to highly helpful reviews.  

  
**XGBoost model's top15 most important features:**  


## Other:  
### NMF result for Home & Kitchen products:
Top 20 words found in each NMF groups  
Topic #1:  
machine blender ice use make juicer juice cream bowl mixer clean blade easy food bread time fruit smoothie good dough  
Topic #2:  
coffee cup water maker machine brew filter grind carafe grinder espresso hot ground pot make bean use pour mug taste  
Topic #3:  
vacuum carpet floor clean dyson dirt bag suction hair brush dust cleaner attachment hose use canister pick hoover vac filter  
Topic #4:  
knife blade sharp set sharpen cut edge chef sharpener handle slice steel block use henckel wusthof good steak hand dull  
Topic #5:  
pan stick cook non use heat cookware pot egg set oil iron food grill handle cast skillet clean surface calphalon  
Topic #6:  
mattress bed pillow sleep sheet foam memory pad night topper soft comfortable feel firm cover like smell queen wake good  
Topic #7:  
rice cooker cook pot pressure slow brown cooking steam cup water lid time use warm crock food make steamer minute  
Topic #8:  
unit fan air room water heater filter heat cool run window temperature work low turn quiet use high noise setting  
Topic #9:  
oven toaster toast bread cook pizza convection bake bagel use microwave timer rack slice burn heat setting time door tray  
Topic #10:  
product buy look item amazon use good like make purchase order review work return set say price quality time great  
  
The NMF result shows that NMF splits reviews by type of Home & Kitchen product that user reviewed (topic #1 through #9).  
The Topic #10 seems to be the NMF group for reviews with positive user sentiment and
my models show that reviews that are highly correlated to this topic tend to be highly helpful reviews.   
  

## Resources:  
  
http://jmcauley.ucsd.edu/data/amazon/links.html  
J. McAuley, A. Yang. Addressing Complex and Subjective Product-Related Queries with Customer Reviews. WWW, 2016　
R. He, J. McAuley. Modeling the visual evolution of fashion trends with one-class collaborative filtering. WWW, 2016  
J. McAuley, C. Targett, J. Shi, A. van den Hengel. Image-based recommendations on styles and substitutes. SIGIR, 2015  
