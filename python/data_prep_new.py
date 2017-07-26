from __future__ import unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
import gzip
import spacy
np.random.seed(32113)
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF
from nltk.corpus import stopwords
from spacy.en import English
import string
parser = English()
import xgboost as XGB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


##############################################################################
#                      Stop Words and Lemmetizer setting                     #
##############################################################################

#this STOPLIST is used as a parameter for Sklearn Tf-idf Vectorizer.
STOPLIST = list(set(stopwords.words('english') + ["n't", "'s", "'m", "ca"] + \
                list(ENGLISH_STOP_WORDS))) +\
                 " ".join(string.punctuation).split(" ") + \
                 ["-----", "---", "...", "..", "....", "", " ", "\n", "\n\n"]
nlp = spacy.load('en')

def lemma(doc):
    '''
    FUNCTION:
    This function reads a document and run spaCy lemmetizer on the document.
    This lemmetizer will be used as a parameter for Sklearn Tf-idf Vectorizer.

    INPUT:
    doc = document ('string')

    OUTPUT:
    lemmetized tokens of a document
    '''
    doc = nlp(doc)
    return [n.lemma_.lower().strip() if n.lemma_ != "-PRON-" \
                                                    else n.lower_ for n in doc]

##############################################################################
#                             Aggregated functions                           #
##############################################################################

def df_prep1(data_path,meta_path,topic,filter_num):
    '''
    FUNCTION:
    This is a function reads a document and run spaCy lemmetizer on the document
    This lemmetizer will be used as a parameter for Sklearn Tf-idf Vectorizer

    INPUT:
    path = path of json.gz file (string)
    meta_path = math of meta json.gz file (string)
    category = string that indicates category of json.gz file (string)

    OUTPUT:
    merged_df = pandas dataframe contains features needed to run xgboost.
    tfidf_model = tfidf model created with list of reviews
    tfidf_matrix = tfidf matrix created with list of reviews
    '''
    start_time = time.time()

    df = Data_prep1(data_path)
    meta = getDF(meta_path)
    merged_df = add_meta_info(df,meta,topic,filter_num)
    merged_df = price_adjustor(merged_df)
    merged_df['price']=merged_df.price.round(2)
    merged_df = merged_df.drop(['rank_keys'],axis = 1)

    print("--- %s seconds ---" % (time.time() - start_time))
    return merged_df

def df_prep_train1(merged_df,filename,tfidf_set=[2,0.95,10000,'l2']):

    start_time = time.time()

    tfidf_matrix, tfidf_model = Data_prep2_tfidf(merged_df,tfidf_set)
    with open(r"./{}_tfidf_model.pickle".format(filename), "wb") as output_file:
        pickle.dump(tfidf_model, output_file)
    with open(r"./{}_tfidf_matrix.pickle".format(filename), "wb") as output_file:
        pickle.dump(tfidf_matrix, output_file)

    print("--- %s seconds ---" % (time.time() - start_time))
    return tfidf_matrix, tfidf_model


def df_prep_train2(df, filename, tfidf_matrix, tfidf_model, max_feature = 1000, NMF_list = [6,'cd',0.1,0.5], tfidf_set=[2,0.95,10000,'l2']):

    start_time = time.time()

    df,NMF_model = Data_prep2_NMF(df, tfidf_matrix, tfidf_model, NMF_list)
    df,tfidf_feat_model = Data_prep3(df, max_feature, tfidf_set)
    df.to_pickle("./preped_{}_max_feature{}.pkl".format(filename, max_feature))
    with open(r"./{}_NMF_model.pickle".format(filename), "wb") as output_file:
        pickle.dump(NMF_model, output_file)
    with open(r"./{}_tfidf_feat_model.pickle".format(filename), "wb") as output_file:
        pickle.dump(tfidf_feat_model, output_file)
    print("--- %s seconds ---" % (time.time() - start_time))
    return df, NMF_model, tfidf_feat_model


def df_prep_test(df_test, tfidf_model,NMF_model, tfidf_feat_model,filename):

    start_time = time.time()

    tfidf_matrix_test = tfidf_model.transform(df_test)
    NMFresults_test = NMF_model.transform(tfidf_matrix_test)
    NMF_result_df = pd.DataFrame(NMFresults_test)
    df_NMF_test = pd.concat([df_test,NMF_result_df],axis = 1)
    df_NMF_test = df_NMF_test.drop(['reviewerName','num_of_helpful_review',], \
                                                                    axis = 1)
    df_NMF_test = df_NMF_test.rename(columns = {0:'percent_GROUP_1', 1:'percent_GROUP_2',\
                            2:'percent_GROUP_3', 3:'percent_GROUP_4', \
                            4:'percent_GROUP_5', 5:'percent_GROUP_6', \
                            6:'percent_GROUP_7', 7:'percent_GROUP_8', \
                            8:'percent_GROUP_9', 9:'percent_GROUP_10',\
                            10:'percent_GROUP_11', 11:'percent_GROUP_12', \
                            12:'percent_GROUP_13', 13:'percent_GROUP_14', \
                            14:'percent_GROUP_15', 15:'percent_GROUP_16', \
                            16:'percent_GROUP_17', 17:'percent_GROUP_18', \
                            18:'percent_GROUP_19', 19:'percent_GROUP_20'})
    docs = df_NMF_test['reviewText']
    docs_np = np.array(docs).astype('U')
    docs_np = docs_np.tolist()
    tfidf_feat_test = tfidf_feat_model.transform(docs_np)
    vocab_dict = {}
    for k,v in enumerate(tfidf_feat_model.vocabulary_):
        vocab_dict[k]=v
    tfidf_mf = tfidf_feat_test.toarray()
    tfidf_df = pd.DataFrame(tfidf_mf)
    tfidf_df= tfidf_df.rename(columns =vocab_dict)

    df_NMF_test = pd.concat([df_NMF_test,tfidf_df],axis = 1)
    df_NMF_test = df_NMF_test.drop(['reviewText','summary', 'reviewerID', 'helpful_total_review'],axis = 1)
    df_NMF_test.to_pickle("./preped_{}_df_test.pkl".format(filename))
    print("--- %s seconds ---" % (time.time() - start_time))
    return df_NMF_test

##############################################################################
#                               importing Data                               #
##############################################################################

'''
the original data and following 2 functions
were created by Julian McAuley

Julian McAuley
http://jmcauley.ucsd.edu/data/amazon/links.html
'''
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

##############################################################################
#                             Data preparation                               #
##############################################################################

def Data_prep1(path):
    '''
    FUNCTION:
    reads path of review dataset (json.gz format) and add few new features.
    1. create 'helpful_percent' feature. This feature is calculated based on
        a: number of positive votes of a review
        b: total number of votes of a review
    2. create 'text length' feature. This is a word count of each review.
    returns Pandas Dataframe

    INPUT:
    path = path of your review dataset

    OUTPUT:
    Pandas dataframes with additional features explained above
    '''

    game_df = getDF(path)

    #creating new features from helpful feature.
    helpful = np.array(game_df['helpful'])
    helpful = helpful.reshape(len(game_df),1)
    helpful_num1 = np.zeros((len(helpful),1))
    helpful_num2 = np.zeros((len(helpful),1))
    for i in range(len(helpful)):
        helpful_num1[i] = helpful[i][0][0]
        helpful_num2[i] = helpful[i][0][1]
    game_df['helpful_total_review'] = helpful_num2
    game_df['num_of_helpful_review'] = helpful_num1
    # filter data by total number of votes on a review: votes >20
    new_df = game_df[game_df['helpful_total_review']>20]

    # calculating percentage of positive votes
    new_df = new_df.drop(['helpful','unixReviewTime','reviewTime'],axis = 1)
    new_df['helpful_percent'] = (new_df['num_of_helpful_review']/ \
                                new_df['helpful_total_review']).round(2)

    #resetting index to int
    new_df.index = range(len(new_df))

    #add new feature 'text_length' (= word counts of each review)
    length = np.zeros((len(new_df),1))
    for i in new_df.index:
        length[i] = int(len(new_df['reviewText'][i]))
    new_df['text_length']=length
    return new_df

def Data_prep2_tfidf(new_df,tfidf_set=[2,0.95,10000,'l2']):
    '''
    FUNCTION:
    Using Pandas dataframe that contains reviews as a feature,
    this function will return tfidf matrix and tfidf model.

    INPUT:
    new_df = Pandas dataframe. should contain "reviewText" feature
    tfidf_set = list containing several tfidf model parameters.
        a: minimum document frequency
        b: maximum document frequency
        c: maximum number of features
        d: normalize term vector

    OUTPUT:
    tfidf matrix
    tfidf model
    '''
    docs = new_df['reviewText']
    docs_np = np.array(docs).tolist()

    #running tf-idf vectorizer with custom stop_words and tokenizer
    tfidfmodel = TfidfVectorizer(stop_words=STOPLIST, tokenizer=lemma, \
                            min_df=tfidf_set[0],max_df=tfidf_set[1], \
                            max_features =tfidf_set[2], norm=tfidf_set[3])
    #tfidf matrix is constructed using reiview contents
    tfidf_mat = tfidfmodel.fit_transform(docs_np)
    return tfidf_mat, tfidfmodel

def NMF_elbow(tfidf_mat, K = range(1,16)):
    '''
    This function code came from
    Ryan Henning (https://github.com/acu192)

    FUNCTION:
    the function print out elbow plot of NMF results

    INPUT:
    tfidf_mat = tfidf Matrix
    K = range of component used for NMF calculation

    OUTPUT:
    returns nmf.reconstruction_err_ = reconstruction error of nmf model with k components and M tfidf matrix
    '''

    start_time = time.time()
    error = [_fit_nmf(i,tfidf_mat) for i in K]
    plt.plot(K, error)
    plt.xlabel('k')
    plt.ylabel('Reconstruction Errror')
    print("calculated in %s seconds \n\n" % (time.time() - start_time))

    print "error difference"
    for n in xrange(len(K)-1):
        print error[n]-error[n+1]
    print "\n"
    for nn in xrange(len(K)):
        print "error k = {}: {}".format(K[nn], error[nn])


def Data_prep2_NMF(new_df, tfidf_mat,tfidfmodel, NMF_set = [8,'cd',0.1,0.5]):
    '''
    FUNCTION:
    Using tfidf matrix and tfidf model, this function returns NMF model.
    The function also add NMF results(probability) as features.
    INPUT:
    new_df = Pandas dataframe. Output of Data_prep1 function
    tfidf_mat = tfidf matrix from Data_prep2_tfidf
    tfidfmodel = tfidf model from Data_prep2_tfidf
    NMF_set = list containing several NMF parameters.
        a: n components
        b: name of solver
        c: alpha regularization term value
        d: the regularization mixing parameter

    OUTPUT:
    data frame with NMF results as features.
    NMF model
    '''
    #running NMF with tf-idf vectorized sparse matrix
    NMFer = NMF(n_components=NMF_set[0], solver=NMF_set[1],random_state=32113,\
                                        alpha=NMF_set[2], l1_ratio=NMF_set[3])
    NMFresults = NMFer.fit_transform(tfidf_mat)

    # prints out top words
    tfidf_feature_names = tfidfmodel.get_feature_names()
    print_top_words(NMFer, tfidf_feature_names, 20)

    #converting NMF results to a dataframe
    NMF_result_df = pd.DataFrame(NMFresults)

    df_NMF = pd.concat([new_df,NMF_result_df],axis = 1)
    df_NMF = df_NMF.drop(['reviewerName','num_of_helpful_review',], \
                                                                    axis = 1)
    # renaming NMF topics
    df_NMF = df_NMF.rename(columns = {0:'percent_GROUP_1', 1:'percent_GROUP_2',\
                            2:'percent_GROUP_3', 3:'percent_GROUP_4', \
                            4:'percent_GROUP_5', 5:'percent_GROUP_6', \
                            6:'percent_GROUP_7', 7:'percent_GROUP_8', \
                            8:'percent_GROUP_9', 9:'percent_GROUP_10',\
                            10:'percent_GROUP_11', 11:'percent_GROUP_12', \
                            12:'percent_GROUP_13', 13:'percent_GROUP_14', \
                            14:'percent_GROUP_15', 15:'percent_GROUP_16', \
                            16:'percent_GROUP_17', 17:'percent_GROUP_18', \
                            18:'percent_GROUP_19', 19:'percent_GROUP_20'})

    return df_NMF,NMFer

def Data_prep3(df_NMF, max_feat, tfidf_set=[2,0.95,10000,'l2']):
    '''
    FUNCTION:
    will calculate tfidf matrix with n (max_feat) top words and
    make this tfidf matrix into a data frame.
    convine this data frame with the output of Data_prep2_NMF.

    NOTE:
    I could just use tfidf model from Data_prep2_tfidf, but I thought having
    10000 tfidf term as features are just too much.
    I didn't explore much on finding optimal max_feat. 1000 worked well for me
    but the prediction accuracy may go up with different number.

    INPUT:
    df_NMF = Output data frame of Data_prep2_tfidf function
    max_feat = maximum number of feature you want to include in the tfidf matrix
    tfidf_set = list containing several tfidf model parameters.
        a: minimum document frequency
        b: maximum document frequency
        c: maximum number of features
        d: normalize term vector

    OUTPUT:
    data frame that is ready for prediction modeling.
    tfidf model with n (max_feat) tfidf terms
    '''
    # running tfidf model
    docs = df_NMF['reviewText']
    #somehow I needed to change docs type to unicode. let me know if you know why
    docs_np = np.array(docs).astype('U')
    docs_np = docs_np.tolist()
    #this tfidf_feat model only contains max_feat number of features
    tfidf_feat = TfidfVectorizer(stop_words=STOPLIST, tokenizer=lemma, \
                            min_df=tfidf_set[0],max_df=tfidf_set[1], \
                            max_features =max_feat, norm=tfidf_set[3])
    #tfidf_feat_matrix again only contains max_feat number of features
    tfidf_vectorized_mf = tfidf_feat.fit_transform(docs_np)

    #making tfidf terms into data frame features
    vocab_dict = {}
    for k,v in enumerate(tfidf_feat.vocabulary_):
        vocab_dict[k]=v
    tfidf_mf = tfidf_vectorized_mf.toarray()
    tfidf_df = pd.DataFrame(tfidf_mf)
    tfidf_df= tfidf_df.rename(columns =vocab_dict)

    # convine tfidf dataframe above and output dataframe from Data_prep2_NMF
    df_NMF2 = pd.concat([df_NMF,tfidf_df],axis = 1)
    # gets rid of features that are no longer needed.
    df_NMF2 = df_NMF2.drop(['reviewText','summary', 'reviewerID', 'helpful_total_review'],axis = 1)

    return df_NMF2, tfidf_feat


def label_prep(new_df, line=.75):
    '''
    FUNCTION:
    Create label feature (the prediction model is a binary classifier).
    the label feature has two class
        a: HIGH = review is classified as a highly helpful review
        b: LOW = review is classified as not helpful review

    INPUT:
    new_df = Data frame with 'helpful_percent' feature
    line = value used to separate label classes

    OUTPUT:
    a data frame with 'label' feature.
    '''
    #add new feature 'label' (= label for my prediction model)
    percent_vote_helpfulness2(new_df, line)
    new_df['label']=0
    #if the value of helpful_percent is greater than 'line' input,
    #the label is classified as 'HIGH' (meaning highly helpful review)
    new_df.loc[new_df.helpful_percent >= line, 'label'] = 'HIGH'
    #if not, it will classified as 'LOW' (not helpful review)
    new_df.loc[new_df.helpful_percent < line, 'label'] = 'LOW'
    return new_df

##############################################################################
#                                XGBOOST RUN                                 #
##############################################################################


def df_for_XGBOOST(df, lim):
    '''
    FUNCTION:
    * balance total numbers of 2 label classes with lim input
    * delete 'helpful_percent' and 'asin' features
    * Output X data and label dataset in np ndarray format.
        you want to run this function twice (one for training dataset and
        another for test dataset)

    INPUT:
    df = output dataframe from the df_prep2 function (pandas dataframe)
    lim = maximum number of reviews to extract from df (used in _dataselector function) (int)

    OUTPUT:
    df_np = X data ready for XGboost
    Y2 = label data
    new_df = data_frame after applying function. used for xgb_stats function
    '''
    # extract n numbers of reviews with LOW label and High label
    df.index = range(len(df))
    df_low = _dataselector(df,'LOW',lim)
    df_high = _dataselector(df,'HIGH',lim)
    # the new_df should have 2 balanced label classes
    new_df = pd.concat([df_low,df_high], axis=0)
    lab = ['LOW','HIGH']
    new_df = new_df.drop(['helpful_percent','asin'], axis = 1)
    Y = new_df.pop('label')
    # turning Y into binary (LOW = 0, HIGH = 1)
    b_loon={}
    for i in xrange(len(lab)):
        b_loon[lab[i]] = i
    Y2 = Y.map(b_loon)
    df_np = np.array(new_df)
    #X_tr1,X_te1,y_tr1,y_te1 =train_test_split(df_test_np,Y2,test_size = 0.15, \
    #                                        random_state=831713, stratify = Y2)

    return df_np, Y2, new_df


def XGBOOSTING(X_tr1,X_te1,y_tr1,y_te1,xgb_para=[4000,0.25]):
    '''
    FUNCTION:
    Creates Xgboost model for you.
    It also prints out score for your prediction model

    INPUT:
    X_tr,X_te,y_tr,y_te = both X and y training and test data
    xgb_para = parameters for Xgboost model
        a: number of estimator
        b: learning rate

    OUTPUT:
    Xgboost model

    Note:
    I should explore more options for parameters

    '''
    #runs xgboost fitting. takes a while especially if you dataframe is large
    start_time = time.time()
    xgb = XGB.XGBClassifier(n_estimators=xgb_para[0],\
                                        learning_rate=xgb_para[1])
    xgb.fit(X_tr1,y_tr1)
    print("--- %s seconds ---" % (time.time() - start_time))
    # print out XGboost score (test data overall accuracy)
    score = xgb.score(X_te1,y_te1)
    print "score: {}%".format((score*100).round(2))

    return xgb


def xgb_stats(model,new_df,X_test,y_test):
    '''
    FUNCTION:
    using XGboost model, test data and prediction model dataframe,
    this function will print out stats of XGB modeling results.

    INPUT:
    model = XGBoost model from XGBOOSTING function
    new_df = output data frame from df_for_XGBOOST function
    X_test = X test data
    y_test = y test data

    OUTPUT:
    prints out confusion matrix, top15 important features for prediction, and
    HIGH and LOW prediction accuracy

    '''
    #sorts feature importance and print it out
    ind = np.argsort(model.feature_importances_)
    imp = np.sort(model.feature_importances_)
    imp2 = [new_df.columns[i] for i in ind]
    print ' **TOP15 Important Features**  '
    for i in xrange(15):
        print'{} : {}%  '.format(imp2[i-15], imp[i-15]*100)
    print "\n"
    #runs confusion matrix function
    conf_mat(model,X_test,y_test)

##############################################################################
#                              Adding Meta data                              #
##############################################################################

def add_meta_info(df,meta,key_word,filter_num,test=False):
    '''
    function:
    preprocessing of  meta data features
    1. merge review data frame and meta data frame
    2. add rank_keys and rank_values features
    3. deletes features 'imUrl','related','title','brand','description'
    4. add features with product category names
    5. apply filter:
        * filter by category name (if product category is unrelated
        to the dataset topic, that category will be filtered out)
        * filter out any product without price and rank_values
        * see _filter_merged_df description for more details.
    6. add dummy variables for rank_keys feature
    7. deletes features 'salesRank' and 'categories'

    input:
    df = dataframe that contains review information. this df needs to have 'asin' column
    meta = dataframe that contains metadata information
    key_word = name of the main product category for your meta data and review dataset.
    filter_num = any category where total number of product with this category name
        is less than the filter_num number, is classified as an unwanted category.
        see _filter_merged_df description for more details.
    test = if you want to see what is in the meta dataframe, turn test = True.
        otherwise, just ignore this input (optional). (boolean)

    output:
    merged_df1 = merged dataframe (reviews + product meta data)
        with meta features that are ready for modeling.

    '''
    # add_rankings function will create two new features from salesRank feature of meta df. rank_keys and rank_values
    meta, ave_rank_val = _add_rankings(meta,key_word)
    #if you are running a test to see if there is any product that belongs to unrelated category:
    if test:
        print meta.groupby(['rank_keys']).count()
        return meta
    # merge meta and df by asin column
    merged_df = pd.merge(df, meta, how = 'left', left_on = 'asin', right_on = 'asin')
    # delete meta columns that I will not be using for the prediction model
    merged_df1 = merged_df.drop(['imUrl','related','title','brand','description'], axis = 1)

    #the feature 'num category' is the number of category that product is associated with
    merged_df1['num category'] = merged_df1['categories'].str.len()

    #creating a new dataframe called df_category
    #the df_category is a binary matrix where columns = name of category and rows = product

    #stores 'categories' feature from merged_df1 as a numpy array
    category_np = np.array(merged_df1.categories)
    #extracting a set of category names stored in the 'categories' feature
    words_set = set(_category_scrape(category_np,key_word))
    np_cat_case = np.zeros((len(merged_df1),len(words_set)))
    # creating a df_category dataframe
    df_category= pd.DataFrame(np_cat_case)
    df_category.columns = words_set
    #run a function that fills empty dataframe df_category with 1 and 0
    df_category = _category_fill(merged_df1['categories'],df_category, key_word)

    #drops key_word feature (for example, in videogame dataset, key_word = 'Video Games')
    # reason for dropping: 1. the feature only contain 1 so it is likely to be useless in prediction.
                        #  2. there is going to be a feature with same name later in the process.
    df_category = df_category.drop(key_word, axis=1)

    #merging merged_df and df_category
    merged_df1 = pd.concat([merged_df1,df_category], axis = 1)
    #filtering outliers in the merged_df1
    merged_df1 = _filter_merged_df(meta, merged_df1,ave_rank_val,filter_num)
    #add dummies for rank_keys features
    dum = pd.get_dummies(merged_df1.rank_keys, drop_first=True)
    merged_df1 = pd.concat([merged_df1,dum], axis = 1)
    #dropping categories and salesRank features now that we have additional features related to them
    merged_df1 = merged_df1.drop(['categories','salesRank'], axis = 1)
    return merged_df1

def price_adjustor(merged_df):
    '''
    function:
    Replace null values in 'price' feature with average price of each category name.

    input:
    merged_df1 = merged dataframe (Pandas dataframe)

    output:
    merged dataframe with no Null values under 'price' feature (Pandas dataframe)
    '''
    counting = 0
    #for every category name exist in the 'rank_key' feature,
    for cat in list(set(merged_df['rank_keys'])):
        #create a dataframe for each cat.
        #also the function below will fill any NaN values for 'price' feature.
        df_a = _NANprice_filler(merged_df,cat)
        # concat resulting dataframes with previous data frame
        if counting != 0:
            df_b = pd.concat([df_a,df_b], axis = 0)
        else:
            df_b = df_a
        counting = 1
    return df_b


def _add_rankings(meta, topic):
    '''
    function:
    1. Cleaning feature "salesRank"
        a: fillin null values
    2. Generate 2 new features Rank_keys and Rank_values
        a: Rank_keys stores ranking category (str)
        b: Rank_values stores ranking of the prodcut (int)
    3. Calculate and output average rank_values
    4. Output meta data frame with 2 additional feature (2)
    * note that this function is only used within the function "add_meta_info"

    input:
    meta = dataframe that contains metadata information (Pandas Dataframe)
    topic = name of the main topic of the meta data (str)

    output:
    meta = meta data with additional features (Pandas dataframe)
    ave_rank_val = average of rank_values feature that are not -1 (int)

    notes:
    I think I can simplify this code using dictionary;not lists.
    '''
    #create empty lists for
    values_list = []
    keys_list = []
    #if salesRank has null values, fill it with 0.
    meta['salesRank'].fillna(0)
    for ind in xrange(len(meta)):
        '''
        if and elseif statements here checks for
        any erroneous values in the salesRank feature.
        all erroneous entries will be treated as {topic: -1} where
        topic = main category of the dataset (i.e. books, home & kitchen,etc.)
        '''
        if type(meta.get_value(ind, 'salesRank')) != dict:
            keys_list.append(topic)
            values_list.append(-1)
        elif meta.get_value(ind, 'salesRank') == {}:
            keys_list.append(topic)
            values_list.append(-1)
        '''
        if the salesRank has actual data, dictionary key is stored in keys_list and
        dictionary value is stored in values_list.
        '''
        else:
            keys_list.append(meta.get_value(ind, 'salesRank').keys()[0])
            values_list.append(meta.get_value(ind, 'salesRank').values()[0])
    # make 2 new features called rank_keys and rank_values from two lists
    meta['rank_keys'] = keys_list
    meta['rank_values'] = values_list
    # following codes replace -1 rank_values with average of rank_values that is not -1.
    ave_rank_val = np.mean(meta.rank_values[meta.rank_values != -1])
    meta.loc[meta.rank_values == -1, 'rank_values'] = ave_rank_val
    return meta, ave_rank_val

def _category_scrape(column, topic):
    '''
    Values of 'categories' feature are the list of list.
    with in a list there are lists of word that consist of category names.
    The first item of list is usually big category name
    (I'm calling them "main categories" of a list)
    and following items tend to be sub-category names.

    function:
    This function extracts category names that exist in these lists and
    put them under 1 huge list.
    Only criteria here is that the main category of a list have to match 'topic' input.
    If the main category of a list does not match, that list will be ignored
    and words within this list will not be extracted.

    input:
    column = 'categories' feature series (list of list)
    topic = name of the main category of meta data (str)

    output:
    list of category names

    '''

    return [word for cat_loc in column for category in cat_loc \
                                for word in category if category[0] == topic]

def _category_fill(merged_df1, df_category, key):
    '''
    function:
    the function checks 'categories' features of every reviews in merged_df1
    once this function makes a list of category names for each reviews,
    it will then place these information on to df_category.
    df_category has features with category names and in this function, we are
    making df_category a binary matrix (# of reviews * # of category names)

    input:
    merged_df1 = merged dataframe (Pandas dataframe)
    df_category = empty dataframe with the shape of
                    (len(merged_df1), total number of categories)
    key = main category name (I used "Home & Kitchen" and "Video Games" for my projects)

    output:
    dataframe (binary matrix) that contains
            information about categories of products
    '''
    #for each index, create a list of category names
    for ind in xrange(len(merged_df1)):
        wordlist = list(set([word for category in merged_df1.get_value(ind, 'categories') \
                                for word in category if category[0] == key]))
        # length of wordlist should be short around 2 to 5
        for word in wordlist:
            df_category.set_value(ind,word,1)
    return df_category

def _filter_merged_df(meta, merged_df,average,filter_num):
    '''
    function:
    The rank_keys are names of category that products are ranked in.
    Ideally, rank_keys should only have 1 value for each dataset.
    (For example, if I am working on video game reviews, every product should be
    ranked in "Video Games" category ranking.)
    Since that is not the case, I needed to find a way to filter
    products by rank_keys.
    I don't want reviews of furniture products in my review corpus if
    I want to work on video game review helpfulness prediction. right?
    that furniture product reviews can become outliers in my corpus.

    So the way I filtered here is by grouping meta dataframe by 'rank_keys' and
    applied .count(). That will list me total numbers of product in
    each category (Video Games, furniture, Home and Kitchen etc.)
    I then created a list of category names that are minority in my dataset
    (in our example, furniture category in video game dataset would be that minority)
    Using this list, I filter out any reviews of product
    where product's category is in the list of unwanted categories.

    I also filtered out products that do not have price or rank_values

    input:
    meta = meta dataframe
    merged_df = mereged dataframe that are not filtered
    average = average rank of all products listed in the meta data frame
    filter_num = any category where total number of product with this category name
    is less than the filter_num number, is classified as an unwanted category.

    output:
    dataframe that are filtered
    '''
    # bad_ranks list stores all category names that are probably
    # not related to the main category that we are dealing with.
    bad_ranks = list(meta.groupby(['rank_keys']).count()\
                [meta.groupby(['rank_keys']).count()['asin'] < filter_num].index)
    #dropping rows if the value of 'rank_keys' of the row is included in the bad_ranks list
    test = merged_df[~merged_df.rank_keys.isin(bad_ranks)]
    #dropping rows where metadata had NaN/null values for price and ranking features.
    test2 = test[~((test['price'].isnull()) & (test['rank_values']==average))]
    return test2

def _NANprice_filler(merged_df,rank_key_cat):
    '''
    function:
    this function calculates average price for each category of products.
    It will then fill null values with this average price for each category.
    input:
    merged_df = merged dataframe (Pandas dataframe)
    rank_key_cat =

    output:
    dataframe without NULL VALUES! YAY!!
    '''
    #topic_wo_null is a price feature with rank_keys == name of a topic. It does not contain any null values
    topic_wo_null = merged_df['price'][merged_df['rank_keys']==rank_key_cat].dropna(how='all')
    #takes average value of topic_wo_null and replace null values with this average value.
    average_val = (np.mean(topic_wo_null))
    #print statement showing average price for selected category
    print "average price for {} is ${}".format(rank_key_cat, average_val)
    #filling null values
    df_topic = merged_df[merged_df['rank_keys']==rank_key_cat]
    df_topic['price'] = df_topic['price'].fillna(average_val)
    return df_topic

##############################################################################
#                              OTHER FUNCTIONS                               #
##############################################################################

def percent_vote_helpfulness2(df,line):
    '''
    FUNCTION:
    this code will look at percentage of positive votes
    for each review(row) in your dataframe
    and counts total number of reviews with
    high or low percentage of positive reviews.
    Threshold is set by "line" input value.
    used to determine "limit" value for df_for_XGBOOST function.

    INPUT:
    df = the dataframe that contains feature ['helpful_percent']
    line = reviews with positive vote percentage higher than the line value
    will be determined as reviews with high positive vote percentage
    if lower than the line value, it will be determined as reviews with
    lowe positive vote percentage

    OUTPUT:
    print statement with total number of highly helpful reviews and
    not helpful reviews that exist within the input dataframe

    '''
    high = df[df['helpful_percent']>=line]
    low = df[df['helpful_percent']<line]
    print "highly helpful count: {}".format(len(high))
    print "not helpful count: {}".format(len(low))


def print_top_words(model, feature_names, n_top_words):
    '''
    This function code came from
    http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html

    FUNCTION:
    This function displays NMF results.
    The function prints out 'n' top words that exist within each topic from NMF

    INPUT:
    model= your NMF model (Scikit-learn NMF class)
    feature_names = tfidfvectorizer_model.get_feature_names() (list)
    n_top_words = top n words that represents each topic (int)

    OUTPUT:
    A Print statements that lists n_top_words for each topic
    '''
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print(-- end --)


def _fit_nmf(k,M):
    '''
    FUNCTION:
    the function returns reconstruction error of NMF that is calcurated by input k and M

    INPUT:
    k = number of component used to calculate nmf
    M = tfidf Matrix

    OUTPUT:
    returns nmf.reconstruction_err_ = reconstruction error of nmf model with
    k components and M tfidf matrix.
    '''
    #create new nmf model and fit with tfidf matrix
    nmf = NMF(n_components=k, solver='cd',random_state=32113,\
                                        alpha=0.1, l1_ratio=0.5)
    nmf.fit(M)
    #returns reconstruction error of the nmf model.
    return nmf.reconstruction_err_


def _dataselector(df,label,limit):
    '''
    FUNCTION:
    The _dataselector function will extract n('limit input') or less number of reviews from
    dataframe with label feature = "label input".
    This function is only used within df_for_XGBOOST function.

    INPUT:
    df = pandas dataframe that contains features to run XGBOOST and a "label" feature.
    label = 'LOW' or 'HIGH'
    limit = maximum number of reviews to extract from df

    OUTPUT:
    df_lb = pandas dataframe that has n('limit input') or less number of reviews
    with label feature = "label input".
    '''
    # if the total number of reviews with label feature 'label' is greater than 'limit':
    if df[df['label']==label].count()[0] > limit:
        #df_lb only contains reviews with label feature ='label'
        df_lb = df[df['label']==label]
        #randomly select n (limit) number of index without replacement
        #from the index column of df
        random_c = np.random.choice(list(df_lb.index), limit, replace=False)
        # df_lb now contains reviews where index of reviews is in the list random_c
        df_lb = df_lb.loc[list(random_c)]
    # if the total number of reviews with label feature 'label' is less than 'limit':
    else:
        #df_lb only contains reviews with label feature ='label'
        df_lb = df[df['label']==label]
    return df_lb


def conf_mat(model, X_te1, y_te1):
    '''
    FUNCTION:
    This function prints out confusion matrix of a model
    The function will calculate predicted y and compare it to true Y and
    print out confusion matrix of XGboost model results and prediction accuracy for
    Highly helpful reviews and not so helpful reviews.

    INPUT:
    model= your XGBoost model
    X_te1 = X_test data (np.ndarray)
    y_te1 = y_test data (np.ndarray)

    OUTPUT:
    Prints out confusion matrix of your XGBoost model and prediction accuracies
    for two label classes.
    '''
    #calculate predicted y 'y_hat' from the model
    y_hat = model.predict(X_te1)
    y_test = y_te1
    y_test.index=xrange(len(y_test))
    test=pd.DataFrame(y_test)
    test['y_hat']=y_hat
    #making a confusion matrix 2 by 2 array
    cof_mat=np.zeros((2,2))
    np_test=np.array(test)
    np_test = np_test.tolist()
    for i in xrange(len(np_test)):
        if np_test[i] == [0,0]:
            cof_mat[0,0] += 1
        elif np_test[i] == [1,0]:
            cof_mat[1,0] += 1
        elif np_test[i] == [0,1]:
            cof_mat[0,1] += 1
        elif np_test[i] == [1,1]:
            cof_mat[1,1] += 1

    #confusion matrix print statement
    title = ['NOT HELPFUL PRED',' HIGHLY HELPFUL PRED', \
                    'NOT HELPFUL TRUE','HIGHLY HELPFUL TRUE']
    print " |                 |       {}       |        {}        |\n \
    |:--------------: | :-------------------:|:-----------------------:|\n \
    |       {}     |        {}        |           {}        |\n \
    |       {}     |        {}        |           {}        |\n" \
    .format(title[2],title[3],title[0],cof_mat[0,0],cof_mat[1,0],title[1],cof_mat[0,1],cof_mat[1,1])

    print "LOW prediction rate: {}%".format((cof_mat[0][0]/(cof_mat[0][0]+cof_mat[0][1])*100).round(2))
    print "HIGH prediction rate: {}%\n".format((cof_mat[1][1]/(cof_mat[1][1]+cof_mat[1][0])*100).round(2))
