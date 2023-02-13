# Sentiment-Analysis-on-Drug-Review-Dataset

### ABSTRACT

This project uses the Drug Review Dataset to understand user reviews regarding multiple drugs for different conditions. These reviews are in the form of numeric ratings and textual reviews. The textual reviews are analyzed to predict the polarity of its sentiment and classified into one of 5 classifications. Four classification models have been tested: LGBM, XGBoost, catBoost, and Naive Bayes Classifier. The highest accuracy is 75%, given by the LightGBM classifier. The numeric rating is used to recommend the highly-rated drugs for a given condition to the user.

### DATASET 

https://www.kaggle.com/jessicali9530/kuc-hackathon-winter-2018

### METHODOLOGY

Initially the data has been cleaned of null and empty values (preferred over imputation as imputing reviews desensitizes the aim of the study) and then sorted based on the user ratings. We have extracted the unique conditions to help base the recommendation system here along with extracting the top 10 user preferred drugs (based on highest, here 10 pointer ,weighted_rating).

Textblob, a python library for NLP technique, is used to give a sentiment polarity of the review. TextBlob returns polarity and subjectivity of the given user review in sentence. The polarity lies between [-1,1], -1 defines a negative sentiment and 1 defines a positive sentiment. Negation words reverse the polarity. TextBlob has semantic labels that help with fine-tuned analysis. The correlation matrix for the cleaned and uncleaned reviews shows that the removal of stopwords and snowball stemmers are impacting the review to be having a completely different sentiment and hence cleaning is done without the stopwords removal.

The weighted rating is calculated and added to the dataset as a feature and this adds the prioritization of the ratings to improve the recommendation system.

The correlation matrix plotted as a heat map, shows the linear dependence of each feature with all the other features in the data set.

The final step in preprocessing is the Label encoding of the drug name and the condition into numeric values to help aid in the machine learning of this data. We have used label encoding albeit with the one disadvantage that cannot be avoided (also a drawback of this study) is that Label Encoding classifies the data into numbers, and this causes an interpretation of the numbers to be ranked. One hot encoding solves this issue in a general case but cannot be implemented here as there are 3600+ unique values indicating that the dataset will increase with as many dimensions and these created dummy variables create a trap to multicollinearity (dependence between independent variables). 

Machine learning models such as LGBM, XgBoost Classifier, CatBoost classifier etc that have been trained with massive amounts of data provided in the dataset helped us to investigate the viability of using machine learning to categorize user ratings based on their textual review in order to discover areas of contingency in this project. 

The models are trained to predict the target output,'Sentiment Rating'.

### RESULTS

LGBM has accuracy 75.2% with a TP of 24251. We can calculate the TN, FN, FP accordingly.

XGBoost has an accuracy of about 55% with a higher TP of 24390(more sensitivity for this model). 

The accuracy is same as Cat Boost classifier with 55% but lower TP values comparatively indicating low sensitivities.

Naive Bayes classifier has the least accuracy and can be hence avoided to predict the sentiment rating of the user reviews on its drug and the associated condition.
