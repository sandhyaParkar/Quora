##---------------------------Quora Question Pairs--------------------##
##-------------------------------------------------------------------##
##--------------------------Problem Statement------------------------##
# Predict which of the provided pairs of questions contain two questions
# with the same meaning.

# cleaning environment
rm(list=ls())

# Get and set working directory
curr_dir <- getwd()
curr_dir
setwd("D:/Quora Project")
getwd()

# Installing all required packages and loading the libraries
x <- c("text2vec", "stringr", "data.table", "pryr", "caret", "glmnet", "SnowballC")
install.packages(x)

library(text2vec)
library(pryr) # pryr::object_size(data1,data2,data3...) - provides combined memory of all objects in the list
library(data.table)
library(caret)
library(stringr) #for string replace_all function
library(tidytext) # for stop_words
library(glmnet)

train=read.csv("train.csv" ,header=TRUE)
str(train)
summary(train)
#--------------- Custom Functions - Start --------------#

# removes all html tags
delete_htmltags <- function(htmlString) {
  return(gsub("<.*?>", "", htmlString))
}

#removes line breaks
remove_linebreaks <- function(htmlString) {
  return(gsub("\r?\n|\r", " ", htmlString))
}

# clean string by removing special characters, numbers newline characters, 
# punctuations and converting into lower case.

clean_string <- function(string) {
  tempString <- tolower(string)
  tempString <- str_replace_all(tempString, "[^[:alnum:]]", " ") 
  tempString <- gsub('[[:digit:]]+', '', tempString)
  tempString <- gsub("([><-])|[[:punct:]]", "\\1", tempString)
}

# remove stop words

rm_words <- function(string, words) {
  stopifnot(is.character(string), is.character(words))
  splitted <- strsplit(string, " ", fixed = TRUE) # fixed = TRUE for speedup
  vapply(splitted, function(x) paste(x[!tolower(x) %in% words], collapse = " "), character(1))
}


#------------ Custom Functions End ---------------#


start.time <- proc.time()

# Split data into train and test (p=0.8 for 80:20 ratio)

trainIndex <- createDataPartition(y= train$is_duplicate, p=0.8, list= FALSE)

train <- train[trainIndex,]
test <- train[-trainIndex,]

###-----------------------------------------------------------------------###
###---------------------Preprocessing and Feature Engineering-------------###
###-----------------------------------------------------------------------###

# Removing all ids from train and test data

train$qid1 = NULL
train$qid2 = NULL

test$qid1=NULL
test$qid2=NULL

###--------------Applying custom functions on train data--------###

train$question1 <- delete_htmltags(train$question1)
train$question2 <- delete_htmltags(train$question2)

train$question1 <- remove_linebreaks(train$question1)
train$question2 <- remove_linebreaks(train$question2)

train$question1 <- clean_string(train$question1)
train$question2 <- clean_string(train$question2)

# Remove stop words from question 1 and 2
custom_stop_words <- stop_words

train$question1 <- rm_words(train$question1, custom_stop_words$word)
train$question2 <- rm_words(train$question2, custom_stop_words$word)

#head(train$question1)
#head(train$question2)

train$question1 <- as.character(train$question1)
train$question2 <- as.character(train$question2)

#pryr::object_size(train)

train$question <- paste0(train$question1, train$question2)

# Creating iterator over tokens using itoken() for train data
train_set <- itoken(train$question, ids=train$id, progressbar = FALSE)

# creating vocabulary using create_vocabulary() function
 
vocab = create_vocabulary(train_set)
vocab

vectorizer = vocab_vectorizer(vocab)

# Creating iterator over tokens using itoken() for train data
train_set <- itoken(train$question, ids=train$id, progressbar = FALSE)

##---Constructing a document term matrix for train_set
t1 = Sys.time()
dtm_train = create_dtm(train_set, vectorizer)
print(difftime(Sys.time(), t1, units = 'sec'))

#  checking dimensions of our document term matrix
dim(dtm_train)

###----------------predictive modelling------------###
# Binomial classification using regression
# Using glmnet package with 4 fold cross validation
# For logistic regression we need to write family parameter as 
# family = 'binomial'

NFOLDS = 4
t1 = Sys.time()
glmnet_classifier = cv.glmnet(x = dtm_train, y = train[['is_duplicate']], 
                              family = 'binomial', 
                              # L1 penalty
                              alpha = 1,
                              # interested in the area under ROC curve
                              type.measure = "auc",
                              # 4-fold cross-validation
                              nfolds = NFOLDS,
                              # high value is less accurate, but has faster training
                              thresh = 1e-3,
                              # again lower number of iterations for faster training
                              maxit = 1e3)
print(difftime(Sys.time(), t1, units = 'sec'))

plot(glmnet_classifier)
print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))

#-------------- data preprocessing and feature engineering for test set -------#

###-----------------Applying custom functions on test data-----------###

test$question1 <- delete_htmltags(test$question1)
test$question2 <- delete_htmltags(test$question2)

test$question1 <- remove_linebreaks(test$question1)
test$question2 <- remove_linebreaks(test$question2)

test$question1 <- clean_string(test$question1)
test$question2 <- clean_string(test$question2)

test$question1 <- rm_words(test$question1, custom_stop_words$word)
test$question2 <- rm_words(test$question2, custom_stop_words$word)

test$question1 <- as.character(test$question1)
test$question2 <- as.character(test$question2)

test$question <- paste0(test$question1, test$question2)

# creating iterator over token using itoken() function for test data

  test_set <- itoken(test$question, ids=train$id , progressbar = FALSE)
  
# creating document term matrix
  
  dtm_test = create_dtm(test_set, vectorizer)
  
# checking performance on test data  
  
  preds = predict(glmnet_classifier, dtm_test, type = 'response')[,1]
  glmnet:::auc(test$is_duplicate, preds)
  
##------------------------------Pruning Vocabulary----------------------------##
# reducing training time and improving accuracy by pruning vocabulary

    # here we will remove some predefined stop words
  
  stop_words = c("i", "me", "my", "myself","we", "our", "ours","ourselves", "you", "your", "yours")
  
  t1 = Sys.time()
  
  # creating vocabulary with these predefined stop words
  
  vocab = create_vocabulary(train_set, stopwords = stop_words)
  
  print(difftime(Sys.time(), t1, units = 'sec'))
  
  pruned_vocab = prune_vocabulary(vocab, 
                                  term_count_min = 10, 
                                  doc_proportion_max = 0.5,
                                  doc_proportion_min = 0.001)
  
  vectorizer = vocab_vectorizer(pruned_vocab)
  
  # create dtm for train data with pruned vocabulary vectorizer
  
  t1 = Sys.time()
  
  dtm_train  = create_dtm(train_set, vectorizer)
  
  print(difftime(Sys.time(), t1, units = 'sec'))
  
  dim(dtm_train)
  
  # creating dtm for test data with same vectorizer
  
  dtm_test   = create_dtm(test_set, vectorizer)
  
  dim(dtm_test)
  
  t1 = Sys.time()
  
  ##----improving model using N-grams
  # here we are using upto 2 grams
  
  vocab = create_vocabulary(train_set, ngram = c(1L, 2L))
  print(difftime(Sys.time(), t1, units = 'sec'))
  
  vocab = vocab %>% prune_vocabulary(term_count_min = 10, 
                                     doc_proportion_max = 0.5)
  
  bigram_vectorizer = vocab_vectorizer(vocab)
  
  # creating dtm using n gram vectorizer
  
  dtm_train = create_dtm(train_set, bigram_vectorizer)
  
  ##---------applying model----------##
  
  t1 = Sys.time()
  
  glmnet_classifier = cv.glmnet(x = dtm_train, y = train[['is_duplicate']], 
                                family = 'binomial', 
                                alpha = 1,
                                type.measure = "auc",
                                nfolds = NFOLDS,
                                thresh = 1e-3,
                                maxit = 1e3)
  
  print(difftime(Sys.time(), t1, units = 'sec'))
  
  plot(glmnet_classifier)
  
  print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))
  
  # apply bi gram vectorizer on test data for creating dtm
  
  dtm_test = create_dtm(test_set, bigram_vectorizer)
  
  # checking the performance on test data
  
  preds = predict(glmnet_classifier, dtm_test, type = 'response')[,1]
  
  glmnet:::auc(test$is_duplicate, preds)
  
  ##-------Using Feature Hashing in text2vec-------------##
  
  h_vectorizer = hash_vectorizer(hash_size = 2 ^ 14, ngram = c(1L, 2L))
  
  t1 = Sys.time()
  dtm_train = create_dtm(train_set, h_vectorizer)
  
  print(difftime(Sys.time(), t1, units = 'sec'))
  t1 = Sys.time()
  
  glmnet_classifier = cv.glmnet(x = dtm_train, y = train[['is_duplicate']], 
                                family = 'binomial', 
                                alpha = 1,
                                type.measure = "auc",
                                nfolds = 5,
                                thresh = 1e-3,
                                maxit = 1e3)
  print(difftime(Sys.time(), t1, units = 'sec'))
  
  plot(glmnet_classifier)
  
  print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))
  
  ##---checking performance on test data
  
  dtm_test = create_dtm(test_set, h_vectorizer)
  
  preds = predict(glmnet_classifier, dtm_test , type = 'response')[, 1]
  glmnet:::auc(test$is_duplicate, preds)
  
  ##--------tfidf transformation for normalization purpose---------##
  
    vocab = create_vocabulary(train_set)
  vectorizer = vocab_vectorizer(vocab)
  dtm_train = create_dtm(train_set, vectorizer)
  
  # define tfidf model
  tfidf = TfIdf$new()
  
  # fit model to train data and transform it
  dtm_train_tfidf = fit_transform(dtm_train, tfidf)
  
  # applying tf-idf transformation to test data
  dtm_test_tfidf  = create_dtm(test_set, vectorizer) %>% 
    transform(tfidf)
  
  #------Applying model---------#
  
  t1 = Sys.time()
  
  glmnet_classifier = cv.glmnet(x = dtm_train_tfidf, y = train[['is_duplicate']], 
                                family = 'binomial', 
                                alpha = 1,
                                type.measure = "auc",
                                nfolds = NFOLDS,
                                thresh = 1e-3,
                                maxit = 1e3)
  print(difftime(Sys.time(), t1, units = 'sec'))
  
  plot(glmnet_classifier)
  
  print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))
  
  # checking performance on test data
  
  preds = predict(glmnet_classifier, dtm_test_tfidf, type = 'response')[,1]
  glmnet:::auc(test$is_duplicate, preds)
  
  #--------------------------------------------------------------------#
  #--------------------------------------------------------------------#
  #---------------------------Thank You--------------------------------#
  #--------------------------------------------------------------------#
  #--------------------------------------------------------------------#
  
  
  
  