---
layout:     	notebook
title:     		Stack exchange tags prediction Kaggle competition
author:     	Aissa EL OUAFI
tags:         Machine-Learning NLP Kaggle
subtitle:    	Predict tags from models trained on unrelated topics based on transfer learning models
---

# Stack exchange tags prediction Kaggle competition

The goal of this [Kaggle](https://www.kaggle.com/c/transfer-learning-on-stack-exchange-tags) fun competition is to predict Stack Exchange tags based on Transfer Learning approach. In this competition, we provide the titles, text, and tags of Stack Exchange questions from six different site.

The challenge of this competition is to learn appropriate <b>physics </b>tags from others topic (travel, biology, cooking, robotic, crypto and tiy).

I will try to solve this problem using a classic text mining approach, I will try different Machine Learning approach. I will try also to use the [`sparklyr`](http://spark.rstudio.com/) library, It's a R Apache Spark interface.

#### Libraries installation
```{r}
library(sparklyr)
library(dplyr)
library(ggplot2)
library(plotly)
library(stringr)
library(plotly)
library(tm)
library(wordcloud)
library(e1071) #Naive bayes classifier
library(stats)
library(factoextra) # PCA
library(topicmodels)
library(LDAvis)
library(devtools)
```

#### Import data
#### LDA / supervised LDA / LDA arch / Analyse morophologique / Analyse syntaxique / embedding anglais (word2vec)

For memory issue (laptop with only 4Gb of RAM) I will analyze only first 1000 observations of each topic, it's about only <b>~10%</b> of real data ! Kaggle provide also a test dataset with 81926 observations talking about physics. Each dataframe contains 4 colmuns (id, title, content and tags column for the train data).
```{r}
biology =  read.csv("../biology.csv",nrows =1000)
travel = read.csv("../travel.csv",nrows=2000)
robotic = read.csv("../robotics.csv",nrows=1000)
cooking = read.csv("../cooking.csv",nrows=1000)
crypto = read.csv("../crypto.csv",nrows=1000)
diy = read.csv("../diy.csv",nrows=1000)
test = read.csv("../test.csv")
#sample_submission = read.csv("../sample_submission.csv",nrows=1000)
```
#### Explore data

Lets explore the train dataset of multiple topics !

```{r}
sc <- spark_connect(master="local")
travel[1:5,]
```

As I said before, every train `dataframe` contains 4 colmuns, <b>id, title, content and tags</b>. The test `dataframe` contains only <b>id, title and content</b> colmun to be predicted is <b>tags</b>. Every observations contains many tags, So we should predict multiple tags of every document.

The test data that will be used to evaluate our model contains documents about <b>physics</b>, So we should predict physics tags from non physics topic. Lets show the 5 first document of test `dataframe` :

```{r}
test[1:5,]
```

#### Data exploration and visualization

##### Tags analysis
After showing the `dataframe`, I noticed that most of the time, the searched tag is contained in the title of the document. So we will start by analyzing documents titles and compare them with tags. Lets show the most frequent tags and comapre them to the most frequent word in the title corpus. We will count distinct tags frequency of every topic.

```{r}
countDistinctTags <- function(df,nFreq){
  df <- as.data.frame(as.factor(unlist(str_split(df$tags," "))))
  colnames(df) <- c("tags")
  df <- data.frame(table(df$tags))
  colnames(df) <- c("tags","Freq")
  df <- df[order(df$Freq,decreasing = TRUE),]
  df <- df[1:nFreq,]
  return(df)
}
```

`countDistinctTags(df,nFreq)` count the distinct tags frequency that appears at least `nFreq` times on the `df` datafram. This function return a `dataframe` of tags frequency. So lets plot the 50 tags the most frequently tags.

```{r}
travel_tags <- countDistinctTags(travel,10)
biology_tags <- countDistinctTags(biology,10)
robotic_tags <- countDistinctTags(robotic,10)
cooking_tags <- countDistinctTags(cooking,10)
crypto_tags <- countDistinctTags(crypto,10)
diy_tags <- countDistinctTags(diy,10)
```
We can show the most frequent tags on every topic, lets plot a chart with differents topics

```{r,fig.width=13,fig.height=9}
travel_chart <- plot_ly(x = travel_tags$tags, y = travel_tags$Freq,type="bar",name="Travel tags")
biology_chart <- plot_ly(x = biology_tags$tags, y = biology_tags$Freq,type="bar",name="Biology tags")
robotic_chart <- plot_ly(x = robotic_tags$tags, y = robotic_tags$Freq,type="bar",name="Robotic tags")
cooking_chart <- plot_ly(x = cooking_tags$tags, y = cooking_tags$Freq,type="bar",name="Cooking tags")
crypto_chart <- plot_ly(x = crypto_tags$tags, y = crypto_tags$Freq,type="bar",name="Crypto tags")
diy_chart <- plot_ly(x = diy_tags$tags, y = diy_tags$Freq,type="bar",name="Diy tags")
subplot(travel_chart,biology_chart,robotic_chart,cooking_chart,crypto_chart,diy_chart,nrows = 3,margin = 0.12)
```

##### Title analysis

Let's now calculate the tags occurrence probability of tags in the topic title.
```{r}
titleTagsProbability <- function(df){
  title_words <- sapply(str_split(df$title," "),'[',1:max(lengths(str_split(df$title," "))))
  title_words <- t(title_words[,1:ncol(title_words)])
  tag_words <- sapply(str_split(df$tags," "),'[',1:max(lengths(str_split(df$tags," "))))
  tag_words <- t(tag_words[,1:ncol(tag_words)])
  tag_in_title <- tag_words %in% title_words
  return(table(tag_in_title)["TRUE"] / length(tag_in_title))
}
```

We will focus now on documents title's corpus of different topics to define the coorrelation between title and tags. `generateCorpus` generate a title corpus based from `df` title's, We apply different pre-processing functions (remove punctuation, remove numbers, remove stop words ...) to generate the corpus using `tm` package.
```{r}
generateCorpus <- function(df){
  corpus <- Corpus(VectorSource(df))
  corpus <- tm_map(corpus,removePunctuation)
  corpus <- tm_map(corpus,removeNumbers)
  corpus <- tm_map(corpus,tolower)
  corpus <- tm_map(corpus, removeWords, c("can", "get","best","good"))   
  corpus <- tm_map(corpus,removeWords,stopwords("english"))
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, PlainTextDocument)
  return(corpus)
}
```
Lets focus now on document title's, we will generate the title corpus of <b>Travel</b> topic.
```{r}
travel_title_corpus <- generateCorpus(travel$title)
```
Now, lets generate the `document term matrix` associated to the travel titles corpus, and show the most frequent word in this corpus.
```{r}
dtm <- DocumentTermMatrix(travel_title_corpus)
dtm
```
```{r}
freq <- sort(colSums(as.matrix(dtm)),decreasing = TRUE)
wf <- data.frame(word=names(freq), freq=freq)[1:10,]
wf
```

We can see that the most frequent word in the title corpus is verry similar to the list of tags frequency defined before. Let's plot a chart of word frequency and compare it to the previous tags frequency plot.

```{r,fig.width=13}
p <- plot_ly(x = wf$word, y = wf$freq,type="bar",name="Travel title word")
subplot(travel_chart,p,nrows = 1)
```


We can show the <b>word cloud</b> of title corpus and compare them to <b>tags</b> frequency.
```{r}
wordcloud(names(freq),freq,min.freq = 10)
wordcloud((countDistinctTags(travel,40))$tags,(countDistinctTags(travel,40))$Freq)
```

We can see that we have many common word between the word title cloud word and tag distinct word, so most of the time the tags appears in the title.

#### Topic modeling using LDA (Latent Dirichlet allocation)

Describe LDA and the main goal of this approach (define topic and word distribution ! )

```{r}
travel_corpus <- generateCorpus(travel$content)
travel_dtm <- DocumentTermMatrix(travel_corpus)
travel_dtm
```

We will apply the LDA using Gibbs sampling ...
```{r}
#Define LDA parameters :
control <- list(iter = 500)
ldaOut <-LDA(travel_dtm,50, method="Gibbs",control=control)
```

```{r}
#Topic of every document in our Document Term Matrix
ldaOut.topics <- as.data.frame(topics(ldaOut))
colnames(ldaOut.topics) <- "topic"
ldaOut.topics
```

```{r}
#Get top K term probability in every topic
K <- 5
ldaOut.terms <- as.data.frame(terms(ldaOut,K))
colnames(ldaOut.terms) <- 1:ncol(ldaOut.terms)
ldaOut.terms <- as.data.frame(t(ldaOut.terms))
ldaOut.terms
```

```{r}
#Topic probability distribution
topicProbabilities <- as.data.frame(ldaOut@gamma)
topicProbabilities
```

##### LDA visualization
Let's show the most frequent topic in the content corpus
```{r}
topicFrequency <- as.data.frame(table(ldaOut.topics$topic))
```

```{r}
#Merge
cols <- colnames(ldaOut.terms)
ldaOut.terms$tags <- apply(ldaOut.terms[ , cols ] , 1 , paste , collapse = " " )
ldaOut.terms$topic <- 1:nrow(ldaOut.terms)
ldaOut.terms
```

```{r}
travel_tags <- inner_join(ldaOut.topics,ldaOut.terms,by="topic")
travel_tags <- as.data.frame(travel_tags$tags)
travel_tags$id <- 1:nrow(travel_tags)
colnames(travel_tags) <- c("tags","id")
travel_tags
```
##### Generate TfIdf matrix

We will now generate the **tfidf** matrix to reflect how important a word is to a document in the corpus. The **tfidf** is defined as the product of two statistics, term frequency and inverse document frequency.

$tf(t,d) = 0.5 + 0.5 \cdot\left(\frac{\underset{t,d}f}{max \{ \underset{t\prime,d}f \::\: t\prime \in d \}}\right)$

$idf(t,D) = log \left(\frac{N}{|\{ d \in D \:: \:t \in d\}|} \right)$

$tfidf(t,d,D) = tf(t,d)\cdot idf(t,D)$

We will start by generate only the **tfidf** matrix of title corpus defined before (because of memory issues ! )

```{r}
tfidf <- DocumentTermMatrix(travel_title_corpus, control = list(weighting = weightTfIdf))
tfidf
```

##### Dimensionality reduction with PCA

The **tfidf** matrix generated based on only 1000 titles of travel documents contains 2209 distinct words, so we have 2209 dimensions, in this case its necessary to the a dimensionality reduction to reduce dimensions. In this case we will use the <b> Principal component analysis (PCA)</b>, it's a mathematical procedure that transforms a number of correlated variables into a smaller number of uncorrelated variables called **principale components**. The first principal component accounts for as much of the variability in the data as possible, and each succeeding component accounts for as much of the remaining variability as possible.

```{r}
tfidf.pca <- prcomp(tfidf,scale. = TRUE)
```
Let's study variances of the principale components, we will start by calculate the **Eigenvalues** that measure the variability retained by each PC. It's large for the first PC and small for the subsequent PCs, we will also calculate the **variance** and the **cumulative variances** of each PC.
```{r}
eig <- (tfidf.pca$sdev)^2
variance <- eig*100/sum(eig)
cumvariance <- cumsum(variance)
eig.dataframe <- data.frame(eig = eig, variance = variance, cumvariance = cumvariance)
eig.dataframe
```

Let show the importance of the first 50 principal components (PC) using a simple barplot.
```{r,fig.width=13}
eig.dataframe <- eig.dataframe[1:50,]
plot_ly(x = nrow(eig.dataframe):1, y = eig.dataframe$cumvariance,type="bar",name="PCs importance") %>% layout(yaxis = list(title = 'Cumulative variance'),xaxis = list(title = "Principal components"))
```


```{r}
sum(eig.dataframe$variance)
```
We can see that the first 50  PCs contains only <b>11.52 %</b> of total variance, we lose <b>88.48 %</b> of data informations, so we can't take only the first 50 PCs.
