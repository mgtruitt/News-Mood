

```python
import tweepy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rand
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
from config import (consumer_key, 
                    consumer_secret, 
                    access_token, 
                    access_token_secret)
```


```python
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
target_news = ["@BBCNews", "@CBSNews", "@CNN", "@FoxNews", "@nytimes"]
```


```python
counter = 0
sentiments = []

for news in target_news:
    public_tweets = api.user_timeline(news, count = 100)
    counter = 0
    for tweet in public_tweets:
        compound = analyzer.polarity_scores(tweet["text"])["compound"]
        pos = analyzer.polarity_scores(tweet["text"])["pos"]
        neu = analyzer.polarity_scores(tweet["text"])["neu"]
        neg = analyzer.polarity_scores(tweet["text"])["neg"]
        tweets_ago = counter
        sentiments.append({"Media Sources": news, "Text": tweet["text"], "Date": tweet["created_at"], "Compound": compound,
                          "Positive": pos, "Negative": neg, "Neutral": neu, "Tweet Count": counter})
        #print("Tweet %s: %s" % (counter, tweet["text"]))
        
        #tweets + 1
        counter = counter + 1
```


```python
sentiment_results = pd.DataFrame.from_dict(sentiments)
results_csv = sentiment_results[['Media Sources', 'Date', 'Text', 'Compound', 'Positive', 
                                 'Neutral', 'Negative','Tweet Count']]
results_csv.to_csv("News_Moods.csv")
results_csv.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Media Sources</th>
      <th>Date</th>
      <th>Text</th>
      <th>Compound</th>
      <th>Positive</th>
      <th>Neutral</th>
      <th>Negative</th>
      <th>Tweet Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@BBCNews</td>
      <td>Sun Jul 15 02:13:35 +0000 2018</td>
      <td>Cuba to recognise private property under new c...</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@BBCNews</td>
      <td>Sun Jul 15 01:03:24 +0000 2018</td>
      <td>Heatwave causes spike in insect bite calls to ...</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@BBCNews</td>
      <td>Sun Jul 15 00:42:30 +0000 2018</td>
      <td>'Why I keep my vasectomy a secret' https://t.c...</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@BBCNews</td>
      <td>Sun Jul 15 00:39:59 +0000 2018</td>
      <td>Haiti Prime Minister Jack Guy Lafontant resign...</td>
      <td>-0.3182</td>
      <td>0.0</td>
      <td>0.753</td>
      <td>0.247</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@BBCNews</td>
      <td>Sun Jul 15 00:33:00 +0000 2018</td>
      <td>Trump to leave UK after two-night stay https:/...</td>
      <td>-0.0516</td>
      <td>0.0</td>
      <td>0.854</td>
      <td>0.146</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.xlim(110, -10)
for news in target_news:
    scatterdf = results_csv.loc[results_csv["Media Sources"] == news]
    plt.scatter(scatterdf["Tweet Count"], scatterdf["Compound"], label = news)
    
    plt.title("Sentiment Analysis of Media Tweets (7/14/2018)")
    plt.xlabel("Tweets Ago")
    plt.ylabel("Tweet Polarity")
    plt.legend(bbox_to_anchor = (1,1), title = 'Media Sources')
    plt.savefig("Scatter_Sentiment.png")
    
```


![png](output_5_0.png)



```python
Overall = results_csv.groupby("Media Sources")["Compound"].mean()
Overall
```




    Media Sources
    @BBCNews   -0.013223
    @CBSNews   -0.202967
    @CNN       -0.029087
    @FoxNews    0.103220
    @nytimes   -0.007490
    Name: Compound, dtype: float64




```python
x_axis = np.arange(len(Overall))
xlabels = Overall.index
count = 0
for result in Overall:
   plt.text(count, result +.01, str(round(result,2)))
   count = count + 1
plt.bar(x_axis, Overall, tick_label = xlabels, color = ['skyblue', 'g', 'r', 'b', 'y'])
plt.title("Overall Sentiment Based on Twitter (7/14/2018)")
plt.ylabel("Tweet Polarity")
plt.savefig("Overall_Sentiment.png")
```


![png](output_7_0.png)


Trend Analysis

1. Fox News was the only media source with a positive overall sentiment average.
2. Out of the overall nevagive media sources, CBS News was the most negative.
3. Looking at the scatter plot, NY Times appears to have the most compounds of zero or close to zero. Looking at the CSV, that is confirmed.
