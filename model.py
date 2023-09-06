#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import re
import operator
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation 
from tensorflow.keras.layers import Embedding 
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 
import os, re, csv, math, codecs 


# In[436]:


training = pd.read_csv('train.csv') #### import train.csv 
print([column for column in training.columns])

####Il existe de nombreuses valeurs nulles dans humor_rating. Attribuez la valeur nulle à 0, 
###car Comme humour_rating est associé à is_humor, si is_humor est égal à 0, alors humour_rating doit être égal à 0.
training['humor_rating'].fillna(0, inplace= True)                                                                    #### humor_rating中有很多空值，把不幽默的值赋值为0. 
print(training.info())

test = pd.read_csv('public_dev.csv')  ####import public_dev

test.dropna(axis=0, how='any', inplace = True) 

training['text'] = training['text'].apply(lambda x: x.lower())  
test['text'] = test['text'].apply(lambda x: x.lower()) 
a = training['humor_rating'].max()   ####Trouvez la valeur maximale de humor_rating                                   ####找到humor_rating的最大值
training['humor_rating'] = training['humor_rating']/a    
#####Normalization归一化，cela renvoie la valeur de cette colonne de 0 à 1. Parce que j'utiliserai sigmoïde plus tard.      因为后面我将使用sigmoid
####Elle équivaut à la valeur exprimée en pourcentage, la plus humor va être plus proche 1.                                          把值归位0-1.相当于用百分比表示，最幽默的是1.

b = training['offense_rating'].max()
training['offense_rating'] = training['offense_rating']/b


# In[4]:


word_vectors_dict = {}      
with open("glove.6B.50d.txt", 'r', encoding="utf-8") as f:  #### f représente toutes les lignes de glove.6B.50d.txt       f代表glove.6B.50d.txt的所有行
    for line in f:                        ####     line représente une ligne en f                                       line代表f中的一行 
        values = line.split()             #### Coupez le contenu de chaque ligne par des virgules                               把每一行中的内容按照逗号切开
        print(values) 
        word = values[0] 
        vector = np.asarray(values[1:], "float32") 
        word_vectors_dict[word] = vector     ####Construire la matrice de vecteur de mot


# In[ ]:


def count_vocab(text_column):    #### Afin de connaître le nombre d'occurrences de chaque mot dans tous les textes
                ####text_column == Colonne de texte dans train.csv                                                          text_column 数据集中的文本列 
    tweets = text_column.apply(lambda s: s.split()).values      
    ####Coupez tout le texte de la colonne de texte en fonction des espaces.                                                   把这一文本列的所有文本进行切割 
    vocab = {}
    
    for tweet in tweets: ####Itérer chaque tweet. un tweet==un text                                                           对每一条推特进行遍历。
        for word in tweet: ####Itérer sur chaque mot dans chaque tweet.                                                                  对每一条推特的每一个词进行遍历。
            try:
                vocab[word] += 1                  
            except KeyError:
                vocab[word] = 1                
    return vocab  
####Enfin, j'obtiendrai un tableau de tous les mots de tout le texte et connais le nombre d'occurrences de chaque mot                                         最终我将获得一个由所有文本的所有词组成的数组 


# In[439]:


def check_embeddings_coverage(X, embeddings):   
    
    vocab = count_vocab(X)    
    
    covered = {}

    oov = {}     #### oov est un mot méconnaissable                                                                      oov是未匹配上的单词
    n_covered = 0
    n_oov = 0
    
    for word in vocab:
        try:
            covered[word] = embeddings[word]
            n_covered += vocab[word]                     
        except:
            oov[word] = vocab[word]
            n_oov += vocab[word]
            
    vocab_coverage = len(covered) / len(vocab)    #### Le rapport entre les mots reconnus et les mots du texte total                                                           被识别的单词占总文本中的单词的比率 
    text_coverage = (n_covered / (n_covered + n_oov))    #### le taux de Couverture de texte                                      文本覆盖率 
    
    return vocab_coverage, text_coverage


# In[440]:


def clean_data(tweet):


	### remove punctuations and special characters
	punctuations = '\@\#\!\?\+\&\*\[\]\-\%\.\:\/\(\)\;\$\=\>\<\|\"\{\}\^\,\.\—¥º¬£_' + "\`"                                     ####一条推特中的特殊字符P（p in '\@\#\!\?\+\&\*\[\]\-\%\.\:\/\(\)\;\$\=\>\<\|\"\{\}\^\,\.\—¥º¬£_' + "\`"）被''代替
	for p in punctuations:
		tweet = tweet.replace(p,'')

    # Contractions
	tweet = re.sub(r"he's" or "he’s", "he is", tweet)
	tweet = re.sub(r"there's|there’s", "there is", tweet)
	tweet = re.sub(r"we're|we’re", "we are", tweet)
	tweet = re.sub(r"that's|that’s", "that is", tweet)
	tweet = re.sub(r"won't|won’t", "will not", tweet)
	tweet = re.sub(r"they're|they’re", "they are", tweet)
	tweet = re.sub(r"can't|can’t", "cannot", tweet)
	tweet = re.sub(r"wasn't|wasn’t", "was not", tweet)
	tweet = re.sub(r"don't|don’t", "do not", tweet)
	tweet = re.sub(r"aren't|aren’t", "are not", tweet)
	tweet = re.sub(r"isn't|isn’t", "is not", tweet)
	tweet = re.sub(r"what's|what’s", "what is", tweet)
	tweet = re.sub(r"haven't|haven’t", "have not", tweet)
	tweet = re.sub(r"hasn't|hasn’t", "has not", tweet)
	tweet = re.sub(r"there's|there’s", "there is", tweet)
	tweet = re.sub(r"he's|he’s", "he is", tweet)
	tweet = re.sub(r"it's|it’s", "it is", tweet)
	tweet = re.sub(r"shouldn't|shouldn’t", "should not", tweet)
	tweet = re.sub(r"wouldn't|wouldn’t", "would not", tweet)
	tweet = re.sub(r"i'm|i’m", "i am", tweet)
	tweet = re.sub(r"isn't|isn’t", "is not", tweet)
	tweet = re.sub(r"here's|here’s", "here is", tweet)
	tweet = re.sub(r"you've|you’ve", "you have", tweet)
	tweet = re.sub(r"we're|we’re", "we are", tweet)
	tweet = re.sub(r"what's|what’s", "what is", tweet)
	tweet = re.sub(r"couldn't|couldn’t", "could not", tweet)
	tweet = re.sub(r"we've|we’ve", "we have", tweet)
	tweet = re.sub(r"who's|who’s", "who is", tweet)
	tweet = re.sub(r"y'all|y’all", "you all", tweet)
	tweet = re.sub(r"would've|would’ve", "would have", tweet)
	tweet = re.sub(r"it'll|it’ll", "it will", tweet)
	tweet = re.sub(r"we'll|we’ll", "we will", tweet)
	tweet = re.sub(r"we've|we’ve", "we have", tweet)
	tweet = re.sub(r"he'll|he’ll", "he will", tweet)
	tweet = re.sub(r"weren't|weren’t", "were not", tweet)
	tweet = re.sub(r"didn't|didn’t", "did not", tweet)
	tweet = re.sub(r"they'll|they’ll", "they will", tweet)
	tweet = re.sub(r"they'd|they’d", "they would", tweet)
	tweet = re.sub(r"they've|they’ve", "they have", tweet)
	tweet = re.sub(r"i'd|i’d", "i would", tweet)
	tweet = re.sub(r"should've|should’ve", "should have", tweet)
	tweet = re.sub(r"where's|where’s", "where is", tweet)
	tweet = re.sub(r"we'd|we’d", "we would", tweet)
	tweet = re.sub(r"i'll|i’ll", "i will", tweet)
	tweet = re.sub(r"weren't|weren’t", "were not", tweet)
	tweet = re.sub(r"they're|they’re", "they are", tweet)
	tweet = re.sub(r"let's|let’s", "let us", tweet)
	tweet = re.sub(r"it's|it’s", "it is", tweet)
	tweet = re.sub(r"can't|can’t", "cannot", tweet)
	tweet = re.sub(r"don't|don’t", "do not", tweet)
	tweet = re.sub(r"you're|you’re", "you are", tweet)
	tweet = re.sub(r"i've|i’ve", "i have", tweet)
	tweet = re.sub(r"that's|that’s", "that is", tweet)
	tweet = re.sub(r"i'll|i’ll", "i will", tweet)
	tweet = re.sub(r"doesn't|doesn’t", "does not", tweet)
	tweet = re.sub(r"i'd|i’d", "i would", tweet)
	tweet = re.sub(r"didn't|didn’t", "did not", tweet)
	tweet = re.sub(r"ain't|ain’t", "am not", tweet)
	tweet = re.sub(r"you'll|you’ll", "you will", tweet)
	tweet = re.sub(r"i've|i’ve", "i have", tweet)
	tweet = re.sub(r"don't|don’t", "do not", tweet)
	tweet = re.sub(r"i'll|i’ll", "i will", tweet)
	tweet = re.sub(r"i'd|i’d", "i would", tweet)
	tweet = re.sub(r"let's|let’s", "let us", tweet)
	tweet = re.sub(r"you'd|you’d", "you would", tweet)
	tweet = re.sub(r"it's|it’s", "it is", tweet)
	tweet = re.sub(r"ain't|ain’t", "am not", tweet)
	tweet = re.sub(r"haven't|haven’t", "have not", tweet)
	tweet = re.sub(r"could've|could’ve", "could have", tweet)
	tweet = re.sub(r"you've|you’ve", "you have", tweet)  

	tweet = ' '.join(re.sub("'s", "", tweet).split())
	tweet = ' '.join(re.sub("who'll", "who will", tweet).split())
	tweet = ' '.join(re.sub("'", "", tweet).split())
	tweet = ' '.join(re.sub("“", "", tweet).split())
	tweet = ' '.join(re.sub("”", "", tweet).split())
	

	###wrong spelling
	tweet = re.sub(r"foodsupplies", "food supplies", tweet)
	tweet = re.sub(r"glowinthedark", "glow in the dark", tweet)  
	tweet = re.sub(r"poundforpound", "pound for pound", tweet)  
	tweet = re.sub(r"blowjobs':", "blow jobs", tweet)  
	tweet = re.sub(r"handjobs", "hand jobs", tweet)  
	tweet = re.sub(r"16yearsold", "16 years old", tweet)  
	tweet = re.sub(r"killcount", "kill count", tweet)  
	tweet = re.sub(r"20yearold", "20 years old", tweet)  
	tweet = re.sub(r"7yearold", "7 years old", tweet)  	
	tweet = re.sub(r"twoyearold", "two years old", tweet)  	
	tweet = re.sub(r"yeahthat", "yeah that", tweet)  
	tweet = re.sub(r"40mins", "40 minutes", tweet)  
	tweet = re.sub(r"15mins", "15 minutes", tweet)
	tweet = re.sub(r"4day", "4 days", tweet)
	tweet = re.sub(r"45mph", "45 miles per hour", tweet)
	tweet = re.sub(r"28yearsold", "28 years old", tweet)
	tweet = re.sub(r"bouillonaire", "billionaire", tweet)
	tweet = re.sub(r"candylike", "candy like", tweet)
	tweet = re.sub(r"helpseeking", "help seeking", tweet)
	tweet = re.sub(r"metoo", "me too", tweet)
	tweet = re.sub(r"lowcarb", "low carbonate", tweet)
	tweet = re.sub(r"skinfriendly", "skin friendly", tweet)
	tweet = re.sub(r"mother’s", "mother", tweet)
	tweet = re.sub(r"herbefore", "her before", tweet)
	tweet = re.sub(r"country’s", "country", tweet)
	tweet = re.sub(r"therell", "the rell", tweet)
	tweet = re.sub(r"thered", "the red", tweet)
	tweet = re.sub(r"marketwe", "martket we", tweet)
	tweet = re.sub(r"pussay", "pussy", tweet)
	tweet = re.sub(r"lovetoo", "love too", tweet)
	tweet = re.sub(r"nobell", "no bell", tweet)
	tweet = re.sub(r"fourcar", "four car", tweet)
	tweet = re.sub(r"threeinch", "three inch", tweet)
	tweet = re.sub(r"memoryenhancing", "memory enhancing", tweet)
	tweet = re.sub(r"10yr", "10 years", tweet)
	tweet = re.sub(r"freeradicals", "free radicals", tweet)
	tweet = re.sub(r"moneysaver", "money saver", tweet)
	tweet = re.sub(r"timethe", "time the", tweet)
	tweet = re.sub(r"lifeshe", "life she", tweet)
	tweet = re.sub(r"downlow", "down low", tweet)
	tweet = re.sub(r"poundmetoo", "pound me too", tweet)
	tweet = re.sub(r"funtoo", "fun too", tweet)
	tweet = re.sub(r"soontoo", "soon too", tweet)
	tweet = re.sub(r"muchtoo", "much too", tweet)
	tweet = re.sub(r"sincewell", "since well", tweet)
	tweet = re.sub(r"mayflowers", "may flowers", tweet)
	tweet = re.sub(r"youjustdo", "you just do", tweet)
	tweet = re.sub(r"topspeed", "top speed", tweet)
	tweet = re.sub(r"publicshame", "public shame", tweet)
	tweet = re.sub(r"afterwork", "after work", tweet)
	tweet = re.sub(r"72hour", "72 hour", tweet)
	tweet = re.sub(r"wayyyyy", "way", tweet)
	tweet = re.sub(r"xrays", "x-rays", tweet)
	tweet = re.sub(r"everyone’s", "everyone", tweet)
	tweet = re.sub(r"highpressure", "high pressure", tweet)
	tweet = re.sub(r"12yearold", "12 year old", tweet)
	tweet = re.sub(r"menfrequentlymistakenforwomen", "men frequently mistaken for women", tweet)
	tweet = re.sub(r"twoincome", "two income", tweet)
	tweet = re.sub(r"1dayold", "1 day old", tweet)
	tweet = re.sub(r"30day", "30 days", tweet)
	tweet = re.sub(r"haaaaaaay", "hey", tweet)
	tweet = re.sub(r"haaaaay", "hey", tweet)
	tweet = re.sub(r"bestfriend", "best friend", tweet)
	tweet = re.sub(r"ontimelate", "on time late", tweet)
	tweet = re.sub(r"1yearold", "1 year old", tweet)
	tweet = re.sub(r"verylaughing", "very laughing", tweet)
	tweet = re.sub(r"maam", "women", tweet)
	tweet = re.sub(r"folow", "follow", tweet)
	tweet = re.sub(r"sunbathes", "sun bathes", tweet)
	tweet = re.sub(r"exgirlfriend", "previous girlfriend", tweet)
	tweet = re.sub(r"exfriends", "previous friends", tweet)
	tweet = re.sub(r"fingerpaint", "finger paint", tweet)
	tweet = re.sub(r"76yearold", "76 year old", tweet)
	tweet = re.sub(r"13yearold", "13 year old", tweet)
	tweet = re.sub(r"200millionyear", "200 million years", tweet)
	tweet = re.sub(r"drivein", "drive in", tweet)
	tweet = re.sub(r"crossfitter", "cross fitter", tweet)
	tweet = re.sub(r"wantknow", "want know", tweet)
	tweet = re.sub(r"notyou", "not you", tweet)
	tweet = re.sub(r"supernatura", "super natural", tweet)
	tweet = re.sub(r"yknow", "you know", tweet)
	tweet = re.sub(r"covid", "coronavirus", tweet)
	tweet = re.sub(r"thatmmmmmmm", "that", tweet)
	tweet = re.sub(r"teacoffee", "tea coffee", tweet)
	tweet = re.sub(r"heyyyy", "hey", tweet)
	tweet = re.sub(r"hospitalone", "hospital one", tweet)



	#Special case not handled previously.
	tweet = tweet.replace('\x98',"'")
	tweet = tweet.replace('\x89',"'")
	tweet = tweet.replace('\x9b\_',"'")
	tweet = tweet.replace('\x94',"'")
	tweet = tweet.replace('\x9b',"'")
	tweet = tweet.replace('\x8f',"'")

	###remove all numbers
	tweet = re.sub(r"[1-9]+\.?[0-9]*", "", tweet)
	#####remove invalid twitter
	tweet = re.sub(r"jokinfjreoiwjrtwe4to8rkljreun8f4ny84c8y4t58lym4wthylmhawt4mylt4amlathnatyn", "", tweet)
	tweet = re.sub(r"httptcoqsncnmqs", "", tweet)
	tweet = re.sub(r"oclock", "", tweet)
	


	tweet = ' '.join(re.sub("’", "", tweet).split())      
	tweet = ' '.join(re.sub("'", "", tweet).split())
	return tweet


# In[441]:


#### before cleaning
train_vocab_coverage_original, train_text_coverage_original = check_embeddings_coverage(training['text'], word_vectors_dict)
test_vocab_coverage_original, test_text_coverage_original = check_embeddings_coverage(test['text'], word_vectors_dict)
                                                                     
print('GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Training Set'.format(train_vocab_coverage_original, train_text_coverage_original))
print('GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Test Set'.format(test_vocab_coverage_original, test_text_coverage_original))


# In[442]:


predict = pd.DataFrame()   
training['cleaned_text'] = training['text'].apply(lambda x: clean_data(x))  
####Nettoyer tout le texte de l'ensemble de données de training （train.csv)                                                          对training数据集所有文本进行清洗

training_controversy = training.dropna() 
#### Supprimer la valeur nulle de ligne de training, ne laissant que les données avec valeur.                               #### dropna()使用NaN作为缺失数据的标记。把training数据集空值的数据行给删除 
#### Entraînez la colonne humor_controversy séparément                                                                       把training数据集中空值全部删除，把humor_controversy一列单独进行训练，
#### Étant donné que la colonne humor_controversy n'a pas de connexion logique évidente avec les trois autres colonnes, 
#### et que certaines sont même en conflit avec les valeurs des trois autres colonnes, 
#### elle est retirée séparément pour l‘entraîner.                                                                                              因为这一列和其余三列并无明显的逻辑关联，甚至还有冲突，因此单独拎出来进行训练
training_controversy['cleaned_text'] = training_controversy['text'].apply(lambda x: clean_data(x))

predict['cleaned_text'] = test['text'].apply(lambda x: clean_data(x))  
### Nettoyer'public_dev.csv '

train_vocab_coverage_original, train_text_coverage_original = check_embeddings_coverage(training['cleaned_text'], word_vectors_dict)
test_vocab_coverage_original, test_text_coverage_original = check_embeddings_coverage(predict['cleaned_text'], word_vectors_dict)
print('GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in cleaned Training Set'.format(train_vocab_coverage_original, train_text_coverage_original))
print('GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in cleaned Test Set'.format(test_vocab_coverage_original, test_text_coverage_original))


# In[443]:


####1.
raw_docs_train = training['cleaned_text'].tolist()  ####Transformez le texte nettoyé de training en liste
                                                                                                                      ####把training清洗过的文本变成list
####2.
raw_training_controversy = training_controversy['cleaned_text'].tolist()  
####Transformez le texte qui a été nettoyé et “cleaned_text” sans valeurs nulles en une liste
                                                                                                                         ####把training清洗过的没有空值的文本变成list
####3.
predict = predict['cleaned_text'].tolist()

y_train = training[['is_humor', 'humor_rating', 'humor_controversy', 'offense_rating']].to_numpy()
#### Mettez les valeurs de ces colonnes dans le tableau

y_train_controversy = training_controversy[ 'humor_controversy'].to_numpy()
####Mettez les valeurs de ce colonne de humor_controversy dans le tableau                                                        把humor_controversy一列单独进行训练 
print(y_train_controversy)


# In[465]:


from tensorflow.keras.preprocessing.text import Tokenizer                                                                   ####Tokenizer可以对所有文本里面的不同单词进行编号
####Tokenizer peut numéroter différents mots dans tous les textes

tokenizer = Tokenizer(num_words=11285, lower=True, char_level=False)      ####lower=True   minuscule                                小写
tokenizer.fit_on_texts(raw_docs_train +  predict)                                                                         ####raw_docs_train是train.csv的liste版本
####Numéroter tous les mots du texte                                                                                    对文本单词进行编号 

word_seq_train = tokenizer.texts_to_sequences(raw_docs_train) 
####train--Chaque phrase sera remplacée par un numéro de mot                                                                    把每一句话用单词编号代替，也就是说现在一条文本都由一串数字表示
word_seq_train_controversy = tokenizer.texts_to_sequences(raw_training_controversy)
#### controversy--Chaque phrase sera remplacée par un numéro de mot                                                                                                   那一列把每一句话用单词编号好代替，上同
predict_seq = tokenizer.texts_to_sequences(predict) 
#### predict--Chaque phrase sera remplacée par un numéro de mot                                                                                                把每一句话用单词编号好代替 ，上同

word_index = tokenizer.word_index     #### word_index==Numéro de mot
print("dictionary size: ", len(word_index))  ####Nous pouvons calculer le nombre de mots compilés                              编了多少个单词，为下一步


# In[466]:


from tensorflow.keras.preprocessing import sequence 
#### J'ai défini une longueur de 30 mots pour chaque tweet                                                               每一个推特取30个词的长度 
word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=30, padding='post', truncating='post') 
####maxlen=30 Acceptez jusqu'à 30 mots
#### padding='post'Cela signifie que si le nombre de mots dans un texte est inférieur à 30, 
####ajoutez 0 après le dernier mot du tableau.                   

print(word_seq_train)                          

word_seq_train_controversy = sequence.pad_sequences(word_seq_train_controversy, maxlen=30,padding='post', truncating='post') 
print(word_seq_train_controversy)

word_seq_predict = sequence.pad_sequences(predict_seq, maxlen=30, padding='post', truncating='post')


# In[467]:


#training params
batch_size = 128   ####   Apprenez 128 textes à la fois                                                                                                超参数，不停的换。数小学的细，学的越慢，并且工作量大。反之亦然。
num_epochs = 50   ####    Étudiez 50 fois au total

####batch_size, Il s'agit d'un hyperparamètre, 
####nous devons constamment ajuster la valeur du paramètre pour obtenir le résultat optimal.
####Plus la valeur du paramètre est petite, 
####plus l'apprentissage automatique est précis, plus le temps d'apprentissage est long 
####et plus la charge de travail est lourde.

#model parameters
num_filters = 32   #### Nombre de neurones convolutifs                                                                  卷积神经元的数量，过滤的神经元。不宜太多，太多容易过拟合。
#### Convolutional neurons/filtered neurons. If there are too many neurons, it is easy to overfit.
embed_dim = 50     #### Parce que j'utilise glove.6B.50d.txt, voici donc 50


# In[474]:


words_not_found = []  ####Construire une matrice vectorielle de tous les mots. 
####Le objectif est qu'un mot correspond à un chiffre et ce chiffre correspond à un vecteur.
                                                                                                                                 ####所有的11285单词的向量矩阵
nb_words = min(11285, len(word_index)) 
embedding_matrix = np.zeros((nb_words, embed_dim)) 
####Construire une matrice 0                                                                                                     /构建0矩阵，矩阵行数就是单词个数                #####用的是glove50，embed_dim列数是50
for word, i in word_index.items():
    if i >= nb_words:
        continue
        
    embedding_vector = word_vectors_dict.get(word)  #### Obtenir les vecteur d'un certain mot                                     得到某一个词的向量
    if (embedding_vector is not None) and len(embedding_vector) > 0:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector 
    else:
        words_not_found.append(word)

print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))


# In[475]:


print("sample words not found: ", np.random.choice(words_not_found, 10))                                                       ####哪些没有进入向量矩阵的单词
####Montrer quels mots ne sont pas dans la matrice


# In[476]:


##1. first row(is_hurmour)------------------------------------------------------------------------------------------------                                                词向量矩阵构造完毕，正式进入训练模型


# In[477]:


from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
print("training CNN ...") 
#### Étape 1：Choisir un modèle

model = Sequential()   ####Sequence model                                                                                        简单 对新人友好 Modèle de séquence 函数的不好更改

#### Étape 2: Step 2: Créer the network layer

####Embedding Établir une connexion entre la couche et la matrice vectorielle Embedding                                        建立了层与向量矩阵的连接
model.add(Embedding(nb_words, embed_dim,  
          weights=[embedding_matrix], input_length=30, trainable=False))                                               #### trainable=False我已经有了，不需要电脑自我学习了

model.add(Conv1D(num_filters, 7, activation='relu', padding='same')) 
#### Utilisez des filtres pour mettre en évidence les principales caractéristiques d'une phrase                         通过filters去突出一句话的主要特征
model.add(GlobalMaxPooling1D())  #### Extraire les principales caractéristiques d'une phrase                                      抓去被突出的一句话特征

model.add(Dropout(0.5))     ####Utilise 50% dropout pour Abandonner certains neurones pour éviter Overfitting                                            舍弃一部分神经元，防止过拟合

model.add(Dense(32, activation='relu')) 
#### Les caractéristiques du feature map sont représentées par 
#### 32 neurones entièrement connectés (représentant 32 caractéristiques). Cette opération rend le modèle plus robuste.                    将feature map中的特征，用32个全连接神经元表示（代表32个特征），此操作使得模型robust更强。

model.add(Dense(1, activation='sigmoid'))   #### Utilisez sigmoïde comme fonction d'activation
####sigmoïde garantit que le résultat final est compris entre 0 et 1.                                                             ####sigmoid保证结果0到1之间


####Étape 3: compiler

adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)    
##### Optimiser la fonction, définir le taux d'apprentissage (lr) et d'autres paramètres
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])   
#### Use cross entropy as loss function    
                                                                                                                        ####binary_crossentropy二分类，f1.score
model.summary()


# In[478]:


early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)                                              #### val_loss，min_delta=0.01低于啥也没学到， patience=4，
#### J'ai défini le seuil optimal sur 0,01 et je me concentre uniquement sur l'apprentissage qui dépasse cette valeur
#### la valeur de perte calculée à epoch diminue progressivement 
#### et la valeur de perte de variation est inférieure à 0,01 pendant 4 fois consécutives
#### C'est à dire que la machine ne peux plus étudier beaucoup de chose                                                                                    学不到什么东西了可以实现早停
callbacks_list = [early_stopping]    


# In[ ]:





# In[479]:


####Étape 4: Training
hist = model.fit(word_seq_train, y_train[:,0], batch_size=batch_size, 
                 epochs=num_epochs, callbacks=callbacks_list, validation_split=0.1, shuffle=False, verbose=2)
#### model.fit() est utilisée pour effectuer le processus de l‘entraînement
#### validation_split = 0,1. Divisez 0,1 de l'ensemble de test à l'ensemble d'apprentissage


# In[ ]:





# In[480]:


import matplotlib.pyplot as plt

plt.figure() 
plt.plot(hist.history['loss'], lw=2.0, color='b', label='train')
plt.plot(hist.history['val_loss'], lw=2.0, color='r', label='val') 
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.legend(loc='upper right')
plt.show()       
#### train_Lost est inférieur à val_Lost. 
#### À partir de la 10e génération, overfitting apparaît, mais nous pouvons l'accepter.                                                                 trainLost小于valLost从10代开始，过拟合Overfitting


# In[481]:



result = model.predict(word_seq_predict)
print(result) 


# In[482]:


result = model.predict(word_seq_predict)
final_result_humor_or_not = []
for i in result:
    if i > 0.5: 
        ####Si la valeur de prévision est supérieure à 0,5, alors nous donnons à is_humor la valeur 1
        final_result_humor_or_not.append(1)
    else:
        final_result_humor_or_not.append(0)


# In[ ]:





# In[483]:


submission_df = pd.DataFrame(columns=['is_humor'])
submission_df['text'] = test['text'].values 
submission_df['is_humor'] = final_result_humor_or_not


# In[618]:


##2.'humor_rating' 预测有多幽默，预测数字------------------------------------------------------------------------------------


# In[ ]:


### training params
batch_size = 64        
num_epochs = 50 

#model parameter
num_filters = 32
embed_dim = 50 


# In[631]:


from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
print("training CNN ...")
model = Sequential()   
model.add(Embedding(nb_words, embed_dim,
          weights=[embedding_matrix], input_length=30, trainable=False))
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))                                                    #####        加了一层， 提取特征更仔细一点
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  

adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])   #### «mse» peut mieux prédire la valeur.                                                      ####='mse'  预测数值
model.summary()


# In[633]:


early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]                                                                                            ####shuffle=False不能打乱顺序，效果更好


# In[634]:


hist = model.fit(word_seq_train, y_train[:,1], batch_size=batch_size, epochs=num_epochs,  
                                            callbacks=callbacks_list, validation_split=0.1, shuffle=False, verbose=2)


# In[635]:


import matplotlib.pyplot as plt

plt.figure()
plt.plot(hist.history['loss'], lw=2.0, color='b', label='train')
plt.plot(hist.history['val_loss'], lw=2.0, color='r', label='val')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.legend(loc='upper right')
plt.show()


# In[636]:


result = model.predict(word_seq_predict)
result


# In[637]:


submission_df['humor_rating'] = result


# In[638]:


submission_df['humor_rating'] = submission_df['humor_rating']*a    #### Normalisation inverse                                         反向归一化


# In[639]:


##3.'humor_controversy' 


# In[659]:


#training params
batch_size = 32
num_epochs = 50 

#model parameters
num_filters = 16
embed_dim = 50 


# In[664]:


from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
print("training CNN ...")
model = Sequential()
model.add(Embedding(nb_words, embed_dim,
          weights=[embedding_matrix], input_length=30, trainable=False))

#### supprimer model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
#### 删除model.add(MaxPooling1D(2))     为了让模型更简单，batch_size = 32
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  
adam = optimizers.SGD(lr=0.0001)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()


# In[665]:


early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]


# In[666]:


hist = model.fit(word_seq_train_controversy, y_train_controversy, batch_size=batch_size, 
                                                epochs=num_epochs,  validation_split=0.1, shuffle=False, verbose=2)


# In[667]:


import matplotlib.pyplot as plt

plt.figure()
plt.plot(hist.history['loss'], lw=2.0, color='b', label='train')
plt.plot(hist.history['val_loss'], lw=2.0, color='r', label='val')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.legend(loc='upper right')
plt.show()


# In[668]:


result = model.predict(word_seq_predict)
final_result_humor_controversy = []
for i in result:
    if i > 0.5:
        final_result_humor_controversy.append(1)
    else:
        final_result_humor_controversy.append(0)


# In[670]:


submission_df['humor_controversy'] = final_result_humor_controversy


# In[350]:


#第4列 offense rating ----------------------------------------------------------------------------------------------------------
#####预测是一个数，只看val——lost  La prédiction n'est qu'un nombre, il suffit de regarder val——lost.


# In[699]:


#training params
batch_size = 32
num_epochs = 50 

#model parameters
num_filters = 16
embed_dim = 50 


# In[700]:


from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
print("training CNN ...")
model = Sequential()
model.add(Embedding(nb_words, embed_dim,
          weights=[embedding_matrix], input_length=30, trainable=False))

model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5)) 
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  

adam = optimizers.SGD(lr=0.0001)
model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])     
model.summary()


# In[ ]:





# In[701]:


early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]


# In[702]:


hist = model.fit(word_seq_train, y_train[:,3], batch_size=batch_size, 
                                                    epochs=num_epochs,validation_split=0.1, shuffle=False, verbose=2)


# In[703]:


import matplotlib.pyplot as plt

plt.figure()
plt.plot(hist.history['loss'], lw=2.0, color='b', label='train')
plt.plot(hist.history['val_loss'], lw=2.0, color='r', label='val')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')  
plt.legend(loc='upper right')
plt.show()   


# In[704]:


result = model.predict(word_seq_predict)
result


# In[705]:


submission_df['offensive_rating'] = result
submission_df['offensive_rating'] = submission_df['offensive_rating']*b ####  Reverse normalization                                 反向归一化


# In[706]:


submission_df.to_csv('submission.csv')     ####Enfin, le fichier de résultats est généré


# In[ ]:




