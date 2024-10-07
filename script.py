import csv
import pandas as pd
import json 
from nrclex import NRCLex
import nltk

nltk.download('punkt')

df1 = pd.read_csv('./archive/instagram_data/captions_csv.csv')
df2 = pd.read_csv('./archive/instagram_data2/captions_csv2.csv')
df = pd.concat([df1, df2], ignore_index=True)

emotions_list = ['fear', 'anger', 'anticipation', 'trust', 'surprise', 
                 'positive', 'negative', 'sadness', 'disgust', 'joy']

for emotion in emotions_list:
    df[emotion] = 0
for index, caption in df.iterrows():
    emotion = NRCLex(str(caption['Caption']))
    emotion_frequencies = emotion.raw_emotion_scores

    for em in emotions_list:
        df.at[index, em] = emotion_frequencies.get(em, 0)  
all_top_captions = []
for emotion in emotions_list:
    top_10 = df.nlargest(10, emotion)[['Caption', 'Image File']].copy()
    top_10['Emotion'] = emotion 
    all_top_captions.append(top_10)

final_df = pd.concat(all_top_captions)
final_df.to_csv('captions.csv', index=False)
print("Top 4 captions for each emotion have been saved to a single CSV file!")


