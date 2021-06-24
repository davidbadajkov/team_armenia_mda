
#! pip install boto3
import boto3
import os

import pandas as pd
import re

session = boto3.Session(aws_access_key_id = 'AKIATJJR2V5VZLS2FLF7', aws_secret_access_key = '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')
s3 = session.resource('s3')
bucket = s3.Bucket('s3grouparmenia')

for file in bucket.objects.all():
  if file.key.endswith('.txt'):
    print(file.key)
    break

transcripts = []
years = []
countries = []
sessions = []
years_cleaned = []

for file in bucket.objects.all():
  if file.key.startswith('data/Converted sessions'):
    name = str(file.key)
    if name.endswith('.txt'):
      years.append(name.split('_')[-1])
      sessions.append(name.split('_')[1])
      countries.append(name.split('_')[0][-3:])
      obj = file.get()['Body'].read()
      transcripts.append(obj)

for year in years:
  years_cleaned.append(year.replace('.txt', ''))

dic = {
    'Year': years_cleaned,
    'Session': sessions,
    'Country': countries,
    'Transcript': transcripts
}

df = pd.DataFrame(dic)
df.head()

i = df.loc[df.Year == 'Store-to-UTF-8'].index
df.drop(i, inplace = True)
df.reset_index
df.info()

df.head()

df.to_csv('consolidated_transcripts.csv')
bucket.upload_file(Filename = 'consolidated_transcripts.csv', Key = 'consolidated_transcripts.csv')

