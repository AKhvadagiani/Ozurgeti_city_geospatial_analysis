import pandas as pd

import re
from rapidfuzz import fuzz
import string

data =pd.read_excel('Ozurgeti.xlsx')

# Columns of interest
data = data[['საიდენტიფიკაციო ნომერი',
             'პირადი ნომერი',
             'დაბის/თემის/სოფლის საკრებულო (ფაქტობრივი)',
             'ორგანიზაციულ-სამართლებრივი ფორმა',
             'სუბიექტის დასახელება','ფაქტობრივი მისამართი',
             'საქმიანობის დასახელება NACE Rev.2',
             'საქმიანობის კოდი NACE Rev.2',
             'აქტიური ეკონომიკური სუბიექტები',
             'ბიზნესის ზომა']]

# Filter for Ozurgeti
data = data[(data['დაბის/თემის/სოფლის საკრებულო (ფაქტობრივი)'].isna()) |(data['დაბის/თემის/სოფლის საკრებულო (ფაქტობრივი)'].str.contains('ქ. ოზურგეთი'))]

# Remove rows with null 'ფაქტობრივი მისამართი'
data = data[~(data['ფაქტობრივი მისამართი'].isnull())]

# Data cleaning functions
def clean_text(input_string):
    cleaned = re.sub(r'[^\w\s]',' ',input_string)
    return cleaned

def remove_punctuation(input_string):
    result = input_string
    for char in string.punctuation:
        result = result.replace(char, ' ')
    return result

data['ნომერი'] = data['ფაქტობრივი მისამართი'].apply(lambda x: "".join([i for i in x if i.isnumeric()]))
data["ქუჩა"] = data["ფაქტობრივი მისამართი"].apply(clean_text)
data["ქუჩა"] = data["ფაქტობრივი მისამართი"].apply(remove_punctuation)
data['ქუჩა'] = data['ქუჩა'].apply(lambda x: "".join([re.sub('\d+', '',i)  for i in x ]))
data['ქუჩა'] = data['ქუჩა'].apply(lambda x: "".join([re.sub(r'\\N', '',i)  for i in x ]))
data['ქუჩა'] =data['ქუჩა'].apply(lambda x: ''.join([i for i in x if i != 'N']))

data['ქუჩა'] =data['ქუჩა'].apply(lambda x: ' '.join([i for i in x.split() if i not in [ 'ქუჩა',
                                                                                        'საქართველო',
                                                                                        'ოზურგეთი',
                                                                                        'ოზურგეთის',
                                                                                        'რაიონში',
                                                                                        'სახელობის',
                                                                                        'შესახვევი',
                                                                                        'შესახ',
                                                                                        'ჩიხი'
                                                                                        ]
                                                                                    ]
                                                                                )
                                                                            )
data['ქუჩა'] = data['ქუჩა'].apply(lambda x: ' '.join([i for i in x.split() if i.isnumeric() or len(i) >3]))
data['ქუჩა'] = data['ქუჩა'].apply(
    lambda x: re.sub(r'თაყაიშვილ(ი|ის)', 'საჯავახო — ჩოხატაური — ოზურგეთი — ქობულეთი', x) if isinstance(x, str) else x)
data = data[ (data['ქუჩა']!='')]
data = data[(data['ნომერი']!='')]
data.sort_values(by='ქუჩა', ascending=True)

def fuzzy_match(ops_name, customer_input):
    similarity_score = fuzz.ratio(ops_name, customer_input)
    return similarity_score, customer_input

OPS_Streets = pd.read_csv('Ozurgeti_streets.txt',header =None, names=['street_name'])
OPS_Streets['street_name'] = OPS_Streets['street_name'].apply(lambda x: x.replace('ქუჩა','') if 'ქუჩა' in x.split() else x)
OPS_Streets = OPS_Streets[~OPS_Streets['street_name'].str.contains(r'\b(?:ჩიხი|შესახვევი)\b', regex=True)]

def match_ops_street(row):
    if row['დაბის/თემის/სოფლის საკრებულო (ფაქტობრივი)'] == 'ქ. ოზურგეთი':
        for i in OPS_Streets['street_name']:
            if isinstance(row['ქუჩა'], str) and row['ქუჩა'].strip() in i.split() and len(row['ქუჩა'].strip()) <= len(
                    i.strip()):
                return i
        return row['ქუჩა']

    elif pd.isna(row['დაბის/თემის/სოფლის საკრებულო (ფაქტობრივი)']):
        for i in OPS_Streets['street_name']:
            if isinstance(row['ქუჩა'], str) and row['ქუჩა'].strip() in i.split() and len(row['ქუჩა'].strip()) <= len(
                    i.strip()):
                return i
        return None


# Apply the function row-wise
data['ქუჩა_OPS'] = data.apply(match_ops_street, axis=1)

data['Similarity'] = data['ქუჩა_OPS'].apply(
    lambda x: max([fuzzy_match(x, street) for street in OPS_Streets['street_name']])
)

data[['similarity_score', 'matched_street']] = pd.DataFrame(data['Similarity'].tolist(), index=data.index)

data.drop(columns='Similarity', inplace=True)

def fill_ops(row):
    if row['similarity_score'] > 90:
        return row['matched_street']
    else:
        return row['ქუჩა_OPS']

data['ქუჩა_საბოლოო'] = data.apply(fill_ops, axis=1)

data = data[~data['ქუჩა_საბოლოო'].isna()]

data.loc[:, 'ქუჩა_საბოლოო'] = data['ქუჩა_საბოლოო'].apply(str.strip)

data.loc[:,'St_Full_Name']=data['ნომერი'] + ' ' + data['ქუჩა_საბოლოო'] + ' ქუჩა, Ozurgeti, Georgia'

data.drop(columns=['პირადი ნომერი','დაბის/თემის/სოფლის საკრებულო (ფაქტობრივი)','ფაქტობრივი მისამართი','ნომერი','ქუჩა','ქუჩა_OPS','similarity_score','matched_street','ქუჩა_საბოლოო'])