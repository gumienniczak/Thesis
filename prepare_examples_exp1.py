import pandas as pd
import contractions
import re
from collections import Counter
from utilities import mask_word, combine_sen, counter_cosine_sim

# pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

with open('negation_items.xlsx', 'rb') as file:
    data_df = pd.read_excel(file)

df = data_df.copy()

beg_opinion = ['they think', 'they believe', 'they suppose', 'they imagine', 'they expect', 'they reckon', 'they feel']
# beg_opinion_people = ['people think', 'people believe', 'people suppose', 'people imagine', 'people expect', 'people reckon', 'people feel']
beg_op_neg = ['they do not think', 'they do not believe', 'they do not suppose', 'they do not imagine', 'they do not expect', 'they do not reckon', 'they do not feel']
# beg_op_ppl_neg = ['people do not think', 'people do not believe', 'people do not suppose', 'people do not imagine', 'people do not people do not expect', 'people do not reckon', 'people do not feel']
beg_perception = ['it seems', 'it appears', 'it looks like', 'it sounds like', 'it feels like']
beg_per_neg = ['it does not seem', 'it does not appear', 'it does not look like', 'it does not sound like', 'it does not feel like']
beg_probability = ['it is probable', 'it is likely', 'it figures']
beg_prob_neg = ['it is not probable', 'it is not likely', 'it does not figure']
beg_judgement = ['they suggest']
# beg_jug_ppl = ['people suggest']
beg_jud_neg = ['they do not suggest']
# beg_jud_ppl = ['people do not suggest']

begs_pos = beg_opinion + beg_perception + beg_probability + beg_judgement
begs_neg = beg_op_neg + beg_per_neg + beg_prob_neg + beg_jud_neg

# print(f'BEG POS: {len(begs_pos)}')
# print(f'BEG NEG: {len(begs_neg)}')

# the code below simplifies the sentences so they do not contain additional conjuctions and parts of the sentence.
for idx, row in df.iterrows():
    string = df.at[idx, 'sentence']
    conj = r'\b' + re.escape(df.at[idx, 'cw_plus1']) + r'\b'
    matches = list(re.finditer(conj, string))
    if matches:
        last_match_start = matches[-1].start()
        new_sentence = df.at[idx, 'sentence'][:last_match_start].strip()
        df.at[idx, 'sentence'] = new_sentence + '.'
    else:
        df.at[idx, 'sentence'] = df.at[idx, 'sentence'].strip() + '.'

# print(len(df), 'len1')

# expanding contractions in sentences and rate
df['sentence'] = df['sentence'].map(lambda sentence: contractions.fix(sentence))
df['rate'] = df['rate'].map(lambda rate: contractions.fix(rate))

# some values are not represented as strings but as boolean values, the code below fixes that
df['cw'] = df['cw'].apply(lambda verb: verb.replace('TRUE', 'true').replace('FALSE', 'false'))
df['cw1'] = df['cw1'].apply(lambda verb: verb.replace('TRUE', 'true').replace('FALSE', 'false'))
df['cw2'] = df['cw2'].apply(lambda verb: verb.replace('TRUE', 'true').replace('FALSE', 'false'))

# some negative sentences do not contain the word not, instead they contain the word 'never'.
problematic = df[(df['condi'].isin(['NT', 'NF'])) & (~df['sentence'].str.contains('not', case=False))]

# editing the below rows, as these are problematic
antar_weather = df[df['mixout'] == 23]
for idx, row in antar_weather.iterrows():
    new_sen = row['sentence'].replace('sometimes sunny but ', '')
    new_rate = row['rate'].replace('sometimes sunny but ', '')
    df.at[idx, 'sentence'] = new_sen
    df.at[idx, 'rate'] = new_rate

not_to_never_map = {
    'do not think': 'never think',
    'do not believe': 'never believe',
    'do not suppose': 'never suppose',
    'do not imagine': 'never imagine',
    'do not expect': 'never expect',
    'do not reckon': 'never reckon',
    'does not seem': 'never seems',
    'does not appear': 'never appears',
    'does not look': 'never looks',
    'doe not sound': 'never sounds',
    'does not feel': 'never feels',
    'not probable': 'never probable',
    'not likely': 'never likely',
    'does not figure': 'never figures',
    'do not suggest': 'never suggest'
}

proper_nouns = ['Al', 'Saddam', 'Michael', 'Ozzy', 'Ellen', 'Elton', 'Tom', 'Ryan', 'Oprah', 'Justin', 'Beyonce', 'Shaquille']

def construct_sen(el1pos, el1neg, sen, condi):
    """This function takes positive and negative elements and combines simple sentence, sen, with these elements,
    forming complex sentences"""
    sentences = []

    def normalise_text(text):
        """Replaces non-breaking spaces with standard spaces in the text."""
        return text.replace('\xa0', ' ')

    def combine_elements(el1, el3):
        """This function combines elements 1, 2 and 3, to form a complex sentence.
        Returns a tuple consisting of the sentence,
        together with values -1,0,1,2 for the position if not within the sentence,
        and values 0,1 depending on if the sentence is negated."""
        mixouts = [95, 135, 212, 222, 298]
        for part1 in el1:
            part1 = normalise_text(part1)
            for part3 in el3:
                part3 = normalise_text(part3)
                if ' not ' in part1 and ' not ' in part3:
                    continue
                # if ' not ' in part1 and ' always ' in part3:
                if ' not ' in part1 and ' always ' in part3 and condi not in mixouts:
                    continue
                if ' like' in part1 and ' likely' not in part1:
                    if part3.split()[0] in proper_nouns:
                        sen_fin = part1 + ' ' + part3
                    else:
                        sen_fin = part1 + ' ' + part3[0].lower() + part3[1:]
                else:
                    if part3.split()[0] in proper_nouns:
                        sen_fin = part1 + ' that ' + part3
                    else:
                        sen_fin = part1 + ' that ' + part3[0].lower() + part3[1:]
                # if ' not ' in part1 and ' is always ' in part3:
                #     continue
                # if ' not ' in part1 and ' are always ' in part3:
                #     continue
                if ' not ' in part1 and ' never ' in part3:
                    sen_fin = sen_fin.replace('never ', '')
                    sorted_phrases = sorted(not_to_never_map.keys(), key=len, reverse=True)
                    for phrase in sorted_phrases:
                        # Use regex to replace whole word phrases
                        pattern = r'\b' + re.escape(phrase) + r'\b'
                        sen_fin = re.sub(pattern, not_to_never_map[phrase], sen_fin)
                sen_fin = sen_fin[0].upper() + sen_fin[1:]
                nots = sen_fin.count(' not ')
                nevers = sen_fin.count(' never ')
                negation = nots + nevers
                position = -1
                if ' not ' in part1 or ' never ' in part1:
                    position = 1
                elif ' not ' in part3 or ' never ' in part3:
                    position = 0

                sentences.append((sen_fin, position, negation))

    combine_elements(el1pos + el1neg, sen)
    return sentences


df['exp1sen'] = df.apply(lambda row: construct_sen(begs_pos, begs_neg, [row['sentence']], row['mixout']), axis=1)

df = df.explode('exp1sen')


negation_elements = ['do not', 'does not', 'is not', 'are not', 'was not', 'were not', 'must not', 'never']

df.reset_index(inplace=True)

for idx, row in df.iterrows():
    sentence = row['exp1sen'][0]
    if df.at[idx, 'condi'] == 'AT':
        new_condi = 'NF'
    elif df.at[idx, 'condi'] == 'AF':
        new_condi = 'NT'
    else:
        new_condi = df.at[idx, 'condi']

    if any(element in sentence.lower() for element in negation_elements):
        df.at[idx, 'condi'] = new_condi


def insert_neg(sentence):
    words = sentence.split()
    neg_indices = [i for i, word in enumerate(words) if word in ['is', 'are', 'does', 'do', 'was', 'were', 'must', 'should', 'may']]
    always_indices = [i for i, word in enumerate(words) if word in ['always']]
    never_indices = [i for i, word in enumerate(words) if word in ['never']]
    if neg_indices:
        last_index = neg_indices[-1]
        if last_index + 1 < len(words) and words[last_index + 1] == 'always':
            words[last_index + 1] = 'never'
        elif last_index + 1 < len(words) and words[last_index + 1] == 'sometimes' and always_indices:
            words[always_indices[-1]] = 'never'
        elif not never_indices:
            words.insert(last_index + 1, 'not')
    return ' '.join(words)


df[['full_sen', 'raised_position', 'negation']] = pd.DataFrame(df['exp1sen'].tolist(), index=df.index)

df['question'] = df['rate']

df['rate'] = df['rate'].map(lambda x: ''.join([x, '.']))

df['question'] = df.apply(lambda row: mask_word(row['rate'], row['cw']), axis=1)

# getting rid of not in the question column
df['question'] = df['question'].map(lambda sen: sen.replace(' not', ''))
df['question'] = df['question'].map(lambda sen: sen.replace(' never', ' always'))

df['sec_sen_condi'] = 'A'

even_new_df = df.copy()
neg_ques = []

for idx, row in even_new_df.iterrows():
    new_row = row.copy()
    new_row['question'] = insert_neg(new_row['question'])
    new_row['sec_sen_condi'] = 'N'
    neg_ques.append(new_row)

df_neg_ques = pd.DataFrame(neg_ques)
df_with_neg_ques = pd.concat([df, df_neg_ques], ignore_index=False)

condition_map = {
    ('AT', 'A'): 'AT_A',
    ('AF', 'A'): 'AF_A',
    ('AT', 'N'): 'AT_N',
    ('AF', 'N'): 'AF_N',
    ('NT', 'A'): 'NT_A',
    ('NF', 'A'): 'NF_A',
    ('NT', 'N'): 'NT_N',
    ('NF', 'N'): 'NF_N'
}

df_with_neg_ques['new_condi'] = df_with_neg_ques.apply(lambda row: condition_map.get((row['condi'], row['sec_sen_condi'])), axis=1)

conj_A = 'So,'
conj_N = 'Nevertheless,'

df_with_neg_ques['conjuncture'] = False

new_df_with_neg_ques = df_with_neg_ques.copy()

sen_with_conj = []

for idx, row in new_df_with_neg_ques.iterrows():
    if row['new_condi'] in ['NT_A', 'NF_A', 'AT_N', 'AF_N']:
        new_row = row.copy()
        new_row['conjuncture'] = True
        if row['question'].split()[0] not in proper_nouns:
            new_row['question'] = combine_sen(conj_N, new_row['question'].lower()[0] + new_row['question'][1:])
        else:
            new_row['question'] = combine_sen(conj_N, new_row['question'])
    else:
        new_row = row.copy()
        new_row['conjuncture'] = True
        if row['question'].split()[0] not in proper_nouns:
            new_row['question'] = combine_sen(conj_A, new_row['question'].lower()[0] + new_row['question'][1:])
        else:
            new_row['question'] = combine_sen(conj_A, new_row['question'])
    sen_with_conj.append(new_row)

df_conj = pd.DataFrame(sen_with_conj)

df_fin = pd.concat([df_with_neg_ques, df_conj], ignore_index=False)

for d in [df, df_conj, df_neg_ques, df_fin]:
    d['sen_ques'] = d.apply(lambda row: combine_sen(row['full_sen'], row['question']), axis=1)

for idx, row in df_fin.iterrows():
    sen_split = row['sen_ques'].split('.')
    counterA = Counter(sen_split[0].split())
    counterB = Counter(sen_split[1].split())
    df_fin.at[idx, 'cos_sim_sen_ques'] = counter_cosine_sim(counterA, counterB)

