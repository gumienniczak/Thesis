import pandas as pd
import contractions
import re
from utilities import mask_word, replace_with_mask, combine_sen
from prepare_examples_exp1 import insert_neg, counter_cosine_sim
from collections import Counter

pd.set_option('display.max_colwidth', None)

with open('negation_items.xlsx', 'rb') as file:
    data_df = pd.read_excel(file)

df = data_df.copy()

#print(len(df))

# sen1 = 'it seems'
# sen1neg = 'it does not seem'
# sen2 = 'the newspapers claim'

# sen3 = 'I do not believe that he wants me to think that he did it'
# sen3_1 = 'I believe that he does not want me to think that he did it'
# sen3_2 = 'I believe that he does not want me to think that he did not do it'

proper_nouns = ['Al', 'Saddam', 'Michael', 'Ozzy', 'Ellen', 'Elton', 'Tom', 'Ryan', 'Oprah', 'Justin', 'Beyonce', 'Shaquille']

sen1s_pos = ['I believe', 'I think', 'I suppose', 'I imagine', 'I expect', 'I reckon', 'I feel', 'It seems',
             'It appears', 'It looks like', 'It sounds like', 'It feels like', 'It is probable', 'It is likely',
             'It figures', 'I suggest']
sen1s_neg = ['I do not believe', 'I do not think', 'I do not suppose', 'I do not imagine', 'I do not expect',
             'I do not reckon', 'I do not feel', 'It does not seem', 'It does not appear', 'It does not look like',
             'It does not sound like', 'It does not feel like', 'It is not probable', 'It is not likely',
             'It does not figure', 'I do not suggest']
# sen2s_pos = ['he wants me to think', 'he wants me to suppose', 'he wants me to imagine', 'he wants me to expect',
#              'he wants me to reckon', 'he wants me to feel']
sen2s_pos = ['he wants me to think']
# sen2s_neg = ['he does not want me to think', 'he does not want me to suppose', 'he does not want me to imagine',
#              'he does not want me to expect', 'he does not want me to reckon', 'he does not want me to feel']
sen2s_neg = ['he does not want me to think']
sen1 = 'I believe'
sen1_1 = 'I think'
sen1_2 = 'I suppose'
sen1_neg = 'I do not believe'
sen1_1neg = 'I do not think'
sen1_2neg = 'I do not suppose'
sen2 = 'he wants me to think'
sen2_1 = 'he wants me to believe'
sen2_2 = 'he wants me to suppose'
sen2_neg = 'he does not want me to think'
sen2_1neg = 'he does not want me to believe'
sen2_2neg = 'he does not want me to suppose'
# sen_AT = 'Investments in Iraqi oil are very risky and can be very profitable, AT.'
# sen_AF = 'Investments in Iraqi oil are very safe and can be very profitable, AF.'
# sen_NT = 'Investments in Iraqi oil are not very safe and can be very profitable, NT.'
# sen_NF = 'Investments in Iraqi oil are not very risky and can be very profitable, NF.'

# beg_pos = [sen1, sen1_1, sen1_2]
# beg_neg = [sen1_neg, sen1_1neg, sen2_2neg]
# beg = [sen1, sen1_neg]
# mid_pos = [sen2, sen2_1, sen2_2]
# mid_neg = [sen2_neg, sen2_1neg, sen2_2neg]
# end_pos = [sen_AT, sen_AF]
# end_neg = [sen_NT, sen_NF]
# dum = [sen_AT, sen_AF, sen_NT, sen_NF]

beg_pos = sen1s_pos
beg_neg = sen1s_neg
mid_pos = sen2s_pos
mid_neg = sen2s_neg

df['cw'] = df['cw'].apply(lambda verb: verb.replace('TRUE', 'true').replace('FALSE', 'false'))
df['cw1'] = df['cw1'].apply(lambda verb: verb.replace('TRUE', 'true').replace('FALSE', 'false'))
df['cw2'] = df['cw2'].apply(lambda verb: verb.replace('TRUE', 'true').replace('FALSE', 'false'))

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

# replacing but with and
# df.loc[df['cw_plus1'] == 'but', 'sentence'] = df.loc[df['cw_plus1'] == 'but', 'sentence'].str.replace('but', 'and', case=False)

# expanding contractions in sentences and rate
df['sentence'] = df['sentence'].map(lambda sentence: contractions.fix(sentence))
df['rate'] = df['rate'].map(lambda rate: contractions.fix(rate))

# making the first letter in the sentence lowercase
df['sentence'] = df['sentence'].map(lambda sentence: sentence[0].lower() + sentence[1:])

# some values are not represented as strigns but as boolean values, below fixes that
df['cw'] = df['cw'].apply(lambda verb: verb.replace('TRUE', 'true').replace('FALSE', 'false'))

# some negative sentences do not contain the word not, instead they contain the word 'never'.
problematic = df[(df['condi'].isin(['NT', 'NF'])) & (~df['sentence'].str.contains('not', case=False))]
# print(problematic)

#the following replaces the word never with not always in the problematic rows.
# for idx, row in df.iterrows():
#     if row['condi'] in ['NT', 'NF'] and 'not' not in row['sentence'].lower():
#         df.at[idx, 'sentence'] = row['sentence'].replace('never', 'not always')

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

def construct_sen(el1pos, el1neg, el2pos, el2neg, sen, mixout):
    """This function takes positive and negative elements and combines simple sentence, sen, with these elements,
    forming complex sentences"""
    sentences = []


    def normalise_text(text):
        """Replaces non-breaking spaces with standard spaces in the text."""
        return text.replace('\xa0', ' ')

    def combine_elements(el1, el2, el3):
        """This function combines elements 1, 2 and 3, to form a complex sentence.
        Returns a tuple consisting of the sentence,
        together with values -1,0,1,2 for the position if not within the sentence,
        and values 0,1 depending on if the sentence is negated."""
        mixouts = [95, 135, 212, 222, 298]
        for part1 in el1:
            part1 = normalise_text(part1)
            for part2 in el2:
                part2 = normalise_text(part2)
                if ' not ' in part1 and ' not ' in part2:
                    continue
                for part3 in el3:
                    part3 = normalise_text(part3)
                    if (' not ' in part1 and ' not ' in part3) or (' not ' in part2 and ' not ' in part3):
                        continue
                    if (' not ' in part1 and ' always ' in part3 and mixout not in mixouts) or (' not ' in part2 and ' always ' in part3 and mixout not in mixouts):
                        continue
                    if ' like' in part1 and ' likely' not in part1:
                        if part3.split()[0] in proper_nouns:
                            sen_fin = part1 + ' ' + part2 + ' that ' + part3
                        else:
                            sen_fin = part1 + ' ' + part2 + ' that ' + part3[0].lower() + part3[1:]
                    else:
                        if part3.split()[0] in proper_nouns:
                            sen_fin = part1 + ' that ' + part2 + ' that ' + part3
                        else:
                            sen_fin = part1 + ' that ' + part2 + ' that ' + part3[0].lower() + part3[1:]
                    if (' not ' in part1 and ' never ' in part3) or (' not ' in part2 and ' never ' in part3):
                        sen_fin = sen_fin.replace('never ', '')
                        sorted_phrases = sorted(not_to_never_map.keys(), key=len, reverse=True)
                        for phrase in sorted_phrases:
                            # Use regex to replace whole word phrases
                            pattern = r'\b' + re.escape(phrase) + r'\b'
                            sen_fin = re.sub(pattern, not_to_never_map[phrase], sen_fin)
                    nots = sen_fin.count(' not ')
                    nevers = sen_fin.count(' never ')
                    negation = nots + nevers
                    position = -1
                    if ' not ' in part1 or ' never ' in part1:
                        position = 2
                    elif ' not ' in part2 or ' never ' in part2:
                        position = 1
                    elif ' not ' in part3 or ' never ' in part3:
                        position = 0

                    sentences.append((sen_fin, position, negation))

    combine_elements(el1pos + el1neg, el2pos + el2neg, sen)
    return sentences


df['exp2sen'] = df.apply(lambda row: construct_sen(beg_pos, beg_neg, mid_pos, mid_neg, [row['sentence']], row['mixout']),
                         axis=1)




# for idx, row in df.iterrows():
#     print(idx, len(row['exp2sen']), row['condi'])


df = df.explode('exp2sen')

# df['group_index'] = df.groupby(df.index).cumcount()

print(len(df), 'LEN1')


negation_elements = ['do not', 'does not', 'is not', 'are not', 'was not', 'were not', 'must not', 'never']

df.reset_index(inplace=True)


for idx, row in df.iterrows():
    sentence = row['exp2sen'][0]
    if df.at[idx, 'condi'] == 'AT':
        new_condi = 'NF'
    elif df.at[idx, 'condi'] == 'AF':
        new_condi = 'NT'
    else:
        new_condi = df.at[idx, 'condi']

    if any(element in sentence.lower() for element in negation_elements):
        df.at[idx, 'condi'] = new_condi



# for idx, row in df.iterrows():
#     print(idx, row['exp2sen'], row['condi'])


df[['full_sen', 'raised_position', 'negation']] = pd.DataFrame(df['exp2sen'].tolist(), index=df.index)

df['question'] = df['rate']

df['rate'] = df['rate'].map(lambda x: ''.join([x, '.']))

df['question'] = df.apply(lambda row: mask_word(row['rate'], row['cw']), axis=1)

# getting rid of not in the question column
df['question'] = df['question'].map(lambda sen: sen.replace(' not', ''))
df['question'] = df['question'].map(lambda sen: sen.replace(' never', ' always'))
df['sec_sen_condi'] = 'A'

new_df = df.copy()
neg_ques = []

for idx, row in df.iterrows():
    new_row = row.copy()
    new_row['question'] = insert_neg(new_row['question'])
    new_row['sec_sen_condi'] = 'N'
    neg_ques.append(new_row)

# for idx, row in df.iterrows():
#     print(idx, row['sen_ques'])

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

print(len(df_with_neg_ques), 'LEN2')

# sel_lines = df.loc[(df['raised_position'] == -1) & (df['condi'].isin(['NT', 'NF']))]


#print(df['raised_position'].value_counts())

# df_neg = df[df['condi'].isin(['NT', 'NF'])]
# df_pos = df[df['condi'].isin(['AT', 'AF'])]

#print(df_pos['raised_position'].value_counts())
#x = df_pos.loc[df_pos['raised_position'] == 2]
# rai_2 = df_pos.loc[df_pos['raised_position'] == 2]
#print(rai_2)

#print(df['exp2sen'][0])
#print(df['condi'][0])

conj_A = 'So,'
conj_N = 'Nevertheless,'

df_with_neg_ques['conjuncture'] = False

print(len(df))


new_df_with_neg_ques = df_with_neg_ques.copy()

sen_with_conj = []

for idx, row in new_df_with_neg_ques.iterrows():
    if row['new_condi'] in ['NT_A', 'NF_A', 'AT_N', 'AF_N']:
    # if row['condi'] in ['NT', 'NF']:
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

# for idx, row in df.head().iterrows():
#     print(idx, row)

# for idx, row in df_fin.iterrows():
#     print(idx, row['sen_ques'])

df_fin['sen_ques'] = df_fin.apply(lambda row: combine_sen(row['full_sen'], row['question']), axis=1)

# optimists = df_fin[df_fin['mixout'] == 81]

for idx, row in df_fin.iterrows():
    sen_split = row['sen_ques'].split('.')
    counterA = Counter(sen_split[0].split())
    counterB = Counter(sen_split[1].split())
    df_fin.at[idx, 'cos_sim_sen_ques'] = counter_cosine_sim(counterA, counterB)

print('DF_FIN COS SIM BY NEW CONDI')
print(df_fin.groupby('new_condi')['cos_sim_sen_ques'].mean() * 100)

# print(df_fin['condi'].value_counts())

target = df_fin[df_fin['new_condi'].isin(['AT_A', 'NT_N'])]

for idx, row in df_fin.iterrows():
    sen_split = row['sen_ques'].split('.')
    sen_split_lowercase = row['sen_ques'].lower().split('.')
    counterA = Counter(sen_split[0].split())
    counterB = Counter(sen_split[1].split())
    counterC = Counter(sen_split_lowercase[0].split())
    counterD = Counter(sen_split_lowercase[1].split())
    df_fin.at[idx, 'cos_sim_sen_ques'] = counter_cosine_sim(counterA, counterB)
    df_fin.at[idx, 'cos_sim_sen_ques_lowercase'] = counter_cosine_sim(counterC, counterD)

print('COS SIM UPPERCASE')
print(df_fin.groupby('new_condi')['cos_sim_sen_ques'].mean() * 100)

print('COS SIM LOWERCASE')
print(df_fin.groupby('new_condi')['cos_sim_sen_ques_lowercase'].mean() * 100)

