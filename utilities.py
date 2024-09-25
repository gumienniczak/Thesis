import pandas as pd
import contractions
import re
import math


with open('negation_items.xlsx', 'rb') as file:
    data_df = pd.read_excel(file)

beg_opinion = ['they think', 'they believe', 'they suppose', 'they imagine', 'they expect', 'they reckon', 'they feel']
beg_opinion_people = ['people think', 'people believe', 'people suppose', 'people imagine', 'people expect', 'people reckon', 'people feel']
beg_op_neg = ['they do not feel', 'they do not believe', 'they do not suppose', 'they do not expect', 'they do not reckon']
beg_op_ppl_neg = ['people do not think', 'people do not believe', 'people do not suppose', 'people do not imagine', 'people do not people do not expect', 'people do not reckon', 'people do not feel']
beg_perception = ['it seems that', 'it appears', 'it looks like', 'it sounds like', 'it feels like']
beg_per_neg = ['it does not seem like', 'it does not appear', 'it does not look like', 'it does not feel like']
beg_probability = ['it is probable', 'it is likely', 'it figures']
beg_prob_neg = ['it is not probable', 'it is not likely', 'it does not figure']
beg_judgement = ['they suggest that']
beg_jug_ppl = ['people suggest']
beg_jud_neg = ['they do not suggest']
beg_jud_ppl = ['people do not suggest']
that = ' that'

begs = beg_opinion + beg_perception + beg_probability + beg_judgement
begs_neg = beg_op_neg + beg_per_neg + beg_prob_neg + beg_jud_neg


beg_opinion = ['they think', 'they believe', 'they suppose', 'they imagine', 'they expect', 'they reckon', 'they feel']
df = data_df.copy()

# the code below simplifies the sentences, so they do not contain additional conjunctions and parts of the sentence.
for idx, row in df.iterrows():
    string = df.at[idx, 'sentence']
    conj = r'\b' + re.escape(df.at[idx, 'cw_plus1']) + r'\b'
    match = re.search(conj, row['sentence'])
    if match:
        new_sentence = df.at[idx, 'sentence'][:match.start()].strip()
        df.at[idx, 'sentence'] = new_sentence + '.'
    else:
        df.at[idx, 'sentence'] = df.at[idx, 'sentence'].strip() + '.'

# expanding contractions in sentence and rate
df['sentence'] = df['sentence'].map(lambda sentence: contractions.fix(sentence))
df['rate'] = df['rate'].map(lambda rate:contractions.fix(rate))

# making the first letter in the sentence lowercase
df['sentence'] = df['sentence'].map(lambda sentence: sentence[0].lower() + sentence[1:])

# replacing values which were converted to Bool back to string
df['cw'] = df['cw'].apply(lambda verb: verb.replace('TRUE', 'true').replace('FALSE', 'false'))


def duplicate_rows(df, column, values_to_duplicate):
    """ duplicating rows in a given dataframe. column is the target column.
    values_to_duplicate allow for a specification of specific values for rows meant to be duplicated.
    Rows present in the original df get assigned True values, while the duplicated ones, False."""
    df['original'] = True
    df['original_type'] = df[column]
    rows_to_duplicate = df[df[column].isin(values_to_duplicate)].copy()
    rows_to_duplicate['original'] = False
    df_duplicated = pd.concat([df, rows_to_duplicate]).sort_index(kind='stable').reset_index(drop=True)
    return df_duplicated


# duplicating rows in the dataframe
df = duplicate_rows(df, 'condi', ['NF', 'NT'])

# for beg in beg_perception[:1]:
#     df.loc[df['original'] == True, 'sentence'] = df['sentence'].map(lambda sentence: ' '.join((beg, sentence)))

combined_sentences = []
# creating sentences suitable for neg-rising
for beg in begs:
    for idx, row in df.iterrows():
        if df.at[idx, 'original']:
            new_sentence = f'{beg} {row['sentence']}'
            combined_sentence = row.to_dict()
            combined_sentence['sentence'] = new_sentence
            combined_sentences.append(combined_sentence)


combined_df = pd.DataFrame(combined_sentences)

df = pd.concat([df, combined_df], ignore_index=True)

# in duplicated sentences, getting rid of not in the subordinate clause
df.loc[df['original'] == False, 'sentence'] = df.loc[df['original'] == False, 'sentence'].str.replace(' not', '')


# in duplicated sentences, joining the beginning of the sentence.
for beg in beg_per_neg[:1]:
    df.loc[df['original'] == False, 'sentence'] = df['sentence'].map(lambda sentence: ' '.join((beg, sentence)))

# making the first character in a sentence uppercase
df['sentence'] = df['sentence'].apply(lambda sen: sen[0].upper() + sen[1:] if sen else sen)

# duplicating rate into a new columns, question
df['question'] = df['rate']


def replace_with_mask(sentence, word_to_replace, replacement):
    """replacing given word in a sentence with a specified string"""
    return ' '.join([replacement if word == word_to_replace else word for word in sentence.split()])

def mask_word(rate, word):
    """replacing a word in a string with <mask>"""
    return rate.replace(word, '<mask>')


# masking the target predicate in rate

df['rate'] = df['rate'].map(lambda x: ''.join([x, '.']))

df['question'] = df.apply(lambda row: mask_word(row['rate'], row['cw']), axis=1)

# getting rid of not in the question column
df['question'] = df['question'].map(lambda sen: sen.replace(' not', ''))

#df['question'] = df['question'].map(lambda x: ''.join([x, '.']))

def combine_sen(sentence, question):
    """joining context sentence and sentence containing the mask"""
    return ' '.join([sentence, question])


# combining the context and question together
df['sen_ques'] = df.apply(lambda row: combine_sen(row['sentence'], row['question']), axis = 1)


itisraised = (df['original'] != False) & ((df['condi'] == 'NF') | (df['condi'] == 'NT'))

df['raised'] = itisraised

def replace_predicate(sentence, predicate):
    """replacing the target predicate with an appropriate token, in preparation for template generation"""
    return sentence.replace(predicate, '<predicate>')


def create_templates(df: pd.DataFrame) -> pd.DataFrame:
    """generating templates with an appropriate marking"""
    df_copy = df.copy()
    df['template'] = False
    df_copy['template'] = True
    return df_copy


df_templates = create_templates(df)

# replacing the target predicate with the token
df_templates['sentence'] = df_templates.apply(lambda row: replace_predicate(row['sentence'], row['cw']), axis=1)

df.to_pickle('examples.pkl')

def counter_cosine_sim(sen, ques):
    """ this calculates cosine similarity between two sentences in a pair."""
    terms = set(sen).union(ques)
    dotprod = sum(sen.get(k, 0) * ques.get(k, 0) for k in terms)
    magA = math.sqrt(sum(sen.get(k, 0) ** 2 for k in terms))
    magB = math.sqrt(sum(ques.get(k, 0) ** 2 for k in terms))
    return dotprod / (magA * magB)