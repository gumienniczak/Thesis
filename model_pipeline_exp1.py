import string
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from tqdm import tqdm


def obtain_predictions(df, model_name, top_k):
    new_df = df.copy()

    tokeniser = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    def decode(tokeniser, pred_idx, top_clean):
        ignore_tokens = string.punctuation + '[PAD]'
        tokens = []
        for w in pred_idx:
            token = ''.join(tokeniser.decode(w).split())
            if token not in ignore_tokens:
                tokens.append(token.replace('##', ''))
        return '\n'.join(tokens[:top_clean])

    def encode(tokeniser, text_sentence, add_special_tokens=True):
        # print(f'text sentence before replacing the token: {text_sentence}')
        text_sentence = text_sentence.replace('<mask>', tokeniser.mask_token)
        # print(f'text sentence after replacing the token: {text_sentence}')
        if tokeniser.mask_token == text_sentence.split()[-1]:
            text_sentence += ' .'

        input_ids = torch.tensor([tokeniser.encode(text_sentence, add_special_tokens=add_special_tokens)])

        print("Input IDs:", input_ids)

        mask_idx = torch.where(input_ids == tokeniser.mask_token_id)[1].tolist()

        print("Mask index:", mask_idx)

        if not mask_idx:
            print(f'mask idx not found')

        return input_ids, mask_idx[0]

    def get_all_predictions(text_sentence, top_clean=5):
        input_ids, mask_idx = encode(tokeniser, text_sentence)
        with torch.no_grad():
            predict = model(input_ids)

        attentions = predict[-1]
        tokens = tokeniser.convert_ids_to_tokens(input_ids[0])
        bert = decode(tokeniser, predict[0][0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
        return {'bert': bert, 'attentions': attentions, 'tokens': tokens}

    def get_prediction(input_text):
        res = get_all_predictions(input_text, top_clean=int(top_k))
        return res

    answer_list_of_lists = []
    targets = []

    new_df = new_df.reset_index(drop=True)
    for index, row in tqdm(new_df.iterrows(), total=len(df)):
        # Determine the target and not_target based on 'new_condi' value
        if row['new_condi'] in ['AT_A', 'NT_A']:
            target = row['cw1']
            not_target = row['cw2']
        elif row['new_condi'] in ['AF_A', 'NF_A']:
            target = row['cw2']
            not_target = row['cw1']
        elif row['new_condi'] in ['AT_N', 'NT_N']:
            target = row['cw2']
            not_target = row['cw1']
        elif row['new_condi'] in ['AF_N', 'NF_N']:
            target = row['cw1']
            not_target = row['cw2']
        else:
            raise ValueError(f"Unknown condition: {row['new_condi']}")

        # Print to verify correct assignment
        print(f"Index {index}: Setting target to {target} and target_opp to {not_target}")

        # Assign target and not_target to DataFrame
        new_df.at[index, 'target'] = target
        new_df.at[index, 'target_opp'] = not_target

        sen_ques = row['sen_ques']
        type_condi = row['new_condi']
        print(f'Processing sentence: {sen_ques}, {type_condi}')

        res = get_prediction(sen_ques) if len(sen_ques.split()) > 0 else {}

        if res:
            answer_list = res['bert'].split('\n')
            print(f'Answer_list: {answer_list}')

            answer_list_of_lists.append(answer_list)
            targets.append(target)

            # Update scores based on answer_list
            new_df.at[index, 'score_top1'] = int(answer_list[0] == target)
            new_df.at[index, 'score_top5'] = int(target in answer_list)
            new_df.at[index, 'score2_top5'] = int(target in answer_list and not_target not in answer_list)

            if target in answer_list or not_target in answer_list:
                target_position = answer_list.index(target) if target in answer_list else float('inf')
                opposite_position = answer_list.index(not_target) if not_target in answer_list else float('inf')
                new_df.at[index, 'score3_top5'] = int(target_position < opposite_position)
            else:
                new_df.at[index, 'score3_top5'] = 0

            new_df.at[index, 'answers'] = ', '.join(answer_list)

            # Capture attention and tokens if available in the response
            attention = res.get('attentions', [])
            tokens = res.get('tokens', [])

    # Save the updated DataFrame to a CSV file
    mod_name = model_name.split('/')[-1]
    new_df.to_csv(f'results_{mod_name}.csv', index=False)

    # Print a sample of the DataFrame to validate updates
    print(new_df.head())

    return new_df