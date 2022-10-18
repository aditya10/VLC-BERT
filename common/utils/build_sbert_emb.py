from sentence_transformers import SentenceTransformer
import json
import pickle5 as pickle
import os
from tqdm import tqdm


DATASET = 'aokvqa'
EXP_NAME = 'semqo'
MAX_COMMONSENSE_LEN = 5
BASE_SAVE_PATH = 'data/coco'
USE_QUESTION = True

def filename(exp_name):
    return (exp_name[:-1]+ "." + exp_name[-1]).lower()

def _load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def job():

    if DATASET == 'okvqa':

        commonsense_path = 'data/coco/okvqa/commonsense/'
        commonsense_expansions_train = _load_json(commonsense_path+f'expansions/{filename(EXP_NAME)}_okvqa_train.json')
        commonsense_expansions_val = _load_json(commonsense_path+f'expansions/{filename(EXP_NAME)}_okvqa_val.json')

        if USE_QUESTION:
            question_dict_train = {}
            question_train = _load_json('data/coco/okvqa/OpenEnded_mscoco_train2014_questions_original.json')
            for q in question_train['questions']:
                question_dict_train[str(q['question_id'])] = q['question']
            
            question_dict_val = {}
            question_val = _load_json('data/coco/okvqa/OpenEnded_mscoco_val2014_questions.json')
            for q in question_val['questions']:
                question_dict_val[str(q['question_id'])] = q['question']
        else:
            question_dict_train = None
            question_dict_val = None

        build_and_save(commonsense_expansions_train, question_dict=question_dict_train, subset='train2014')
        build_and_save(commonsense_expansions_val, question_dict=question_dict_val, subset='val2014')

    elif DATASET == 'aokvqa':

        commonsense_path = 'data/coco/aokvqa/commonsense/'
        commonsense_expansions_train = _load_json(commonsense_path+f'expansions/{filename(EXP_NAME)}_aokvqa_train.json')
        commonsense_expansions_val = _load_json(commonsense_path+f'expansions/{filename(EXP_NAME)}_aokvqa_val.json')
        commonsense_expansions_test = _load_json(commonsense_path+f'expansions/{filename(EXP_NAME)}_aokvqa_test.json')

        if USE_QUESTION:
            question_dict_train = {}
            question_train = _load_json('data/coco/aokvqa/aokvqa_v1p0_train.json')
            for q in question_train:
                question_dict_train[str(q['question_id'])] = q['question']
            
            question_dict_val = {}
            question_val = _load_json('data/coco/aokvqa/aokvqa_v1p0_val.json')
            for q in question_val:
                question_dict_val[str(q['question_id'])] = q['question']

            question_dict_test = {}
            question_test = _load_json('data/coco/aokvqa/aokvqa_v1p0_test.json')
            for q in question_test:
                question_dict_test[str(q['question_id'])] = q['question']

        else:
            question_dict_train = None
            question_dict_val = None
            question_dict_test = None

        build_and_save(commonsense_expansions_train, subset='train2017', question_dict=question_dict_train)
        build_and_save(commonsense_expansions_val, subset='val2017', question_dict=question_dict_val)
        build_and_save(commonsense_expansions_test, subset='test2017', question_dict=question_dict_test)

    else:
        print('No dataset found: ', DATASET)


# Runs SBERT embeddings, and saves embeddings for all questions under each image in a pickle file
def build_and_save(commonsense_expansions, question_dict = {}, subset="train"):

    sentence_transformer_model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
    
    ques = '_ques' if USE_QUESTION else ''
    savepath = BASE_SAVE_PATH+'/sbert/'+DATASET+'/'+EXP_NAME+'/'+str(MAX_COMMONSENSE_LEN)+ques+'/'+subset

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    for image_id in tqdm(commonsense_expansions.keys()):
        im_embedding = {}
        for question_id in commonsense_expansions[image_id]:

            expansions = commonsense_expansions[image_id][question_id]
        
            if isinstance(expansions, list):
                commonsense_sentences = expansions[0]
            else:
                commonsense_sentences = expansions
            
            commonsense_sentences = commonsense_sentences.split('.')
            
            commonsense_sentences = [sentence.strip() for sentence in commonsense_sentences]
            commonsense_sentences = [sentence for sentence in commonsense_sentences if sentence != '']

            if len(commonsense_sentences) >= (MAX_COMMONSENSE_LEN):
                commonsense_sentences = commonsense_sentences[:(MAX_COMMONSENSE_LEN)]
            else:
                commonsense_sentences = commonsense_sentences + [''] * ((MAX_COMMONSENSE_LEN) - len(commonsense_sentences))

            if USE_QUESTION:
                question = question_dict[question_id]
                commonsense_sentences.append(question)

            im_embedding[question_id] = sentence_transformer_model.encode(commonsense_sentences, show_progress_bar=False)

        with open(savepath+'/'+image_id+'.pkl', 'wb') as f:
            pickle.dump(im_embedding, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
        job()