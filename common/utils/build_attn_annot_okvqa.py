import json
import random
import numpy as np
from external.pytorch_pretrained_bert import BertTokenizer
import string
from nltk.corpus import stopwords
#nltk.download('stopwords')

DATASET = 'okvqa'
EXP_NAME = 'semqo'
MAX_COMMONSENSE_LEN = 5
RANDOM_SEED = 12345

random.seed(RANDOM_SEED)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
s = set(stopwords.words('english'))

def _load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def filename(exp_name):
    return (exp_name[:-1]+ "." + exp_name[-1]).lower()

def build_automatic():

    # Load expansions
    # Load answers
    # If answer is in expansion, give it a weight of 1
    # If answer is not in expansion, give it a weight of 0

    questions = _load_json('data/coco/okvqa/OpenEnded_mscoco_train2014_questions.json')
    questions = questions['questions']
    annotations = _load_json('data/coco/okvqa/mscoco_train2014_annotations.json')
    annotations = annotations['annotations']

    for annot in annotations:
        for question in questions:
            if question['question_id'] == annot['question_id']:
                annot['question'] = question['question']
                break
        
        direct_answers  = []
        for answer in annot['answers']:
            direct_answers.append(answer['answer'])
        
        annot['direct_answers'] = direct_answers


    expansions = _load_json('data/coco/okvqa/commonsense/expansions/'+filename(EXP_NAME)+'_okvqa_train.json')

    annot_size = 4000
    annotations_subset = random.sample(annotations, annot_size)

    attn_annot = {}
    good_counter = 0
    total_counter = 0
    bad_capacity = 500

    for annot in annotations_subset:

        question_id = annot['question_id']
        image_id = str(annot['image_id'])

        direct_answers = annot['direct_answers']

        exp = expansions['COCO_train2014_{:012d}.jpg'.format(annot['image_id'])][str(annot['question_id'])][0]
        print(exp)
        exp = exp.split('.')
        exp = [e.strip() for e in exp]
        exp = [e for e in exp if e != '']

        if len(exp) > MAX_COMMONSENSE_LEN:
            exp = exp[:MAX_COMMONSENSE_LEN]
        else:
            exp = exp + ['']*(MAX_COMMONSENSE_LEN-len(exp))
        
        weights, good = auto_annotator(exp, direct_answers)

        if not good and bad_capacity <= 0:
            continue
        
        if not good:
            bad_capacity -= 1

        if image_id not in attn_annot:
            attn_annot[image_id] = {}
        
        attn_annot[image_id][question_id] = weights

        total_counter += 1
        good_counter += 1 if good else 0

    with open('data/coco/okvqa/'+EXP_NAME+'_okvqa_train_attn_annot_'+str(MAX_COMMONSENSE_LEN)+'.json', 'w') as f:
        json.dump(attn_annot, f)

    print('Good: {}'.format(good_counter))
    print('Total: {}'.format(total_counter))

def auto_annotator(expansion_list, ans_list):

    ans_text = ' '.join(ans_list)
    ans_text = ans_text.translate(str.maketrans('', '', string.punctuation))
    ans_text = ans_text.lower()
    ans_tokens = tokenizer.tokenize(ans_text)
    ans_tokens = [t for t in ans_tokens if t not in s]

    final_weights = [0.05]*len(expansion_list)
    for i, expansion in enumerate(expansion_list):

        exp_text = expansion.translate(str.maketrans('', '', string.punctuation))
        exp_text = exp_text.lower()
        exp_tokens = tokenizer.tokenize(exp_text)
        exp_tokens = [t for t in exp_tokens if t not in s]

        for token in ans_tokens:
            if token in exp_tokens:
                final_weights[i] = 0.8
                break

    good = False
    if np.sum(final_weights) > (0.05*len(expansion_list)):
        final_weights = np.array(final_weights + [0.05])
        final_weights = final_weights / np.sum(final_weights)
        good = True
    else:
        final_weights = np.array(final_weights + [0.25])
        final_weights = final_weights / np.sum(final_weights)

    assert len(final_weights) == MAX_COMMONSENSE_LEN+1
    return final_weights.tolist(), good

if __name__ == '__main__':

    build_automatic()



