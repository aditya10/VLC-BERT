import argparse
import json
import os

def load_aokvqa(aokvqa_dir, split, version='v1p0'):
    #assert split in ['train', 'val', 'test', 'test_w_ans', 'val_pruned']
    dataset = json.load(open(
        os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json")
    ))
    return dataset

def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")


def run_eval(resFile=None, split='test', save_path=None, multiple_choice=False, strict=True):
    # Load data
    dataset = load_aokvqa('data/coco/aokvqa', split=split)
  
    # Load predictions works only for direct answers
    if not multiple_choice:
        predictions = json.load(open(resFile, 'r'))
        preds = {}
        for d in predictions:
             preds[d['question_id']] = d['answer']
        # for q in predictions.keys():
        #     if 'direct_answer' in predictions[q].keys():
        #         da_predictions[q] = predictions[q]['direct_answer']
        

    if isinstance(dataset, list):
        dataset = { dataset[i]['question_id'] : dataset[i] for i in range(len(dataset)) }

    if multiple_choice:
        dataset = {k:v for k,v in dataset.items() if v['difficult_direct_answer'] is False}

    if strict:
        dataset_qids = set(dataset.keys())
        preds_qids = set(preds.keys())
        assert dataset_qids.issubset(preds_qids)

    # dataset = q_id (str) : dataset element (dict)
    # preds = q_id (str) : prediction (str)

    acc = []

    for q in dataset.keys():
        if q not in preds.keys():
            acc.append(0.0)
            continue

        pred = preds[q]
        choices = dataset[q]['choices']
        direct_answers = dataset[q]['direct_answers']

        ## Multiple Choice setting
        if multiple_choice:
            if strict:
                assert pred in choices, 'Prediction must be a valid choice'
            correct_choice_idx = dataset[q]['correct_choice_idx']
            acc.append( float(pred == choices[correct_choice_idx]) )
        ## Direct Answer setting
        else:
            num_match = sum([pred == da for da in direct_answers])
            vqa_acc = min(1.0, num_match / 3.0)
            acc.append(vqa_acc)

    acc = sum(acc) / len(acc) * 100
    # print accuracies
    print("\n")
    print("Overall Accuracy is: %.02f\n" % (acc))

    return acc

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--resFile', type=str, help='Path to the json file with predictions')
    parser.add_argument('--split', type=str, help='Split to evaluate on')

    opt = parser.parse_args()
    run_eval(opt.resFile, split=opt.split)