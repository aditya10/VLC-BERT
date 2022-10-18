# coding: utf-8
import argparse
import json

from external.PythonEvaluationTools.vqaEval import VQAEval
from external.PythonEvaluationTools.vqa_helper import VQA

def run_eval(resFile=None, save_path=None, pruned=False):

    # set up file names and paths
    taskType = 'OpenEnded'
    dataType = 'mscoco'
    dataSubType = 'val2014'
    data_dir = 'data/coco/okvqa'
    pruned_tag = '_pruned' if pruned else ''
    annFile = "%s/%s_%s_annotations%s.json" % (data_dir, dataType, dataSubType, pruned_tag)
    quesFile = "%s/%s_%s_%s_questions%s.json" % (data_dir, taskType, dataType, dataSubType, pruned_tag)

    fileTypes = ['accuracy', 'evalQA', 'evalQuesType', 'evalAnsType']
    output_dir = save_path

    [accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = ['%s/%s%s.json' % (output_dir, fileType, pruned_tag) for fileType in
                                                                    fileTypes]

    # create vqa object and vqaRes object
    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(resFile, quesFile)

    # create vqaEval object by taking vqa and vqaRes
    vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2

    # evaluate results
    """
    If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
    By default it uses all the question ids in annotation file
    """
    vqaEval.evaluate()

    question_types = {
        "eight": "Plants and Animals",
        "nine": "Science and Technology",
        "four": "Sports and Recreation",
        "six": "Geography, History, Language and Culture",
        "two": "Brands, Companies and Products",
        "other": "Other",
        "one": "Vehicles and Transportation",
        "five": "Cooking and Food",
        "ten": "Weather and Climate",
        "seven": "People and Everyday life",
        "three": "Objects, Material and Clothing"
        }

    # print accuracies
    print("\n")
    print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
    print("Per Question Type Accuracy is the following:")
    for quesType in vqaEval.accuracy['perQuestionType']:
        print("%s : %.02f" % (question_types[quesType], vqaEval.accuracy['perQuestionType'][quesType]))
    print("\n")
    print("Per Answer Type Accuracy is the following:")
    for ansType in vqaEval.accuracy['perAnswerType']:
        print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
    print("\n")

    #save evaluation results to ./Results folder
    json.dump(vqaEval.accuracy, open(accuracyFile, 'w'))
    json.dump(vqaEval.evalQA, open(evalQAFile, 'w'))
    json.dump(vqaEval.evalQuesType, open(evalQuesTypeFile, 'w'))
    json.dump(vqaEval.evalAnsType, open(evalAnsTypeFile, 'w'))

    return vqaEval.accuracy['overall']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resFile', type=str, help='Path to the json file with predictions')
    parser.add_argument('--savepath', type=str, help='Save path')
    parser.add_argument('--pruned', action='store_true', help='Whether to use pruned annotations')

    opt = parser.parse_args()
    run_eval(opt.resFile, opt.savepath, opt.pruned)