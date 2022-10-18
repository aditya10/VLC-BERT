import _init_paths
import os
import argparse
import torch
import subprocess
import json

from okvqa.function.config import config, update_config
from okvqa.function.train import train_net
from okvqa.function.test import test_net
from external.PythonEvaluationTools.okvqa_vqaEval import run_eval


def parse_args():
    parser = argparse.ArgumentParser('Train Cognition Network')
    parser.add_argument('--cfg', type=str, help='path to config file')
    parser.add_argument('--model-dir', type=str, help='root path to store checkpoint')
    parser.add_argument('--log-dir', type=str, help='tensorboard log dir')
    parser.add_argument('--dist', help='whether to use distributed training', default=False, action='store_true')
    parser.add_argument('--slurm', help='whether this is a slurm job', default=False, action='store_true')
    parser.add_argument('--do-test', help='whether to generate csv result on test set',
                        default=True, action='store_true')
    parser.add_argument('--cudnn-off', help='disable cudnn', default=False, action='store_true')

    # easy test pretrain model
    parser.add_argument('--partial-pretrain', type=str)

    args = parser.parse_args()

    if args.cfg is not None:
        update_config(args.cfg)
    if args.model_dir is not None:
        config.OUTPUT_PATH = os.path.join(args.model_dir, config.OUTPUT_PATH)

    if args.partial_pretrain is not None:
        config.NETWORK.PARTIAL_PRETRAIN = args.partial_pretrain

    if args.slurm:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = str(29500)
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)

    return args, config

def _load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def prune_res_file(res_file):
    res = _load_json(res_file)
    annot = _load_json('data/coco/okvqa/OpenEnded_mscoco_val2014_questions_pruned.json')

    res_pruned = []
    for a in annot['questions']:
        qid = a['question_id']

        for r in res:
            if r['question_id'] == qid:
                res_pruned.append(r)
                break
    
    res_pruned_path = res_file[:-5]+'_pruned.json'
    with open(res_pruned_path, 'w') as f:
        json.dump(res_pruned, f)
    
    return res_pruned_path

def main():
    args, config = parse_args()
    rank, model = train_net(args, config)
    if args.do_test and (rank is None or rank == 0):
        res_path, save_path = test_net(args, config)
        run_eval(res_path, save_path, pruned=False)
        res_pruned_path = prune_res_file(res_path)
        run_eval(res_pruned_path, save_path, pruned=True)

if __name__ == '__main__':
    main()


