import os
import json
import _pickle as cPickle
from PIL import Image
import re
import base64
import numpy as np
import csv
import sys
import time
import logging
import pickle5 as pickle

import torch
from torch.utils.data import Dataset
from external.pytorch_pretrained_bert import BertTokenizer

from common.utils.zipreader import ZipReader
from common.utils.create_logger import makedirsExist

from pycocotools.coco import COCO

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']


class AOKVQA(Dataset):
    def __init__(self, image_set, root_path, data_path, answer_vocab_file, use_imdb=True,
                 with_precomputed_visual_feat=False, boxes="36",
                 transform=None, test_mode=False,
                 zip_mode=False, cache_mode=False, cache_db=True, ignore_db_cache=True,
                 tokenizer=None, pretrained_model_name=None,
                 add_image_as_a_box=False, mask_size=(14, 14),
                 aspect_grouping=False, use_sbert=False, commonsense_exp_name='', max_commonsense_len=5, 
                 commonsense_emb_type='', learn_attn=False, **kwargs):
        """
        Visual Question Answering Dataset

        :param image_set: image folder name
        :param root_path: root path to cache database loaded from annotation file
        :param data_path: path to vcr dataset
        :param transform: transform
        :param test_mode: test mode means no labels available
        :param zip_mode: reading images and metadata in zip archive
        :param cache_mode: cache whole dataset to RAM first, then __getitem__ read them from RAM
        :param ignore_db_cache: ignore previous cached database, reload it from annotation file
        :param tokenizer: default is BertTokenizer from pytorch_pretrained_bert
        :param add_image_as_a_box: add whole image as a box
        :param mask_size: size of instance mask of each object
        :param aspect_grouping: whether to group images via their aspect
        :param kwargs:
        """
        super(AOKVQA, self).__init__()

        assert not cache_mode, 'currently not support cache mode!'

        aokvqa_question = {
            "train2017": "aokvqa/aokvqa_v1p0_train.json",
            "val2017": "aokvqa/aokvqa_v1p0_val.json",
            "test2017": "aokvqa/aokvqa_v1p0_test.json",
        }

        if boxes == "36":
            precomputed_boxes = {
                'train2017': ("vgbua_res101_precomputed", "trainval_resnet101_faster_rcnn_genome_36"),
                'val2017': ("vgbua_res101_precomputed", "trainval_resnet101_faster_rcnn_genome_36"),
                'test2017': ("vgbua_res101_precomputed", "test2015_resnet101_faster_rcnn_genome_36"),
            }
        elif boxes == "10-100ada":
            precomputed_boxes = {
                'train2017': ("vgbua_res101_precomputed", "trainval2014_resnet101_faster_rcnn_genome"),
                'val2017': ("vgbua_res101_precomputed", "trainval2014_resnet101_faster_rcnn_genome"),
                'test2017': ("vgbua_res101_precomputed", "test2015_resnet101_faster_rcnn_genome"),
            }
        else:
            raise ValueError("Not support boxes: {}!".format(boxes))
        
        coco_dataset = {
            "train2017": ("train2017", "annotations/instances_train2017.json"),
            "val2017": ("val2017", "annotations/instances_val2017.json"),
            "test2017": ("test2017", "annotations/image_info_test2017.json"),
        }

        commonsense_path = "data/coco/aokvqa/commonsense/"
        self.experiment_name = commonsense_exp_name
        self.use_sbert = use_sbert
        self.max_commonsense_len = max_commonsense_len
        self.commonsense_emb_type = commonsense_emb_type
        self.learn_attn = learn_attn

        if self.experiment_name == 'semqo':
            aokvqa_expansions = {
                'train2017': commonsense_path+'expansions/semq.o_aokvqa_train.json',
                'val2017': commonsense_path+'expansions/semq.o_aokvqa_val.json',
                'test2017': commonsense_path+'expansions/semq.o_aokvqa_test.json',
            }

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [';', r"/", '[', ']', '"', '{', '}',
                      '(', ')', '=', '+', '\\', '_', '-',
                      '>', '<', '@', '`', ',', '?', '!']

        print("Loading OK-VQA dataset: ", image_set)
        self.boxes = boxes
        self.test_mode = test_mode
        self.with_precomputed_visual_feat = with_precomputed_visual_feat
        self.data_path = data_path
        self.root_path = root_path
        with open(answer_vocab_file, 'r', encoding='utf8') as f:
            self.answer_vocab = [w.lower().strip().strip('\r').strip('\n').strip('\r') for w in f.readlines()]
            self.answer_vocab = list(filter(lambda x: x != '', self.answer_vocab))
            self.answer_vocab = [self.processPunctuation(w) for w in self.answer_vocab]
        self.image_sets = [iset.strip() for iset in image_set.split('+')]
        self.q_files = [os.path.join(data_path, aokvqa_question[iset]) for iset in self.image_sets]
        
        self.expansion_files = [aokvqa_expansions[iset] for iset in self.image_sets] \
            if (self.experiment_name != '') else [None for iset in self.image_sets]

        self.precomputed_box_files = [
            os.path.join(data_path, precomputed_boxes[iset][0],
                         '{0}.zip@/{0}'.format(precomputed_boxes[iset][1])
                         if zip_mode else precomputed_boxes[iset][1])
            for iset in self.image_sets]
        self.box_bank = {}
        self.coco_datasets = [(os.path.join(data_path,
                                            coco_dataset[iset][0],
                                            '{{:012d}}.jpg'.format(coco_dataset[iset][0]))
                               if not zip_mode else
                               os.path.join(data_path,
                                            coco_dataset[iset][0] + '.zip@/' + coco_dataset[iset][0],
                                            '{{:012d}}.jpg'.format(coco_dataset[iset][0])),
                               os.path.join(data_path, coco_dataset[iset][1]))
                              for iset in self.image_sets]
        self.transform = transform
        self.zip_mode = zip_mode
        self.cache_mode = cache_mode
        self.cache_db = cache_db
        self.ignore_db_cache = ignore_db_cache
        self.aspect_grouping = aspect_grouping
        self.cache_dir = os.path.join(root_path, 'cache')
        self.add_image_as_a_box = add_image_as_a_box
        self.mask_size = mask_size
        if not os.path.exists(self.cache_dir):
            makedirsExist(self.cache_dir)
        self.tokenizer = tokenizer if tokenizer is not None \
            else BertTokenizer.from_pretrained(
            'bert-base-uncased' if pretrained_model_name is None else pretrained_model_name,
            cache_dir=self.cache_dir)

        if zip_mode:
            self.zipreader = ZipReader()

        self.database = self.load_annotations()
        if self.aspect_grouping:
            self.group_ids = self.group_aspect(self.database)

        self.attn_gt = None
        if self.learn_attn and not self.test_mode:
            self.attn_gt = self._load_json('data/coco/aokvqa/'+self.experiment_name+'_aokvqa_train_attn_annot_'+str(self.max_commonsense_len)+'.json')

    @property
    def data_names(self):
        if self.test_mode:
            return ['image', 'boxes', 'im_info', 'question', 'expansions', 'c_emb']
        else:
            return ['image', 'boxes', 'im_info', 'question', 'expansions', 'c_emb', 'label']

    def __getitem__(self, index):
        idb = self.database[index]

        # image, boxes, im_info
        boxes_data = self._load_json(idb['box_fn'])
        if self.with_precomputed_visual_feat:
            image = None
            w0, h0 = idb['width'], idb['height']

            boxes_features = torch.tensor(
                np.frombuffer(self.b64_decode(boxes_data['features']), dtype=np.float32).reshape((boxes_data['num_boxes'], -1))
            )
        else:
            image = self._load_image(idb['image_fn'])
            w0, h0 = image.size
        boxes = torch.tensor(
            np.frombuffer(self.b64_decode(boxes_data['boxes']), dtype=np.float32).reshape(
                (boxes_data['num_boxes'], -1))
        )

        if self.add_image_as_a_box:
            image_box = torch.as_tensor([[0.0, 0.0, w0 - 1, h0 - 1]])
            boxes = torch.cat((image_box, boxes), dim=0)
            if self.with_precomputed_visual_feat:
                if 'image_box_feature' in boxes_data:
                    image_box_feature = torch.as_tensor(
                        np.frombuffer(
                            self.b64_decode(boxes_data['image_box_feature']), dtype=np.float32
                        ).reshape((1, -1))
                    )
                else:
                    image_box_feature = boxes_features.mean(0, keepdim=True)
                boxes_features = torch.cat((image_box_feature, boxes_features), dim=0)
        im_info = torch.tensor([w0, h0, 1.0, 1.0])
        flipped = False
        if self.transform is not None:
            image, boxes, _, im_info, flipped = self.transform(image, boxes, None, im_info, flipped)

        # clamp boxes
        w = im_info[0].item()
        h = im_info[1].item()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h - 1)

        # flip: 'left' -> 'right', 'right' -> 'left'
        q_tokens = self.tokenizer.tokenize(idb['question'])
        if flipped:
            q_tokens = self.flip_tokens(q_tokens, verbose=False)
        if not self.test_mode:
            answers = idb['answers']
            if flipped:
                answers_tokens = [a.split(' ') for a in answers]
                answers_tokens = [self.flip_tokens(a_toks, verbose=False) for a_toks in answers_tokens]
                answers = [' '.join(a_toks) for a_toks in answers_tokens]
            label = self.get_soft_target(answers)

        # question
        q_retokens = q_tokens
        q_ids = self.tokenizer.convert_tokens_to_ids(q_retokens)

        # commonsense
        exp_ids = []
        commonsense_embeddings = torch.tensor([0])

        if self.experiment_name != '':

            # If we use SBERT, add [MASK] tokens exp_ids, and load the embeddings in commonsense_embeddings
            if self.use_sbert:
                
                if self.commonsense_emb_type == 'fusion':
                    commonsense_embeddings = self.get_cached_expansion_emb(idb['image_fn'].split('/')[-1], idb['question_id'], custom_tag='_ques')
                else:
                    commonsense_embeddings = self.get_cached_expansion_emb(idb['image_fn'].split('/')[-1], idb['question_id'])

                # Now that we have commonsense embeddings, we add the [MASK] tokens that will be replaced by the commonsense embeddings in training code
                if self.commonsense_emb_type == 'fusion':
                    m_tokens = ['[MASK]']
                else:
                    m_tokens = ['[MASK]']*self.max_commonsense_len
                
                m_ids = self.tokenizer.convert_tokens_to_ids(m_tokens)
                exp_ids += m_ids

            # If not SBERT, clean the picked expansions and add them to exp_ids
            else:
                
                # We use picked expansions from knowlege selection process
                picked_exp = idb['picked_exp']

                if isinstance(picked_exp, list):
                    picked_exp = picked_exp[0]
                
                picked_exp = picked_exp.split('.')
                picked_exp = [sentence.strip() for sentence in picked_exp]
                picked_exp = [sentence+'.' for sentence in picked_exp if sentence != '']

                if len(picked_exp) >= self.max_commonsense_len:
                    picked_exp = picked_exp[:self.max_commonsense_len]
                else:
                    picked_exp = picked_exp + [''] * (self.max_commonsense_len - len(picked_exp))

                picked_exp = ' '.join(picked_exp)
                picked_exp_tokens = self.tokenizer.tokenize(picked_exp)
                exp_ids += self.tokenizer.convert_tokens_to_ids(picked_exp_tokens)

        # concat box feature to box
        if self.with_precomputed_visual_feat:
            boxes = torch.cat((boxes, boxes_features), dim=-1)

        if self.attn_gt is not None:
            if str(idb['image_id']) in self.attn_gt and idb['question_id'] in self.attn_gt[str(idb['image_id'])]:
                attn_weight_label = torch.tensor(self.attn_gt[str(idb['image_id'])][idb['question_id']])
            else:
                attn_weight_label = torch.zeros(self.max_commonsense_len+1)
            label = torch.cat((label, attn_weight_label), dim=0)

        if self.test_mode:
            return image, boxes, im_info, q_ids, exp_ids, commonsense_embeddings
        else:
            return image, boxes, im_info, q_ids, exp_ids, commonsense_embeddings, label 

    @staticmethod
    def flip_tokens(tokens, verbose=True):
        changed = False
        tokens_new = [tok for tok in tokens]
        for i, tok in enumerate(tokens):
            if tok == 'left':
                tokens_new[i] = 'right'
                changed = True
            elif tok == 'right':
                tokens_new[i] = 'left'
                changed = True
        if verbose and changed:
            logging.info('[Tokens Flip] {} -> {}'.format(tokens, tokens_new))
        return tokens_new

    @staticmethod
    def b64_decode(string):
        return base64.decodebytes(string.encode())

    def answer_to_ind(self, answer):
        if answer in self.answer_vocab:
            return self.answer_vocab.index(answer)
        else:
            return self.answer_vocab.index('<unk>')

    def get_soft_target(self, answers):

        soft_target = torch.zeros(len(self.answer_vocab), dtype=torch.float)
        answer_indices = [self.answer_to_ind(answer) for answer in answers]
        gt_answers = list(enumerate(answer_indices))
        unique_answers = set(answer_indices)

        for answer in unique_answers:
            accs = []
            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]

                matching_answers = [item for item in other_answers if item[1] == answer]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)
            avg_acc = sum(accs) / len(accs)

            if answer != self.answer_vocab.index('<unk>'):
                soft_target[answer] = avg_acc

        return soft_target

    def processPunctuation(self, inText):

        if inText == '<unk>':
            return inText

        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("",
                                       outText,
                                       re.UNICODE)
        return outText

    def load_annotations(self):
        tic = time.time()
        database = []
        db_cache_name = 'aokvqa_boxes{}_{}'.format(self.boxes, '+'.join(self.image_sets))
        if self.with_precomputed_visual_feat:
            db_cache_name += 'visualprecomp'
        if self.zip_mode:
            db_cache_name = db_cache_name + '_zipmode'
        if self.test_mode:
            db_cache_name = db_cache_name + '_testmode'
        if self.experiment_name != '':
            db_cache_name = db_cache_name + '_' + self.experiment_name
        db_cache_root = os.path.join(self.root_path, 'cache')
        db_cache_path = os.path.join(db_cache_root, '{}.pkl'.format(db_cache_name))

        if os.path.exists(db_cache_path):
            if not self.ignore_db_cache:
                # reading cached database
                print('cached database found in {}.'.format(db_cache_path))
                with open(db_cache_path, 'rb') as f:
                    print('loading cached database from {}...'.format(db_cache_path))
                    tic = time.time()
                    database = cPickle.load(f)
                    print('Done (t={:.2f}s)'.format(time.time() - tic))
                    return database
            else:
                print('cached database ignored.')

        # ignore or not find cached database, reload it from annotation file
        print('loading database of split {}...'.format('+'.join(self.image_sets)))
        tic = time.time()

        for q_file, expansion_file, (coco_path, coco_annot), box_file \
                in zip(self.q_files, self.expansion_files, self.coco_datasets, self.precomputed_box_files):
            qs = self._load_json(q_file)
            expansion_data = self._load_json(expansion_file)
            coco = COCO(coco_annot)
            for q in qs:
                idb = {'image_id': q['image_id'],
                        'image_fn': coco_path.format(q['image_id']),
                        'width': coco.imgs[q['image_id']]['width'],
                        'height': coco.imgs[q['image_id']]['height'],
                        'box_fn': os.path.join(box_file, '{}.json'.format(q['image_id'])),
                        'question_id': q['question_id'],
                        'question': q['question'],
                        "picked_exp": expansion_data[str(coco_path.format(q['image_id']).split('/')[-1])][str(q['question_id'])] if (self.experiment_name != '') else None,
                        "rationales": q['rationales'] if self.experiment_name == 'rationales' else None,
                        'answers': q['direct_answers'] if not self.test_mode else None,
                        "question_type": "other" if not self.test_mode else None,
                        "answer_type": "other" if not self.test_mode else None,
                        }
                database.append(idb)

        print('Done (t={:.2f}s)'.format(time.time() - tic))

        # cache database via cPickle
        if self.cache_db:
            print('caching database to {}...'.format(db_cache_path))
            tic = time.time()
            if not os.path.exists(db_cache_root):
                makedirsExist(db_cache_root)
            with open(db_cache_path, 'wb') as f:
                cPickle.dump(database, f)
            print('Done (t={:.2f}s)'.format(time.time() - tic))

        return database

    @staticmethod
    def group_aspect(database):
        print('grouping aspect...')
        t = time.time()

        # get shape of all images
        widths = torch.as_tensor([idb['width'] for idb in database])
        heights = torch.as_tensor([idb['height'] for idb in database])

        # group
        group_ids = torch.zeros(len(database))
        horz = widths >= heights
        vert = 1 - horz
        group_ids[horz] = 0
        group_ids[vert] = 1

        print('Done (t={:.2f}s)'.format(time.time() - t))

        return group_ids

    def load_precomputed_boxes(self, box_file):
        if box_file in self.box_bank:
            return self.box_bank[box_file]
        else:
            in_data = {}
            with open(box_file, "r") as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
                for item in reader:
                    item['image_id'] = int(item['image_id'])
                    item['image_h'] = int(item['image_h'])
                    item['image_w'] = int(item['image_w'])
                    item['num_boxes'] = int(item['num_boxes'])
                    for field in (['boxes', 'features'] if self.with_precomputed_visual_feat else ['boxes']):
                        item[field] = np.frombuffer(base64.decodebytes(item[field].encode()),
                                                    dtype=np.float32).reshape((item['num_boxes'], -1))
                    in_data[item['image_id']] = item
            self.box_bank[box_file] = in_data
            return in_data

    def get_cached_expansion_emb(self, image_id, question_id, custom_tag=''):

        commonsense_embeddings = None

        for subset in self.image_sets:
            savepath = 'data/coco/sbert/aokvqa/'+self.experiment_name+'/'+str(self.max_commonsense_len)+custom_tag+'/'+subset
            
            image_id = str(image_id)
            question_id = str(question_id)

            if not os.path.exists(savepath+'/'+image_id+'.pkl'):
                continue
                
            with open(savepath+'/'+image_id+'.pkl', 'rb') as handle:
                unserialized_data = pickle.load(handle)
                commonsense_embeddings = torch.tensor(unserialized_data[question_id])

        assert commonsense_embeddings is not None, 'No expansion embedding found at {}'.format(savepath+'/'+image_id+'.pkl')
        return commonsense_embeddings

    def __len__(self):
        return len(self.database)

    def _load_image(self, path):
        if '.zip@' in path:
            return self.zipreader.imread(path).convert('RGB')
        else:
            return Image.open(path).convert('RGB')

    def _load_json(self, path):
        if path == None:
            return None
        elif '.zip@' in path:
            f = self.zipreader.read(path)
            return json.loads(f.decode())
        else:
            with open(path, 'r') as f:
                return json.load(f)
