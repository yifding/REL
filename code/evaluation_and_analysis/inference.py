import os
import json
import requests
import argparse
import collections
from tqdm import tqdm
from load_ttl import (
    load_ttl_oke_2015,
    load_ttl_oke_2016,
    load_ttl_n3,
)

API_URL = "http://localhost:5555"


def load_tsv(file, key='', mode='char'):
    def process_token_2_char_4_doc_name2instance(token_doc_name2instance):
        char_doc_name2instance = dict()
        for doc_name, instance in token_doc_name2instance.items():
            starts = []
            ends = []
            entity_mentions = []
            entity_names = []
            assert doc_name == instance['doc_name']
            tokens = instance['tokens']
            sentence = ' '.join(tokens)
            token_entities = instance['entities']

            for token_start, token_end, token_entity_mention, token_entity_name in zip(
                    token_entities['starts'], token_entities['ends'], token_entities['entity_mentions'],
                    token_entities['entity_names']
            ):
                if not 0 <= token_start <= token_end < len(tokens):
                    print(instance)

                assert 0 <= token_start <= token_end < len(tokens)
                # **YD** sentence[char_start: char_end] == mention
                # **YD** ' '.join(tokens[token_start: token_end+1]) == mention ## ignoring the ',', '.' without space cases
                if token_start == 0:
                    start = 0
                else:
                    start = len(' '.join(tokens[:token_start])) + 1
                end = len(' '.join(tokens[:token_end + 1]))
                entity_mention = sentence[start: end]

                starts.append(start)
                ends.append(end)
                entity_mentions.append(entity_mention)
                entity_names.append(token_entity_name)

            char_doc_name2instance[doc_name] = {
                'doc_name': doc_name,
                # 'tokens': tokens,
                'sentence': sentence,
                'entities': {
                    "starts": starts,
                    "ends": ends,
                    "entity_mentions": entity_mentions,
                    "entity_names": entity_names,
                }
            }
        return char_doc_name2instance

    def generate_instance(
        doc_name,
        tokens,
        ner_tags,
        entity_mentions,
        entity_names,
        entity_wikipedia_ids,
    ):
        assert len(tokens) == len(ner_tags) == len(entity_mentions) \
               == len(entity_names) == len(entity_wikipedia_ids)

        instance_starts = []
        instance_ends = []
        instance_entity_mentions = []
        instance_entity_names = []
        instance_entity_wikipedia_ids = []

        tmp_start = -1
        for index, (ner_tag, entity_mention, entity_name, entity_wikipedia_id) in enumerate(
                zip(ner_tags, entity_mentions, entity_names, entity_wikipedia_ids)
        ):

            # judge whether current token is the last token of an entity, if so, generate an entity.
            if ner_tag == 'O':
                continue
            else:
                if ner_tag.startswith('B'):
                    # if the index hits the last one or next ner_tag is not 'I',
                    if index == len(tokens) - 1 or ner_tags[index + 1].startswith('B') or ner_tags[index + 1] == 'O':
                        instance_starts.append(index)
                        instance_ends.append(index)
                        # the end_index select the last token of a entity to allow simple index
                        # location for a transformer tokenizer
                        instance_entity_mentions.append(entity_mention)
                        instance_entity_names.append(entity_name)
                        instance_entity_wikipedia_ids.append(entity_wikipedia_id)
                        tmp_start = -1
                    else:
                        tmp_start = index
                else:
                    assert ner_tag.startswith('I')
                    if index == len(tokens) - 1 or ner_tags[index + 1].startswith('B') or ner_tags[index + 1] == 'O':
                        instance_starts.append(tmp_start)
                        instance_ends.append(index)
                        # the end_index select the last token of a entity to allow simple index
                        # location for a transformer tokenizer
                        instance_entity_mentions.append(entity_mention)
                        instance_entity_names.append(entity_name)
                        instance_entity_wikipedia_ids.append(entity_wikipedia_id)

                        assert tmp_start != -1
                        tmp_start = -1

        instance = {
            'doc_name': doc_name,
            'tokens': tokens,
            'entities': {
                "starts": instance_starts,
                "ends": instance_ends,
                "entity_mentions": instance_entity_mentions,
                "entity_names": instance_entity_names,
                "entity_wikipedia_ids": instance_entity_wikipedia_ids,
            }
        }
        return instance

    doc_name2dataset = dict()
    doc_name = ''
    tokens = []
    ner_tags = []
    entity_mentions = []
    entity_names = []
    entity_wikipedia_ids = []

    assert all(token != ' ' for token in tokens)

    with open(file) as reader:
        for line in reader:
            if line.startswith('-DOCSTART-'):
                if tokens:
                    assert doc_name != ''
                    assert doc_name not in doc_name2dataset

                    instance = generate_instance(
                        doc_name,
                        tokens,
                        ner_tags,
                        entity_mentions,
                        entity_names,
                        entity_wikipedia_ids,
                    )
                    if key in doc_name:
                        doc_name2dataset[doc_name] = instance

                    tokens = []
                    ner_tags = []
                    entity_mentions = []
                    entity_names = []
                    entity_wikipedia_ids = []

                assert line.startswith('-DOCSTART- (')
                tmp_start_index = len('-DOCSTART- (')
                if line.endswith(')\n'):
                    tmp_end_index = len(')\n')
                else:
                    tmp_end_index = len('\n')
                doc_name = line[tmp_start_index: -tmp_end_index]
                assert doc_name != ''

            elif line == '' or line == '\n':
                continue

            else:
                parts = line.rstrip('\n').split("\t")
                # len(parts) = [1, 4, 6, 7]
                # 1: single symbol
                # 4: ['Tim', 'B', "Tim O'Gorman", '--NME--'] or ['David', 'B', 'David', 'David_Beckham']
                # 6: ['House', 'B', 'House of Commons', 'House_of_Commons', 'http://en.wikipedia.org/wiki/House_of_Commons', '216091']
                # 7: ['German', 'B', 'German', 'Germany', 'http://en.wikipedia.org/wiki/Germany', '11867', '/m/0345h']
                assert len(parts) in [1, 4, 6, 7]

                # Gets out of unicode storing in the entity names
                # example: if s = "\u0027", in python, it will be represented as "\\u0027" and not recognized as an
                # unicode, should do .encode().decode("unicode-escape") to output "\'"
                if len(parts) >= 4:
                    parts[3] = parts[3].encode().decode("unicode-escape")

                # 1. add tokens
                # the extra space may destroy the position of token when creating sentences
                # tokens.append(parts[0].replace(' ', '_'))
                tokens.append(parts[0])

                # 2. add ner_tags
                if len(parts) == 1:
                    ner_tags.append('O')
                else:
                    ner_tags.append(parts[1])

                # 3. add entity_names
                if len(parts) == 1:
                    entity_mentions.append('')
                    entity_names.append('')
                else:
                    entity_mentions.append(parts[2])
                    if parts[3] == '--NME--':
                        entity_names.append('')
                    else:
                        entity_names.append(parts[3])

                # 4. add entity_wikiid if possible (only aida dataset has wikiid)
                if len(parts) >= 6 and int(parts[5]) > 0:
                    wikipedia_id = int(parts[5])
                    entity_wikipedia_ids.append(wikipedia_id)
                else:
                    entity_wikipedia_ids.append(-1)

    if tokens:
        assert doc_name != ''
        assert doc_name not in doc_name2dataset

        instance = generate_instance(
            doc_name,
            tokens,
            ner_tags,
            entity_mentions,
            entity_names,
            entity_wikipedia_ids,
        )
        if key in doc_name:
            doc_name2dataset[doc_name] = instance

    if mode == 'token':
        return doc_name2dataset
    else:
        assert mode == 'char', 'MODE(parameter) only supports "token" and "char"'
        return process_token_2_char_4_doc_name2instance(doc_name2dataset)


def process_pred(doc_name2inference, doc_name2instance):
    re_doc_name2inference = dict()

    for doc_name, inference in doc_name2inference.items():
        assert doc_name in doc_name2instance
        sentence = doc_name2instance[doc_name]['sentence'] if 'sentence' in doc_name2instance[doc_name] else ' '.join(doc_name2instance[doc_name]['tokens'])

        instance_starts = []
        instance_ends = []
        instance_entity_mentions = []
        instance_entity_names = []
        instance_entity_ner_labels = []
        instance_candidate_entities = []

        # inference = [
        # [0, 5, 'David', 'David', 0.12854342331601593, 0.9987859129905701, 'PER', 0, 0, 0],
        # [10, 8, 'Victoria_(Australia)', 'Victoria', 0.12246889451824833, 0.998676598072052, 'PER', 0, 0, 0]
        # ]

        for inference_ele in inference:
            char_start_index = inference_ele[0]
            char_end_index = inference_ele[0] + inference_ele[1]
            pred_entity = inference_ele[2]
            pred_mention = inference_ele[3]
            pred_ner_label = inference_ele[6] if len(inference_ele) >= 7 else 'NULL'
            pred_candidate_entities = inference_ele[7] if len(inference_ele) >= 8 else []

            instance_starts.append(char_start_index)
            instance_ends.append(char_end_index)
            instance_entity_mentions.append(pred_mention)
            instance_entity_names.append(pred_entity)
            instance_entity_ner_labels.append(pred_ner_label)
            instance_candidate_entities.append(pred_candidate_entities)

        instance = {
                'doc_name': doc_name,
                'sentence': sentence,
                'entities': {
                    "starts": instance_starts,
                    "ends": instance_ends,
                    "entity_ner_labels": instance_entity_ner_labels,
                    "entity_mentions": instance_entity_mentions,
                    "entity_names": instance_entity_names,
                    "entity_candidates": instance_candidate_entities,
                }
        }
        re_doc_name2inference[doc_name] = instance
    return re_doc_name2inference


def process_pred2token(doc_name2inference, doc_name2instance):
    re_doc_name2inference = dict()

    for doc_name, inference in doc_name2inference.items():
        assert doc_name in doc_name2instance
        tokens = doc_name2instance[doc_name]['tokens']
        sentence = ' '.join(tokens)

        instance_starts = []
        instance_ends = []
        instance_entity_mentions = []
        instance_entity_names = []
        instance_entity_ner_labels = []

        # inference = [
        # [0, 5, 'David', 'David', 0.12854342331601593, 0.9987859129905701, 'PER', 0, 0, 0],
        # [10, 8, 'Victoria_(Australia)', 'Victoria', 0.12246889451824833, 0.998676598072052, 'PER', 0, 0, 0]
        # ]

        str_index2token_index = dict()
        token_index = 0
        for str_index, st in enumerate(sentence):
            str_index2token_index[str_index] = token_index
            if st == ' ':
                token_index += 1

        for inference_ele in inference:
            str_start_index = inference_ele[0]
            str_end_index = inference_ele[0] + inference_ele[1]
            pred_entity = inference_ele[2]
            pred_mention = inference_ele[3]
            pred_ner_label = inference_ele[6] if len(inference_ele) >= 5 else 'NULL'

            if str_start_index not in str_index2token_index or str_end_index not in str_index2token_index:
                continue
            token_start_index = str_index2token_index[str_start_index]
            token_end_index = str_index2token_index[str_end_index]

            instance_starts.append(token_start_index)
            instance_ends.append(token_end_index)
            instance_entity_mentions.append(pred_mention)
            instance_entity_names.append(pred_entity)
            instance_entity_ner_labels.append(pred_ner_label)

        instance = {
                'doc_name': doc_name,
                'tokens': tokens,
                'entities': {
                    "starts": instance_starts,
                    "ends": instance_ends,
                    "entity_ner_labels": instance_entity_ner_labels,
                    "entity_mentions": instance_entity_mentions,
                    "entity_names": instance_entity_names,
                }
        }
        re_doc_name2inference[doc_name] = instance
    return re_doc_name2inference


def evaluate_and_analysis(doc_name2inference, doc_name2instance):
    def dev_by_zero(a, b):
        if b == 0:
            return 0.0
        else:
            return a / b
    # doc_name2inference: predict
    # doc_name2instance: gt

    num_pred_instance = 0
    num_gt_instance = 0
    num_true_positive = 0
    num_gold_true_positive = 0
    incorrect_entity_disambiguation_by_ner = collections.defaultdict(int)
    correct_entity_disambiguation_by_ner = collections.defaultdict(int)

    for doc_name in doc_name2instance:
        gt_entities = doc_name2instance[doc_name]['entities']
        position2entity_name = dict()
        for entity_start, entity_end, entity_name in zip(
                gt_entities['starts'], gt_entities['ends'], gt_entities['entity_names'],
        ):
            if entity_name != '':
                num_gt_instance += 1
            position2entity_name[(entity_start, entity_end)] = entity_name

        if doc_name not in doc_name2inference:
            continue
        else:
            pred_entities = doc_name2inference[doc_name]['entities']
            for entity_start, entity_end, entity_name, entity_ner_label in zip(
                    pred_entities['starts'], pred_entities['ends'], pred_entities['entity_names'],
                    pred_entities.get('entity_ner_labels', 'NULL')
            ):
                num_pred_instance += 1
                if (entity_start, entity_end) in position2entity_name and \
                        position2entity_name[(entity_start, entity_end)] == entity_name:
                    num_true_positive += 1
                    correct_entity_disambiguation_by_ner[entity_ner_label] += 1

                if (entity_start, entity_end) in position2entity_name and \
                        position2entity_name[(entity_start, entity_end)] != entity_name:
                    incorrect_entity_disambiguation_by_ner[entity_ner_label] += 1
                    if position2entity_name[(entity_start, entity_end)] == '':
                        num_pred_instance -= 1

            # compute gold_recall assuming all the GT in candidate entities can correctly selected
            if 'entity_candidates' in pred_entities:
                for entity_start, entity_end, entity_name, entity_ner_label, entity_candidate in zip(
                        pred_entities['starts'], pred_entities['ends'], pred_entities['entity_names'],
                        pred_entities.get('entity_ner_labels', 'NULL'), pred_entities['entity_candidates']
                ):
                    if (entity_start, entity_end) in position2entity_name and \
                            position2entity_name[(entity_start, entity_end)] in entity_candidate:
                        num_gold_true_positive += 1

    precision = dev_by_zero(num_true_positive, num_pred_instance)
    recall = dev_by_zero(num_true_positive, num_gt_instance)
    f1 = dev_by_zero(2 * precision * recall, precision + recall)

    gold_recall = dev_by_zero(num_gold_true_positive, num_gt_instance)
    print(f'num_true_positive: {num_true_positive}; num_pred_instance: {num_pred_instance}; num_gt_instance: {num_gt_instance}')
    print(f'incorrect_entity_disambiguation_by_ner: {incorrect_entity_disambiguation_by_ner}')
    print(f'correct_entity_disambiguation_by_ner: {correct_entity_disambiguation_by_ner}')
    print(f'precision: {precision}, recall: {recall}, f1: {f1}')
    print(f'gold_recall: {gold_recall}')

    out_dict = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'gold_recall': gold_recall,
        'num_true_positive': num_true_positive,
        'num_pred_instance': num_pred_instance,
        'num_gt_instance': num_gt_instance,
        'incorrect_entity_disambiguation_by_ner': incorrect_entity_disambiguation_by_ner,
        'correct_entity_disambiguation_by_ner': correct_entity_disambiguation_by_ner,
    }
    return out_dict


def main(args):
    # 1. obtain dataset
    # input_file = '/nfs/yding4/EL_project/dataset/KORE50/AIDA.tsv'
    # input_file = '/nfs/yding4/EL_project/dataset/AIDA-CONLL/AIDA-YAGO2-dataset.tsv'
    # input_file = '/nfs/yding4/EL_project/dataset/luke/msnbc.conll'
    # input_file = '/nfs/yding4/EL_project/dataset/luke/aquaint.conll'

    # there are two formats of dataset,
    # 1) .tsv: token \t ner-label('B' or 'I') \t mention \t entity \n or token \n
    # 2) .nt or .ttl
    # doc_name2instance = load_tsv(input_file)
    # doc_name2instance = load_ttl_oke_2015()
    # doc_name2instance = load_ttl_oke_2016()
    # doc_name2instance = load_ttl_n3('/nfs/yding4/EL_project/dataset/n3-collection/Reuters-128.ttl')

    if args.mode == 'tsv':
        # doc_name2instance = load_tsv(args.dataset_file, key=args.tsv_key, mode='token')
        doc_name2instance = load_tsv(args.dataset_file, key=args.tsv_key)
    elif args.mode == 'oke_2015':
        doc_name2instance = load_ttl_oke_2015(args.dataset_file)
    elif args.mode == 'oke_2016':
        doc_name2instance = load_ttl_oke_2016(args.dataset_file)
    elif args.mode == 'n3':
        doc_name2instance = load_ttl_n3(args.dataset_file)
    else:
        raise ValueError('unknown mode!')

    # 2. obtain the prediction
    doc_name2inference = dict()
    for doc_index, (doc_name, instance) in enumerate(tqdm(doc_name2instance.items())):
        sentence = instance['sentence'] if 'sentence' in instance else ' '.join(instance['tokens'])

        el_result = requests.post(API_URL, json={
            "text": sentence,
            # "spans": []
            "spans": []
        }).json()

        assert doc_name not in doc_name2inference
        doc_name2inference[doc_name] = el_result

    # 3. process the prediction
    process_doc_name2inference = process_pred(doc_name2inference, doc_name2instance)
    # process_doc_name2inference = process_pred2token(doc_name2inference, doc_name2instance)

    # 4. evaluate and analysis the results.
    out_dict = evaluate_and_analysis(process_doc_name2inference, doc_name2instance)

    with open(os.path.join(args.output_dir, 'evaluation.json'), 'w') as writer:
        writer.write(json.dumps(out_dict, indent=4))
    with open(os.path.join(args.output_dir, 'inference.json'), 'w') as writer:
        writer.write(json.dumps(process_doc_name2inference, indent=4))
    with open(os.path.join(args.output_dir, 'processed_gt_dataset.json'), 'w') as writer:
        writer.write(json.dumps(doc_name2instance, indent=4))


def parse_args():
    parser = argparse.ArgumentParser(
        description='evaluate and analysis the entity linking from tcp communication like REL and end2end_neural_el',
        allow_abbrev=False,
    )
    parser.add_argument(
        "--dataset_file",
        help="the dataset file ",
        default="/nfs/yding4/REL/data/dataset/KORE50/AIDA.tsv",
        # default="/nfs/yding4/REL/data/dataset/oke-challenge/evaluation-data/task1/evaluation-dataset-task1.ttl",
        type=str,
    )
    parser.add_argument(
        "--mode",
        help="mode/type of evaluation dataset file",
        default="tsv",
        choices=['tsv', 'oke_2015', 'oke_2016', 'n3'],
        type=str,
    )
    parser.add_argument(
        "--tsv_key",
        help="used for aida-testa and aida-testb; required showing up in the file for mode=tsv",
        default="",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="the output directory to store evaluation and analysis results of entity linking",
        default="result",
        type=str,
    )

    args = parser.parse_args()
    print(args)
    os.makedirs(args.output_dir, exist_ok=True)
    assert os.path.isfile(args.dataset_file)
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
