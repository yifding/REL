import os
import json
import urllib
import pprint
import collections
import rdflib


def load_ttl_oke_2015(
    file='/nfs/yding4/EL_project/dataset/oke-challenge/evaluation-data/task1/evaluation-dataset-task1.ttl',
):
    def process_sen_char(s):
        assert 'sentence-' in s
        first_parts = s.split('sentence-')
        assert len(first_parts) == 2
        assert '#char=' in first_parts[1]
        second_parts = first_parts[1].split('#char=')
        assert len(second_parts) == 2
        assert ',' in second_parts[1]
        sentence_index = int(second_parts[0])
        third_parts = second_parts[1].split(',')
        assert len(third_parts) == 2
        char_start, char_end = int(third_parts[0]), int(third_parts[1])
        return char_start, char_end, sentence_index

    g = rdflib.Graph()
    g.parse(file, format='ttl')

    module_list = [
        'label', 'anchorOf', 'beginIndex', 'isString', 'sameAs', 'endIndex', 'taIdentRef', 'referenceContext', 'type',
    ]

    # 1. isString: extracts sentence (identified by the sentence number)
    # 2. taIdentRef: extracts mentions and labelled temporary annotations
    # 3. sameAs: bring temporary annotations to dataset-base if database has corresponding entities
    sentence_index2sentence = dict()
    sent_char_index2tmp_entity = dict()
    tmp_entity2entity = dict()

    for node_index, node in enumerate(g):
        parts = node[1].split('#')
        assert len(parts) == 2
        assert parts[1] in module_list

        if parts[1] == 'anchorOf':
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            tmp_str = str(node[2]).rstrip()
            if (char_end - char_start) != len(tmp_str):
                # only one data error: 'Basel, Switzerland'
                tmp_str = tmp_str.split(',')[0]
            assert (char_end - char_start) == len(tmp_str)

        elif parts[1] == 'taIdentRef':
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            assert str(node[2]).count('sentence-') == 1
            tmp_entity = str(node[2]).split('sentence-')[1]
            assert (sentence_index, char_start, char_end) not in sent_char_index2tmp_entity or \
                   sent_char_index2tmp_entity[(sentence_index, char_start, char_end)] in ['Man_4', 'His_4']
            sent_char_index2tmp_entity[(sentence_index, char_start, char_end)] = tmp_entity

        elif parts[1] == 'isString':
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            assert sentence_index not in sentence_index2sentence
            sentence_index2sentence[sentence_index] = str(node[2])

        elif parts[1] == 'sameAs':
            assert str(node[0]).count('sentence-') == 1
            mention = str(node[0]).split('sentence-')[1]

            entity = str(node[2]).split('/')[-1]
            if mention in tmp_entity2entity:
                assert entity == tmp_entity2entity[mention]
            tmp_entity2entity[mention] = entity
            # print(mention, entity)
            # if not str(node[2]).startswith('http://dbpedia.org/resource/'):
            #     print(node)
        else:
            if parts[1] == 'label':
                # 'label' is not useful
                tmp_split_str = str(node[0]).split('sentence-')[1]
                tmp_str = str(node[2])
                assert tmp_split_str == tmp_str.replace(' ', '_')

    num_in = 0
    num_out = 0

    sorted_key = sorted(sent_char_index2tmp_entity.keys(), key=lambda x: (x[0], x[1], x[2]))
    doc_name2instance = dict()
    for (tmp_sent_index, char_start, char_end) in sorted_key:
        sentence = sentence_index2sentence[tmp_sent_index]
        if str(tmp_sent_index) not in doc_name2instance:
            doc_name2instance[str(tmp_sent_index)] = {
                'sentence': sentence,
                'entities': {
                    'starts': [],
                    'ends': [],
                    'entity_mentions': [],
                    'entity_names': [],
                }
            }
        tmp_entity = sent_char_index2tmp_entity[(tmp_sent_index, char_start, char_end)]
        processed_tmp_entity = tmp_entity.replace(' ', '_')
        if processed_tmp_entity in tmp_entity2entity:
            num_in += 1
            entity = tmp_entity2entity[processed_tmp_entity]
            # assert (char_end - char_start) == len(tmp_str)
            mention = sentence[char_start: char_end]
            doc_name2instance[str(tmp_sent_index)]['entities']['starts'].append(char_start)
            doc_name2instance[str(tmp_sent_index)]['entities']['ends'].append(char_end)
            doc_name2instance[str(tmp_sent_index)]['entities']['entity_mentions'].append(mention)
            doc_name2instance[str(tmp_sent_index)]['entities']['entity_names'].append(entity)

        else:
            num_out += 1
    print(f'num_in_kb: {num_in}; num_out_kb: {num_out}; len(tmp_entity2entity): {len(tmp_entity2entity)}')
    # print(json.dumps(doc_name2instance, indent=4))
    return doc_name2instance


def load_ttl_oke_2016(
    file='/nfs/yding4/EL_project/dataset/oke-challenge-2016/evaluation-data/task1/evaluation-dataset-task1.ttl',
):
    def process_sen_char(s):
        assert 'sentence-' in s
        first_parts = s.split('sentence-')
        assert len(first_parts) == 2
        assert '#char=' in first_parts[1]
        second_parts = first_parts[1].split('#char=')
        assert len(second_parts) == 2
        assert ',' in second_parts[1]
        sentence_index = int(second_parts[0])
        third_parts = second_parts[1].split(',')
        assert len(third_parts) == 2
        char_start, char_end = int(third_parts[0]), int(third_parts[1])
        return char_start, char_end, sentence_index

    g = rdflib.Graph()
    g.parse(file, format='ttl')

    module_list = [
        'label', 'anchorOf', 'beginIndex', 'isString', 'sameAs', 'endIndex', 'taIdentRef', 'referenceContext', 'type',
    ]

    # 1. isString: extracts sentence (identified by the sentence number)
    # 2. taIdentRef: extracts mentions and labelled temporary annotations
    # 3. sameAs: bring temporary annotations to dataset-base if database has corresponding entities
    sentence_index2sentence = dict()
    sent_char_index2tmp_entity = dict()
    tmp_entity2entity = dict()

    for node_index, node in enumerate(g):
        parts = node[1].split('#')
        assert len(parts) == 2
        assert parts[1] in module_list

        if parts[1] == 'anchorOf':
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            tmp_str = str(node[2]).rstrip()
            if (char_end - char_start) != len(tmp_str):
                # only one data error: 'Basel, Switzerland'
                tmp_str = tmp_str.split(',')[0]
            assert (char_end - char_start) == len(tmp_str)

        elif parts[1] == 'taIdentRef':
            # print(node)
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            assert str(node[2]).count('task-1/') == 1
            tmp_entity = str(node[2]).split('task-1/')[1]
            assert (sentence_index, char_start, char_end) not in sent_char_index2tmp_entity or \
                   sent_char_index2tmp_entity[(sentence_index, char_start, char_end)] in ['Man_4', 'His_4']
            sent_char_index2tmp_entity[(sentence_index, char_start, char_end)] = tmp_entity

        elif parts[1] == 'isString':
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            assert sentence_index not in sentence_index2sentence
            sentence_index2sentence[sentence_index] = str(node[2])

        elif parts[1] == 'sameAs':
            # print(node)
            assert str(node[0]).count('task-1/') == 1
            mention = str(node[0]).split('task-1/')[1]

            entity = str(node[2]).split('/')[-1]
            if mention in tmp_entity2entity:
                assert entity == tmp_entity2entity[mention]
            tmp_entity2entity[mention] = entity
            # print(mention, entity)
            # if not str(node[2]).startswith('http://dbpedia.org/resource/'):
            #     print(node)
        else:
            assert parts[1] in ['beginIndex', 'label', 'endIndex', 'referenceContext', 'type']

    num_in = 0
    num_out = 0

    sorted_key = sorted(sent_char_index2tmp_entity.keys(), key=lambda x: (x[0], x[1], x[2]))
    doc_name2instance = dict()
    for (tmp_sent_index, char_start, char_end) in sorted_key:
        sentence = sentence_index2sentence[tmp_sent_index]
        if str(tmp_sent_index) not in doc_name2instance:
            doc_name2instance[str(tmp_sent_index)] = {
                'sentence': sentence,
                'entities': {
                    'starts': [],
                    'ends': [],
                    'entity_mentions': [],
                    'entity_names': [],
                }
            }
        tmp_entity = sent_char_index2tmp_entity[(tmp_sent_index, char_start, char_end)]
        processed_tmp_entity = tmp_entity.replace(' ', '_')
        if processed_tmp_entity in tmp_entity2entity:
            num_in += 1
            entity = tmp_entity2entity[processed_tmp_entity]
            # assert (char_end - char_start) == len(tmp_str)
            mention = sentence[char_start: char_end]
            doc_name2instance[str(tmp_sent_index)]['entities']['starts'].append(char_start)
            doc_name2instance[str(tmp_sent_index)]['entities']['ends'].append(char_end)
            doc_name2instance[str(tmp_sent_index)]['entities']['entity_mentions'].append(mention)
            doc_name2instance[str(tmp_sent_index)]['entities']['entity_names'].append(entity)

        else:
            num_out += 1
    print(f'num_in_kb: {num_in}; num_out_kb: {num_out}; len(tmp_entity2entity): {len(tmp_entity2entity)}')
    return doc_name2instance


def load_ttl_n3(
    file='/nfs/yding4/EL_project/dataset/n3-collection/Reuters-128.ttl',
):
    def process_sen_char(s):
        assert s.count('/') == 5
        first_parts = s.split('/')
        assert '#char=' in first_parts[-1]
        second_parts = first_parts[-1].split('#char=')
        assert len(second_parts) == 2
        assert ',' in second_parts[1]
        sentence_index = int(second_parts[0])
        third_parts = second_parts[1].split(',')
        assert len(third_parts) == 2
        char_start, char_end = int(third_parts[0]), int(third_parts[1])
        return char_start, char_end, sentence_index

    # file = '/nfs/yding4/EL_project/dataset/oke-challenge-2016/evaluation-data/task1/evaluation-dataset-task1.ttl'
    g = rdflib.Graph()
    g.parse(file, format='ttl')

    module_list = [
        'label', 'anchorOf', 'beginIndex', 'isString', 'sameAs', 'endIndex', 'taIdentRef', 'referenceContext', 'type', 'taSource', 'hasContext', 'sourceUrl'
    ]

    # 1. isString: extracts sentence (identified by the sentence number)
    # 2. taIdentRef: extracts mentions and labelled temporary annotations
    # 3. sameAs: bring temporary annotations to dataset-base if database has corresponding entities
    sentence_index2sentence = dict()
    sent_char_index2tmp_entity = dict()
    tmp_entity2entity = dict()
    num_in = 0
    num_out = 0

    for node_index, node in enumerate(g):
        # print(node)

        parts = node[1].split('#')
        assert len(parts) == 2
        if parts[1] not in module_list:
            print(str(parts[1]))
        assert parts[1] in module_list

        if parts[1] == 'anchorOf':
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            # tmp_str = str(node[2]).rstrip()
            tmp_str = str(node[2])
            if (char_end - char_start) != len(tmp_str):
                # only one data error: 'Basel, Switzerland'
                tmp_str = tmp_str.split(',')[0]
            assert (char_end - char_start) == len(tmp_str)

        elif parts[1] == 'taIdentRef':
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            assert '/' in str(node[2])
            tmp_entity = str(node[2]).split('/')[-1]
            assert (sentence_index, char_start, char_end) not in sent_char_index2tmp_entity
            if 'notInWiki' in str(node[2]):
                num_out += 1
            else:
                num_in += 1
                assert str(node[2].startswith('http://dbpedia.org/resource/'))

                sent_char_index2tmp_entity[(sentence_index, char_start, char_end)] = urllib.parse.unquote(tmp_entity)

        elif parts[1] == 'isString':
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            assert sentence_index not in sentence_index2sentence
            sentence_index2sentence[sentence_index] = str(node[2])

        elif parts[1] == 'sameAs':
            assert str(node[0]).count('sentence-') == 1
            mention = str(node[0]).split('sentence-')[1]

            entity = str(node[2]).split('/')[-1]
            if mention in tmp_entity2entity:
                assert entity == tmp_entity2entity[mention]
            tmp_entity2entity[mention] = entity
        else:
            if parts[1] == 'label':
                # 'label' is not useful
                tmp_split_str = str(node[0]).split('sentence-')[1]
                tmp_str = str(node[2])
                assert tmp_split_str == tmp_str.replace(' ', '_')

    sorted_key = sorted(sent_char_index2tmp_entity.keys(), key=lambda x: (x[0], x[1], x[2]))
    doc_name2instance = dict()
    for (tmp_sent_index, char_start, char_end) in sorted_key:
        sentence = sentence_index2sentence[tmp_sent_index]
        if str(tmp_sent_index) not in doc_name2instance:
            doc_name2instance[str(tmp_sent_index)] = {
                'sentence': sentence,
                'entities': {
                    'starts': [],
                    'ends': [],
                    'entity_mentions': [],
                    'entity_names': [],
                }
            }
        tmp_entity = sent_char_index2tmp_entity[(tmp_sent_index, char_start, char_end)]
        processed_tmp_entity = tmp_entity.replace(' ', '_')

        # assert (char_end - char_start) == len(tmp_str)
        mention = sentence[char_start: char_end]
        doc_name2instance[str(tmp_sent_index)]['entities']['starts'].append(char_start)
        doc_name2instance[str(tmp_sent_index)]['entities']['ends'].append(char_end)
        doc_name2instance[str(tmp_sent_index)]['entities']['entity_mentions'].append(mention)
        doc_name2instance[str(tmp_sent_index)]['entities']['entity_names'].append(processed_tmp_entity)

    # print(json.dumps(doc_name2instance, indent=4))
    print(f'num_in_kb: {num_in}; num_out_kb: {num_out};')

    return doc_name2instance


if __name__ == '__main__':
    load_ttl_oke_2015()
    load_ttl_oke_2016()
    load_ttl_n3('/nfs/yding4/EL_project/dataset/n3-collection/Reuters-128.ttl')
    load_ttl_n3('/nfs/yding4/EL_project/dataset/n3-collection/RSS-500.ttl')
