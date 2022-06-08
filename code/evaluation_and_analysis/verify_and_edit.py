import os
import json
import copy
import argparse
import collections
from tqdm import tqdm
from REL.wikipedia import Wikipedia
from inference import evaluate_and_analysis


def verify_and_edit(
    doc_name2instance,
    wikiid2hyperlink_wikid,
    wikipedia,
    edit_sliding_window=100,
):
    # 1. add bi-directional and one-directional verification.
    new_doc_name2instance = dict()
    for doc_name, instance in doc_name2instance.items():
        sentence = instance['sentence']
        entities = instance['entities']
        new_entities = copy.deepcopy(entities)
        entity_neighbors = []
        for (
            start,
            end,
            entity_ner_label,
            entity_mention,
            entity_name,
            entity_candidate_list,
        ) in zip(
            entities['starts'],
            entities['ends'],
            entities['entity_ner_labels'],
            entities['entity_mentions'],
            entities['entity_names'],
            entities['entity_candidates'],
        ):
            entity_neighbors.append([])
            for (
                second_start,
                second_end,
                second_entity_ner_label,
                second_entity_mention,
                second_entity_name,
                second_entity_candidate_list
            ) in zip(
                entities['starts'],
                entities['ends'],
                entities['entity_ner_labels'],
                entities['entity_mentions'],
                entities['entity_names'],
                entities['entity_candidates']
            ):

                if (
                    # entity_ner_label in ['PER', 'ORG'] and
                    # second_entity_ner_label in ['PER', 'ORG'] and
                    entity_name != second_entity_name
                ):
                    source_entity_wikiid = str(wikipedia.ent_wiki_id_from_name(entity_name))
                    target_entity_wikiid = str(wikipedia.ent_wiki_id_from_name(second_entity_name))

                    # if source_entity_wikiid == 8618 and target_entity_wikiid == 45979:
                    #     print(entity_name, second_entity_name)

                    if (
                        source_entity_wikiid in wikiid2hyperlink_wikid and
                        target_entity_wikiid in wikiid2hyperlink_wikid[source_entity_wikiid] and
                        second_entity_name not in entity_neighbors[-1]
                    ):
                        entity_neighbors[-1].append(second_entity_name)

        new_entities['entity_neighbors'] = entity_neighbors
        new_doc_name2instance[doc_name] = {
            'doc_name': doc_name,
            'sentence': sentence,
            'entities': new_entities,
        }

    # # store the verified doc_name2instance
    # with open('all_add_verify_neighbor.json', 'w') as writer:
    #     json.dump(new_doc_name2instance, writer, indent=4)


    doc_name2instance = copy.deepcopy(new_doc_name2instance)

    # 2. add verified entity predictions

    for doc_name, instance in doc_name2instance.items():
        entities = instance['entities']
        num_local_org = 0
        num_local_per = 0
        ent_names_set = set()

        # 2-1: consider all the neighbor entities
        for (
            start,
            end,
            entity_ner_label,
            entity_mention,
            entity_name,
            entity_candidate,
            entity_neighbor,
        ) in zip(
            entities['starts'],
            entities['ends'],
            entities['entity_ner_labels'],
            entities['entity_mentions'],
            entities['entity_names'],
            entities['entity_candidates'],
            entities['entity_neighbors'],
        ):
            if entity_ner_label == 'ORG':
                num_local_org += 1
                for neighbor in entity_neighbor:
                    ent_names_set.add(neighbor)
            elif entity_ner_label == 'PER':
                num_local_per += 1
                for neighbor in entity_neighbor:
                    ent_names_set.add(neighbor)
            else:
                for neighbor in entity_neighbor:
                    ent_names_set.add(neighbor)

        # 2-2: obtain all the verified entities
        verify_entity_names = set()
        for (
            start,
            end,
            entity_ner_label,
            entity_mention,
            entity_name,
            entity_candidate,
            entity_neighbor,
        ) in zip(
            entities['starts'],
            entities['ends'],
            entities['entity_ner_labels'],
            entities['entity_mentions'],
            entities['entity_names'],
            entities['entity_candidates'],
            entities['entity_neighbors'],
        ):
            if (
                entity_ner_label == 'ORG' and
                (
                    num_local_per == 0 or
                    entity_name in ent_names_set or
                    len(entity_neighbor) > 0 or
                    len(entity_candidate) == 1
                )
            ):
                verify_entity_names.add(entity_name)

            elif (
                entity_ner_label == 'PER' and
                (
                    (num_local_per == 1 and num_local_org == 0) or
                    entity_name in ent_names_set or
                    len(entity_neighbor) > 0 or
                    len(entity_candidate) == 1
                )
            ):
                verify_entity_names.add(entity_name)

            elif (
                (entity_ner_label == 'LOC' or entity_ner_label == 'MISC') and
                (
                    num_local_per == 0 or
                    entity_name in ent_names_set or
                    len(entity_neighbor) > 0 or
                    len(entity_candidate) == 1
                )
            ):
                verify_entity_names.add(entity_name)

        new_doc_name2instance[doc_name]['verify_entity_names'] = sorted(verify_entity_names)

        # 2-3: obtain the edit candidates
        # 2-3-1: consider modifying single unverified entity
        single_edit_entity_candidates = dict()

        for (
            start,
            end,
            entity_ner_label,
            entity_mention,
            entity_name,
            entity_candidate_list,
        ) in zip(
            entities['starts'],
            entities['ends'],
            entities['entity_ner_labels'],
            entities['entity_mentions'],
            entities['entity_names'],
            entities['entity_candidates'],
        ):
            if entity_name in verify_entity_names or entity_name in single_edit_entity_candidates:
                continue

            for (
                second_start,
                second_end,
                second_entity_ner_label,
                second_entity_mention,
                second_entity_name,
                second_entity_candidate_list,
            ) in zip(
                entities['starts'],
                entities['ends'],
                entities['entity_ner_labels'],
                entities['entity_mentions'],
                entities['entity_names'],
                entities['entity_candidates'],
            ):
                if entity_name == second_entity_name or (
                   min(abs(second_start - end), abs(second_end - start)) >= edit_sliding_window
                ):
                    continue

                for entity_candidate in entity_candidate_list:
                    if entity_candidate == entity_name:
                        continue
                    edit_entity_candidate = [second_entity_name, 0, 0, 0]

                    source_entity_wikiid = str(wikipedia.ent_wiki_id_from_name(entity_candidate))
                    target_entity_wikiid = str(wikipedia.ent_wiki_id_from_name(second_entity_name))

                    if (
                        source_entity_wikiid in wikiid2hyperlink_wikid and
                        target_entity_wikiid in wikiid2hyperlink_wikid[source_entity_wikiid]
                    ):
                        edit_entity_candidate[1] = 1
                    if (
                        target_entity_wikiid in wikiid2hyperlink_wikid and
                        source_entity_wikiid in wikiid2hyperlink_wikid[target_entity_wikiid]
                    ):
                        edit_entity_candidate[2] = 1
                    if second_entity_name in verify_entity_names:
                        edit_entity_candidate[3] = 1

                    if edit_entity_candidate[1] + edit_entity_candidate[2] == 0:
                        continue
                    if entity_name not in single_edit_entity_candidates:
                        single_edit_entity_candidates[entity_name] = collections.defaultdict(list)
                    single_edit_entity_candidates[entity_name][entity_candidate].append(edit_entity_candidate)

        new_doc_name2instance[doc_name]['single_edit_entity_candidates'] = single_edit_entity_candidates

        # 2-3-2: consider modifying double unverified entity
        double_edit_entity_candidates = dict()
        for (
            start,
            end,
            entity_ner_label,
            entity_mention,
            entity_name,
            entity_candidate_list,
        ) in zip(
            entities['starts'],
            entities['ends'],
            entities['entity_ner_labels'],
            entities['entity_mentions'],
            entities['entity_names'],
            entities['entity_candidates'],
        ):
            for (
                second_start,
                second_end,
                second_entity_ner_label,
                second_entity_mention,
                second_entity_name,
                second_entity_candidate_list,
            ) in zip(
                entities['starts'],
                entities['ends'],
                entities['entity_ner_labels'],
                entities['entity_mentions'],
                entities['entity_names'],
                entities['entity_candidates'],
            ):
                if (
                    entity_name == second_entity_name or
                    entity_name in verify_entity_names or
                    second_entity_name in verify_entity_names or
                    entity_name in double_edit_entity_candidates
                ):
                    continue

                for entity_candidate in entity_candidate_list:
                    for second_entity_candidate in second_entity_candidate_list:
                        if (

                            entity_candidate == entity_name or
                            second_entity_candidate == second_entity_name or
                            entity_candidate == second_entity_candidate # rare case
                        ):
                            continue

                        edit_entity_candidate = [entity_candidate, second_entity_candidate, 0, 0]
                        source_entity_wikiid = str(wikipedia.ent_wiki_id_from_name(entity_candidate))
                        target_entity_wikiid = str(wikipedia.ent_wiki_id_from_name(second_entity_candidate))

                        if (
                            source_entity_wikiid in wikiid2hyperlink_wikid and
                            target_entity_wikiid in wikiid2hyperlink_wikid[source_entity_wikiid]
                        ):
                            edit_entity_candidate[2] = 1
                        if (
                            target_entity_wikiid in wikiid2hyperlink_wikid and
                            source_entity_wikiid in wikiid2hyperlink_wikid[target_entity_wikiid]
                        ):
                            edit_entity_candidate[3] = 1

                        if edit_entity_candidate[2] == 0 or edit_entity_candidate[3] == 0:
                            continue

                        if entity_name not in double_edit_entity_candidates:
                            double_edit_entity_candidates[entity_name] = collections.defaultdict(list)

                        double_edit_entity_candidates[entity_name][second_entity_name].append(edit_entity_candidate)

        new_doc_name2instance[doc_name]['double_edit_entity_candidates'] = double_edit_entity_candidates

    return new_doc_name2instance


def process_verify(doc_name2prediction):
    # only select verified instances
    verify_doc_name2prediction = dict()

    for doc_name, prediction in doc_name2prediction.items():
        entities = prediction['entities']

        verify_entity_names = prediction.get('verify_entity_names', [])
        edit_entity_name2new_entity_name = dict()
        for verify_entity_name in verify_entity_names:
            edit_entity_name2new_entity_name[verify_entity_name] = verify_entity_name

        single_edit_entity_candidates = prediction.get('single_edit_entity_candidates', [])
        for edit_entity_name, new_entity_name2list in single_edit_entity_candidates.items():
            for new_entity_name, mapping_list_list in new_entity_name2list.items():
                # inner link, outer link, verified relative entity must have 2 of 3 satisfied.
                for mapping_list in mapping_list_list:
                    mapping_entity = mapping_list[0]
                    if sum(mapping_list[1:]) >= 2:
                        assert edit_entity_name not in verify_entity_names
                        if (
                            edit_entity_name not in edit_entity_name2new_entity_name
                            # and
                            # (
                            #     # mapping_entity in verify_entity_names or
                            #     mapping_entity not in verify_entity_names or
                            #     (
                            #         mapping_entity in verify_entity_names and
                            #         verify_entity_names[mapping_entity] == mapping_entity
                            #     )
                            # )
                        ):
                            edit_entity_name2new_entity_name[edit_entity_name] = new_entity_name
                        if mapping_entity not in verify_entity_names and mapping_entity not in edit_entity_name2new_entity_name:
                            edit_entity_name2new_entity_name[mapping_entity] = mapping_entity

        # 1. consider
        new_starts = []
        new_ends = []
        new_entity_ner_labels = []
        new_entity_mentions = []
        new_entity_names = []
        new_entity_candidates = []
        new_entity_neighbors = []

        for (
            start,
            end,
            entity_ner_label,
            entity_mention,
            entity_name,
            entity_candidate,
            entity_neighbor,
        ) in zip(
            entities['starts'],
            entities['ends'],
            entities['entity_ner_labels'],
            entities['entity_mentions'],
            entities['entity_names'],
            entities['entity_candidates'],
            entities['entity_neighbors'],
        ):
            if entity_name in edit_entity_name2new_entity_name:
                entity_name = edit_entity_name2new_entity_name[entity_name]

                new_starts.append(start)
                new_ends.append(end)
                new_entity_ner_labels.append(entity_ner_label)
                new_entity_mentions.append(entity_mention)
                new_entity_names.append(entity_name)
                new_entity_candidates.append(entity_candidate)
                new_entity_neighbors.append(entity_neighbor)

            # padding 1: mention and entity_name the same
            elif entity_name.lower().replace(' ', '_') == entity_mention.lower().replace(' ', '_'):
                new_starts.append(start)
                new_ends.append(end)
                new_entity_ner_labels.append(entity_ner_label)
                new_entity_mentions.append(entity_mention)
                new_entity_names.append(entity_name)
                new_entity_candidates.append(entity_candidate)
                new_entity_neighbors.append(entity_neighbor)

            elif wikipedia.preprocess_ent_name(entity_name) == wikipedia.preprocess_ent_name(entity_mention):
                new_starts.append(start)
                new_ends.append(end)
                new_entity_ner_labels.append(entity_ner_label)
                new_entity_mentions.append(entity_mention)
                new_entity_names.append(entity_name)
                new_entity_candidates.append(entity_candidate)
                new_entity_neighbors.append(entity_neighbor)

        new_entities = {
            'starts': new_starts,
            'ends': new_ends,
            'entity_ner_labels': new_entity_ner_labels,
            'entity_mentions': new_entity_mentions,
            'entity_names': new_entity_names,
            'entity_candidates': new_entity_candidates,
            'entity_neighbors': new_entity_neighbors,
        }
        verify_doc_name2prediction[doc_name] = {
            'doc_name': doc_name,
            'entities': new_entities,
            'sentence': prediction['sentence'],
        }

    return verify_doc_name2prediction


def parse_args():
    parser = argparse.ArgumentParser(
        description='process the revised REL inference results with verification and edition.',
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input_dir",
        help="the REL inference output directory",
        # default="/nfs/yding4/REL/code/evaluation_and_analysis/RUN_FILES/REL_server_inference/KORE50",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--wikiid2hyperlink_wikid_file",
        help="the dataset file ",
        # default="/nfs/yding4/REL/code/evaluation_and_analysis/wikiid2hyperlink_wikid.json",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--base_url",
        # default="/nfs/yding4/REL/data/",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--wiki_version",
        # default="wiki_2014",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="the output directory to store the processed inference",
        # default='result',
        required=True,
        type=str,
    )

    args = parser.parse_args()
    print(args)
    os.makedirs(args.output_dir, exist_ok=True)
    return args


if __name__ == '__main__':
    args = parse_args()
    # 0-1: prepare wikipedia
    wikipedia = Wikipedia(args.base_url, args.wiki_version)

    # 0-2: load inference file from KORE50
    instance_file = os.path.join(args.input_dir, 'processed_gt_dataset.json')
    prediction_file = os.path.join(args.input_dir, 'inference.json')
    with open(instance_file) as reader:
        doc_name2instance = json.load(reader)
    with open(prediction_file) as reader:
        doc_name2prediction = json.load(reader)

    # 0-3: load hyperlink file
    with open(args.wikiid2hyperlink_wikid_file) as reader:
        wikiid2hyperlink_wikid = json.load(reader)

    verified_doc_name2prediction = verify_and_edit(
        doc_name2prediction,
        wikiid2hyperlink_wikid,
        wikipedia,
    )

    processed_doc_name2prediction = process_verify(verified_doc_name2prediction)

    init_result = evaluate_and_analysis(doc_name2prediction, doc_name2instance)
    verify_edit_result = evaluate_and_analysis(processed_doc_name2prediction, doc_name2instance)

    init_inference_file = os.path.join(args.output_dir, 'init_inference.json')
    with open(init_inference_file, 'w') as writer:
        json.dump(doc_name2prediction, writer, indent=4)

    init_evaluation_file = os.path.join(args.output_dir, 'init_evaluation.json')
    with open(init_evaluation_file, 'w') as writer:
        json.dump(init_result, writer, indent=4)

    verify_edit_inference_file = os.path.join(args.output_dir, 'verify_edit_inference.json')
    with open(verify_edit_inference_file, 'w') as writer:
        json.dump(processed_doc_name2prediction, writer, indent=4)

    verify_edit_evaluation_file = os.path.join(args.output_dir, 'verify_edit_evaluation.json')
    with open(verify_edit_evaluation_file, 'w') as writer:
        json.dump(verify_edit_result, writer, indent=4)

    gt_file = os.path.join(args.output_dir, 'gt.json')
    with open(gt_file, 'w') as writer:
        json.dump(doc_name2instance, writer, indent=4)


