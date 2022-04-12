import os
import json
import argparse
from inference import (
    load_tsv,
    evaluate_and_analysis,
)

dataset_file = "/nfs/yding4/REL/data/dataset/KORE50/AIDA.tsv"
# prediction_file = "/nfs/yding4/REL/code/evaluation_and_analysis/entity_hyperlink_edit/add_verify_neighbor.json"
prediction_file = "/nfs/yding4/REL/code/evaluation_and_analysis/entity_hyperlink_edit/verify_edit_candidates.json"

doc_name2instance = load_tsv(dataset_file)

with open(prediction_file) as reader:
    doc_name2prediction = json.load(reader)

evaluate_and_analysis(doc_name2prediction, doc_name2instance)

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
                        edit_entity_name not in edit_entity_name2new_entity_name and
                        (
                            mapping_entity in verify_entity_names or
                            mapping_entity not in verify_entity_names or
                            (
                                mapping_entity in verify_entity_names and
                                verify_entity_names[mapping_entity] == mapping_entity
                            )
                        )
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


evaluate_and_analysis(verify_doc_name2prediction, doc_name2instance)

with open('evaluate_edit_verify.json', 'w') as writer:
    json.dump(verify_doc_name2prediction, writer, indent=4)
