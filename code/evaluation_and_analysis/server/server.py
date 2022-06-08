import os
import argparse
import json
import jsonlines
from http.server import BaseHTTPRequestHandler

from flair.models import SequenceTagger

from REL.mention_detection import MentionDetection
from REL.utils import process_results
from REL.wikipedia import Wikipedia

from verify_and_edit import (
    verify_and_edit,
    process_verify,
)

API_DOC = "API_DOC"

"""
Class/function combination that is used to setup an API that can be used for e.g. GERBIL evaluation.
"""


def make_handler(
        base_url,
        wiki_version,
        model,
        tagger_ner,
        sentence2ner=None,
        mode='default',
        wikiid2hyperlink_wikid=None,
        wikipedia=None,
):
    class GetHandler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.model = model
            self.tagger_ner = tagger_ner

            self.base_url = base_url
            self.wiki_version = wiki_version

            self.custom_ner = not isinstance(tagger_ner, SequenceTagger)
            self.mention_detection = MentionDetection(base_url, wiki_version)
            self.sentence2ner = sentence2ner
            self.mode = mode

            self.wikipedia = wikipedia
            # self.wikipedia = Wikipedia(base_url, wiki_version)
            self.wikiid2hyperlink_wikid = wikiid2hyperlink_wikid

            # **YD** stats the number of match_ner and number of unmatched_ner and total entities

            # self.num_total_entities = 0
            # self.num_ner_match_entities = 0
            # self.num_ner_unmatch_entities = 0

            super().__init__(*args, **kwargs)

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                bytes(
                    json.dumps(
                        {
                            "schemaVersion": 1,
                            "label": "status",
                            "message": "up",
                            "color": "green",
                        }
                    ),
                    "utf-8",
                )
            )
            return

        def do_HEAD(self):
            # send bad request response code
            self.send_response(400)
            self.end_headers()
            self.wfile.write(bytes(json.dumps([]), "utf-8"))
            return

        def do_POST(self):
            """
            Returns response.
            :return:
            """
            try:
                content_length = int(self.headers["Content-Length"])
                post_data = self.rfile.read(content_length)
                self.send_response(200)
                self.end_headers()

                text, spans = self.read_json(post_data)
                response = self.generate_response(text, spans)

                self.wfile.write(bytes(json.dumps(response), "utf-8"))
            except Exception as e:
                print(f"Encountered exception: {repr(e)}")
                self.send_response(400)
                self.end_headers()
                self.wfile.write(bytes(json.dumps([]), "utf-8"))
            return

        def read_json(self, post_data):
            """
            Reads input JSON message.
            :return: document text and spans.
            """

            data = json.loads(post_data.decode("utf-8"))
            text = data["text"]
            text = text.replace("&amp;", "&")

            # GERBIL sends dictionary, users send list of lists.
            if "spans" in data:
                try:
                    spans = [list(d.values()) for d in data["spans"]]
                except Exception:
                    spans = data["spans"]
                    pass
            else:
                spans = []

            return text, spans

        def generate_response(self, text, spans):
            """
            Generates response for API. Can be either ED only or EL, meaning end-to-end.
            :return: list of tuples for each entity found.
            """

            if len(text) == 0:
                return []

            if len(spans) > 0:
                # ED.
                processed = {API_DOC: [text, spans]}
                mentions_dataset, total_ment = self.mention_detection.format_spans(
                    processed
                )
            else:
                # EL
                processed = {API_DOC: [text, spans]}
                mentions_dataset, total_ment = self.mention_detection.find_mentions(
                    processed, self.tagger_ner
                )

            # Disambiguation
            predictions, timing = self.model.predict(mentions_dataset)

            # Process result.

            # if self.sentence2ner is None:
            #     result = process_results(
            #         mentions_dataset,
            #         predictions,
            #         processed,
            #         include_offset=False if ((len(spans) > 0) or self.custom_ner) else True,
            #     )
            # else:
            result = yd_process_results(
                mentions_dataset,
                predictions,
                processed,
                self.sentence2ner,
                include_offset=False if ((len(spans) > 0) or self.custom_ner) else True,
                mode=self.mode,
            )

            result = transform_tcp_to_doc_name2prediction(result)

            result = verify_and_edit(
                result,
                self.wikiid2hyperlink_wikid,
                self.wikipedia,
            )

            result = process_verify(result, self.wikipedia)

            result = transform_doc_name2prediction_to_tcp(result)


            # Singular document.
            if len(result) > 0:
                tmp_result = [*result.values()][0]
            else:
                tmp_result = []

            return tmp_result

    return GetHandler


def yd_process_results(
    mentions_dataset,
    predictions,
    processed,
    sentence2ner,
    include_offset=False,
    mode='default',
    rank_pred_score=True,
):
    """
    Function that can be used to process the End-to-End results.
    :return: dictionary with results and document as key.
    """
    assert mode in ['best_candidate', 'remove_invalid', 'default']

    res = {}
    for doc in mentions_dataset:
        if doc not in predictions:
            # No mentions found, we return empty list.
            continue
        pred_doc = predictions[doc]
        ment_doc = mentions_dataset[doc]
        text = processed[doc][0]
        res_doc = []

        for pred, ment in zip(pred_doc, ment_doc):
            sent = ment["sentence"]
            idx = ment["sent_idx"]
            start_pos = ment["pos"]
            mention_length = int(ment["end_pos"] - ment["pos"])

            if pred["prediction"] != "NIL":
                candidates = [
                    {
                        'cand_rank': cand_rank,
                        'cand_name': cand_name,
                        'cand_score': cand_score,
                    }
                    for cand_rank, (cand_name, cand_mask, cand_score)
                    in enumerate(zip(pred['candidates'], pred['masks'], pred['scores']))
                    if float(cand_mask) == 1
                ]

                if rank_pred_score:
                    candidates = sorted(candidates, key=lambda x: float(x['cand_score']), reverse=True)

                # make sure that ed_model predict is always in the first place.
                for cand_index, candidate in enumerate(candidates):
                    if candidate['cand_name'] == pred['prediction']:
                        if cand_index != 0:
                            candidates[0], candidates[cand_index] = candidates[cand_index], candidates[0]
                        break

                if len(candidates) == 1:
                    temp = (
                        start_pos,
                        mention_length,
                        pred["prediction"],
                        ment["ngram"],
                        pred["conf_ed"],
                        ment["conf_md"] if "conf_md" in ment else 0.0,
                        ment["tag"] if "tag" in ment else "NULL",
                        [tmp_candidate['cand_name'] for tmp_candidate in candidates],
                    )
                    res_doc.append(temp)
                else:

                    if mode == 'best_candidate':
                        for cand_index, candidate in enumerate(candidates):
                            tmp_cand_name = candidate['cand_name'].replace('_', ' ')
                            if sentence2ner is not None and \
                                tmp_cand_name in sentence2ner and \
                                ment["tag"] != sentence2ner[tmp_cand_name]:
                                continue
                            else:
                                temp = (
                                    start_pos,
                                    mention_length,
                                    candidate['cand_name'],
                                    ment["ngram"],
                                    pred["conf_ed"],
                                    ment["conf_md"] if "conf_md" in ment else 0.0,
                                    ment["tag"] if "tag" in ment else "NULL",
                                    [tmp_candidate['cand_name'] for tmp_candidate in candidates],
                                )
                                res_doc.append(temp)
                                break
                    elif mode == 'remove_invalid':
                        tmp_cand_name = pred["prediction"].replace('_', '')
                        if sentence2ner is not None and \
                            tmp_cand_name in sentence2ner and \
                            ment["tag"] != sentence2ner[tmp_cand_name]:
                            pass
                        else:
                            temp = (
                                start_pos,
                                mention_length,
                                pred["prediction"],
                                ment["ngram"],
                                pred["conf_ed"],
                                ment["conf_md"] if "conf_md" in ment else 0.0,
                                ment["tag"] if "tag" in ment else "NULL",
                                [tmp_candidate['cand_name'] for tmp_candidate in candidates],
                            )
                            res_doc.append(temp)
                    elif mode == 'default':
                        temp = (
                            start_pos,
                            mention_length,
                            pred["prediction"],
                            ment["ngram"],
                            pred["conf_ed"],
                            ment["conf_md"] if "conf_md" in ment else 0.0,
                            ment["tag"] if "tag" in ment else "NULL",
                            [tmp_candidate['cand_name'] for tmp_candidate in candidates],
                        )
                        res_doc.append(temp)

        res[doc] = res_doc
    return res


def transform_tcp_to_doc_name2prediction(
    doc_name2instance,
):
    """
    :param doc_name2instance:
    doc_name: [(
        start_pos,
        mention_length,
        pred["prediction"],
        ment["ngram"],
        pred["conf_ed"],
        ment["conf_md"] if "conf_md" in ment else 0.0,
        ment["tag"] if "tag" in ment else "NULL",
        [tmp_candidate['cand_name'] for tmp_candidate in candidates],
    )]
    :param wikiid2hyperlink_wikid:
    :param wikipedia:
    :return:
    """
    doc_name2prediction = dict()
    for doc_name, instances in doc_name2instance.items():
        new_starts = []
        new_ends = []
        new_entity_ner_labels = []
        new_entity_mentions = []
        new_entity_names = []
        new_entity_candidates = []

        for instance in instances:
            start = instance[0]
            end = instance[0] + instance[1]
            entity_name = instance[2]
            entity_mention = instance[3]
            entity_ner_label = instance[6]
            entity_candidate = instance[7]

            new_starts.append(start)
            new_ends.append(end)
            new_entity_ner_labels.append(entity_ner_label)
            new_entity_mentions.append(entity_mention)
            new_entity_names.append(entity_name)
            new_entity_candidates.append(entity_candidate)

        new_entities = {
            'starts': new_starts,
            'ends': new_ends,
            'entity_ner_labels': new_entity_ner_labels,
            'entity_mentions': new_entity_mentions,
            'entity_names': new_entity_names,
            'entity_candidates': new_entity_candidates,
        }

        doc_name2prediction[doc_name] = {
            'doc_name': doc_name,
            'sentence': '',
            'entities': new_entities,
        }

    return doc_name2prediction


def transform_doc_name2prediction_to_tcp(
    doc_name2instance,
):
    doc_name2tcp = dict()
    for doc_name, instance in doc_name2instance.items():
        doc_name2tcp[doc_name] = []
        entities = instance['entities']
        for (
            start,
            end,
            entity_ner_label,
            entity_mention,
            entity_name,
            entity_candidate,
        ) in zip(
            entities['starts'],
            entities['ends'],
            entities['entity_ner_labels'],
            entities['entity_mentions'],
            entities['entity_names'],
            entities['entity_candidates'],
        ):

            tcp_instance = (
                start,
                end-start,
                entity_name,
                entity_mention,
                0.0,
                0.0,
                entity_ner_label,
                entity_candidate,
            )
            doc_name2tcp[doc_name].append(tcp_instance)
    return doc_name2tcp


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from http.server import HTTPServer

    from REL.entity_disambiguation import EntityDisambiguation
    from REL.ner import load_flair_ner

    p = argparse.ArgumentParser()
    p.add_argument("--base_url", default="/nfs/yding4/REL/data/")
    p.add_argument("--wiki_version", default="wiki_2014")
    # p.add_argument("--sentence2ner-file", default="/nfs/yding4/REL/data/sentence2ner/sample_sentence2ner.jsonl")
    p.add_argument("--sentence2ner-file", default="")
    p.add_argument("--mode", default="default", choices=['best_candidate', 'remove_invalid', 'default'])
    p.add_argument("--ed-model", default="ed-wiki-2014")
    p.add_argument("--ner-model", default="ner-fast")
    # 'https://huggingface.co/flair/ner-english-fast'
    p.add_argument("--bind", "-b", metavar="ADDRESS", default="localhost")
    p.add_argument("--port", "-p", default=5555, type=int)
    args = p.parse_args()

    sentence2ner = None
    if os.path.isfile(args.sentence2ner_file):
        sentence2ner = dict()
        print('find sentence2ner_file! starts to load!')
        with jsonlines.open(args.sentence2ner_file) as reader:
            for line in reader:
                tmp_wikiname = line['wikiname']
                tmp_ner_label = line['ner_label']
                if tmp_wikiname != 'YD_None':
                    sentence2ner[tmp_wikiname] = tmp_ner_label


    with open('/nfs/yding4/REL/code/evaluation_and_analysis/RUN_FILES/wiki_2014_with_2019_hyperlink/wikiid2hyperlink_wikid.json') as reader:
        wikiid2hyperlink_wikid = json.load(reader)

    ner_model = load_flair_ner(args.ner_model)
    ed_model = EntityDisambiguation(
        args.base_url, args.wiki_version, {"mode": "eval", "model_path": args.ed_model}
    )
    wikipedia = Wikipedia(args.base_url, args.wiki_version)
    server_address = (args.bind, args.port)
    server = HTTPServer(
        server_address,
        make_handler(
            args.base_url,
            args.wiki_version,
            ed_model,
            ner_model,
            sentence2ner,
            mode=args.mode,
            wikiid2hyperlink_wikid=wikiid2hyperlink_wikid,
            wikipedia=wikipedia,
        ),
    )

    try:
        print("Ready for listening.")
        server.serve_forever()
    except KeyboardInterrupt:
        print()
        exit(0)
