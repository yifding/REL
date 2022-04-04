import os
import argparse
import json
import jsonlines
from http.server import BaseHTTPRequestHandler

from flair.models import SequenceTagger

from REL.mention_detection import MentionDetection
from REL.utils import process_results

API_DOC = "API_DOC"

"""
Class/function combination that is used to setup an API that can be used for e.g. GERBIL evaluation.
"""


def make_handler(base_url, wiki_version, model, tagger_ner, sentence2ner=None):
    class GetHandler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.model = model
            self.tagger_ner = tagger_ner

            self.base_url = base_url
            self.wiki_version = wiki_version

            self.custom_ner = not isinstance(tagger_ner, SequenceTagger)
            self.mention_detection = MentionDetection(base_url, wiki_version)
            self.sentence2ner = sentence2ner

            # **YD** stats the number of match_ner and number of unmatched_ner and total entities

            self.num_total_entities = 0
            self.num_ner_match_entities = 0
            self.num_ner_unmatch_entities = 0

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
            result = process_results(
                mentions_dataset,
                predictions,
                processed,
                include_offset=False if ((len(spans) > 0) or self.custom_ner) else True,
            )

            # Singular document.
            if len(result) > 0:
                tmp_result = [*result.values()][0]
            else:
                tmp_result = []

            re_result = []
            for ele in tmp_result:
                assert len(ele) == 7
                tmp_entity = ele[2]
                tmp_ner = ele[6]

                if self.sentence2ner is not None and tmp_ner != "NULL":
                    self.num_total_entities += 1
                    tmp_processed_entity = tmp_entity.replace('_', ' ')
                    if tmp_processed_entity in self.sentence2ner:
                        # if self.sentence2ner[tmp_processed_entity] == tmp_ner or (self.sentence2ner[tmp_processed_entity] == 'LOC' and tmp_ner == 'MISC'):
                        if self.sentence2ner[tmp_processed_entity] == tmp_ner and tmp_ner != "MISC":
                            self.num_ner_match_entities += 1
                        else:
                            self.num_ner_unmatch_entities += 1
                            continue

                ele += (self.num_total_entities, self.num_ner_match_entities, self.num_ner_unmatch_entities)
                re_result.append(ele)

            return re_result

    return GetHandler


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from http.server import HTTPServer

    from REL.entity_disambiguation import EntityDisambiguation
    from REL.ner import load_flair_ner

    p = argparse.ArgumentParser()
    p.add_argument("--base_url", default="/nfs/yding4/REL/data")
    p.add_argument("--wiki_version", default="wiki_2014")
    p.add_argument("--sentence2ner-file", default="/nfs/yding4/REL/data/sentence2ner/sample_sentence2ner.jsonl")
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


    ner_model = load_flair_ner(args.ner_model)
    ed_model = EntityDisambiguation(
        args.base_url, args.wiki_version, {"mode": "eval", "model_path": args.ed_model}
    )
    server_address = (args.bind, args.port)
    server = HTTPServer(
        server_address,
        make_handler(args.base_url, args.wiki_version, ed_model, ner_model, sentence2ner),
    )

    try:
        print("Ready for listening.")
        server.serve_forever()
    except KeyboardInterrupt:
        print()
        exit(0)
