import os
import json
import argparse
import collections

from REL.wikipedia import Wikipedia
from REL.wikipedia_yago_freq import WikipediaYagoFreq


def main(args):
    base_url = args.base_url
    wiki_version = args.wiki_version
    wikipedia = Wikipedia(base_url, wiki_version)
    wiki_yago_freq = WikipediaYagoFreq(base_url, wiki_version, wikipedia)
    wiki_anchor_file = args.wiki_anchor_file
    assert os.path.isfile(wiki_anchor_file)

    num_lines = 0
    num_valid_hyperlinks = 0
    disambiguation_ent_errors = 0

    num_distance = 0
    last_processed_id = -1
    exist_id_found = False


    wikiid2hyperlink_wikid = dict()
    wikiname2hyperlink_wikiname = dict()

    with open(wiki_anchor_file, "r", encoding="utf-8") as f:
        for line in f:
            num_lines += 1
            if num_lines % 5000000 == 0:
            # test the first 10000 lines
            # if num_lines % 10000 == 0:
                print(
                    "Processed {} lines, valid hyperlinks {}".format(
                        num_lines, num_valid_hyperlinks
                    )
                )

            if '<doc id="' in line:
                num_distance = 0
                id = int(line[line.find("id") + 4: line.find("url") - 2])
                if id <= last_processed_id:
                    exist_id_found = True
                    continue
                elif id not in wikipedia.wiki_id_name_map["ent_id_to_name"]:
                    exist_id_found = True
                    continue
                else:
                    exist_id_found = False
                    last_processed_id = id
                    if id not in wikiid2hyperlink_wikid:
                        wikiid2hyperlink_wikid[id] = dict()
                        wikiname = wikipedia.wiki_id_name_map[
                            "ent_id_to_name"
                        ][id].replace(" ", "_")
                        wikiname2hyperlink_wikiname[wikiname] = dict()
            else:
                if exist_id_found:
                    continue

                num_distance += 1
                (
                    list_hyp,
                    disambiguation_ent_error,
                    print_values,
                ) = wiki_yago_freq.extract_text_and_hyp(line)

                disambiguation_ent_errors += disambiguation_ent_error

                for el in list_hyp:
                    mention = el["mention"]
                    ent_wiki_id = el["ent_wikiid"]
                    ent_num_mentions = el['cnt']

                    if (
                        ent_wiki_id
                        in wikipedia.wiki_id_name_map["ent_id_to_name"]
                    ):
                        num_valid_hyperlinks += 1
                        if ent_wiki_id not in wikiid2hyperlink_wikid[id]:
                            wikiid2hyperlink_wikid[id][ent_wiki_id] = (num_distance, ent_num_mentions)
                            ent_wiki_name = wikipedia.wiki_id_name_map[
                            "ent_id_to_name"
                        ][ent_wiki_id].replace(" ", "_")
                            wikiname2hyperlink_wikiname[wikiname][ent_wiki_name] = (num_distance, ent_num_mentions)

                        else:
                            pass


    with open('wikiid2hyperlink_wikid.json', 'w') as writer:
        writer.write(json.dumps(wikiid2hyperlink_wikid, indent=4))

    with open('wikiname2hyperlink_wikiname.json', 'w') as writer:
        writer.write(json.dumps(wikiname2hyperlink_wikiname, indent=4))

    for entity in ['david beckham', 'victoria beckham', 'U.S. Open (golf)', 'tiger woods', 'Japan_national_football_team']:
        wikiid = wikipedia.ent_wiki_id_from_name(entity)
        if wikiid != -1:
            ent_name = wikipedia.wiki_id_name_map[
                "ent_id_to_name"
            ][wikiid].replace(' ', '_')
            print('ent_name', ent_name)
            print(wikiname2hyperlink_wikiname[ent_name])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--base_url", default="/nfs/yding4/REL/data/")
    p.add_argument("--wiki_version", default="wiki_2014")
    p.add_argument("--wiki_anchor_file", default="/nfs/yding4/REL/data/wiki_2014/basic_data/anchor_files/textWithAnchorsFromAllWikipedia2014Feb.txt")
    p.add_argument("--output_dir", default="hyper_link_wiki_2014")

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
