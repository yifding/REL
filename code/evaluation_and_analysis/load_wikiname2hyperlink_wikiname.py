import json

from REL.wikipedia import Wikipedia
from REL.wikipedia_yago_freq import WikipediaYagoFreq

with open('wikiname2hyperlink_wikiname.json') as reader:
    wikiname2hyperlink_wikiname = json.load(reader)

base_url = '/nfs/yding4/REL/data/'
wiki_version = 'wiki_2014'
wikipedia = Wikipedia(base_url, wiki_version)

for entity in ['david beckham', 'victoria beckham', 'U.S. Open (golf)', 'tiger woods', 'Justin Bieber']:
    wikiid = wikipedia.ent_wiki_id_from_name(entity)
    if wikiid != -1:
        ent_name = wikipedia.wiki_id_name_map[
            "ent_id_to_name"
        ][wikiid].replace(' ', '_')
        print('ent_name', ent_name)
        print(wikiname2hyperlink_wikiname[ent_name])
