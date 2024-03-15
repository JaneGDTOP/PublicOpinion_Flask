import json
import os
from transformers import BertConfig, BertTokenizer


def read_data(json_filepath):
    data = []
    with open(json_filepath, 'r', encoding='utf-8') as file:
        for line in file:
            temp = json.loads(line)
            data.append(temp)
    return data


def load_tokenizer(bert_name, cache_dir=None):
    if cache_dir != None:
        tokenizer = BertTokenizer.from_pretrained(cache_dir)
    else:
        tokenizer = BertTokenizer.from_pretrained(bert_name)
    return tokenizer


def process_data(data, tokenizer):
    new_data = []
    target_template = {
        "doc_id": "",
        "sent_id": "",
        "tokens": [],
        "pieces": [],
        "token_lens": [],
        "sentence": "莫 斯科索非常愤怒地指责哥国武装团体的暴行并下令警察署立 刻加派警力前往边境加强保护。",
        "entity_mentions": [
            {
                "id": "CBS20001016.0800.0768-E2-20",
                "text": "莫 斯科索",
                "entity_type": "PER",
                "mention_type": "NAM",
                "entity_subtype": "Individual",
                "start": 0,
                "end": 4
            }
        ],
        "relation_mentions": [
            {
                "id": "CBS20001016.0800.0768-R1-1",
                "relation_type": "GEN-AFF",
                "relation_subtype": "GEN-AFF:Citizen-Resident-Religion-Ethnicity",
                "arguments": [
                    {
                        "entity_id": "CBS20001016.0800.0768-E5-4",
                        "text": "团体",
                        "role": "Arg-1"
                    },
                    {
                        "entity_id": "CBS20001016.0800.0768-E6-21",
                        "text": "哥国",
                        "role": "Arg-2"
                    }
                ]
            }
        ],
        "event_mentions": [
            {
                "id": "CBS20001016.0800.0768-EV1-2",
                "event_type": "Conflict:Attack",
                "trigger": {
                    "text": "暴行",
                    "start": 18,
                    "end": 20
                },
                "arguments": [
                    {
                        "entity_id": "CBS20001016.0800.0768-E5-4",
                        "text": "团体",
                        "role": "Attacker"
                    }
                ]
            }
        ]
    }
    for sent in data:
        sent_id = sent["id"]
        sent_text = sent["text"]
        sent_entities_triggers = sent["entities"]
        sent_relations = sent['relations']

        entities = []
        triggers = []

        for entity_or_trigger in sent_entities_triggers:
            if entity_or_trigger['label'] == 'Trigger':
                triggers.append(entity_or_trigger)
            else:
                entities.append(entity_or_trigger)

        tokens = tokenizer.tokenize(sent_text)
        if len(tokens) == len(sent_text):
            pieces = tokens
            token_lens = [1 for _ in tokens]
        else:
            AssertionError("出现错误")

        pass


def get_sentences(data, sentences_filepath='../data/our_data/sentences.txt'):
    sentences = []
    for line in data:
        text = [temp.strip()+'。' for temp in line['text'].split('。')
                if len(temp.strip()) > 0]
        sentences.extend(text)

    with open(sentences_filepath, 'w', encoding='utf-8') as file:
        for line in sentences:
            file.write(line+'\n')


if __name__ == '__main__':
    json_filepath = '../data/our_data/all.jsonl'
    data = read_data(json_filepath)
    get_sentences(data)

    # bert_name = 'bert-base-chinese'
    # cache_dir = '../bert/bert-base-chinese'
    # tokenizer = load_tokenizer(bert_name, cache_dir)

    # sent = "莫 斯科索非常愤怒地指责哥国武装团体的暴行并下令警察署立 刻加派警力前往边境加强保护。"
    # temp = tokenizer.tokenize(sent)
    # print(temp)
