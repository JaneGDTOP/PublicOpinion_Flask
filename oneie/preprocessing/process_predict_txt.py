import json
import os


def get_data(json_filepath):
    data = []
    with open(json_filepath, 'r', encoding='utf-8') as file:
        for line in file:
            temp = json.loads(line)
            data.append(temp)
    return data


def process_data2annotation(data, output_filepath):
    new_data = []
    for line in data:
        temp_line = dict()
        tokens = line['tokens']
        graph = line['graph']
        sentence = ''.join(tokens)
        assert len(sentence) == len(tokens), "please check your sentence"

        _entities = graph['entities']
        _triggers = graph['triggers']
        _relations = graph['relations']
        _roles = graph['roles']

        temp_line['sentence'] = sentence
        temp_line['tokens'] = tokens
        temp_line['entity_mentions'] = []

        for index, entity_info in enumerate(_entities):
            temp_entity = dict()
            temp_entity["id"] = index
            temp_entity["start"] = entity_info[0]
            temp_entity["end"] = entity_info[1]
            # print(sentence, entity_info[0], entity_info[1], len(sentence))
            assert entity_info[0] < len(sentence) or entity_info[1] < len(
                sentence), "please check token is correct"
            temp_entity["text"] = sentence[entity_info[0]:entity_info[1]]
            temp_entity["entity_type"] = entity_info[2]
            temp_entity["mention_type"] = entity_info[3]
            temp_entity["probability"] = entity_info[4]
            temp_line["entity_mentions"].append(temp_entity)

        temp_line['relation_mentions'] = []
        for index, relation in enumerate(_relations):
            temp_relation = dict()
            entity_1 = temp_line["entity_mentions"][relation[0]]
            entity_2 = temp_line["entity_mentions"][relation[1]]
            temp_relation['id'] = index
            temp_relation['relation_type'] = relation[2]
            temp_relation['arguments'] = [
                {
                    "entity_id": relation[0],
                    "text": entity_1['text'],
                    "role": "Arg-1",
                    "probability":entity_1['probability']
                },
                {
                    "entity_id": relation[1],
                    "text": entity_2['text'],
                    "role": "Arg-2",
                    "probability":entity_2['probability']
                }
            ]
            temp_line['relation_mentions'].append(temp_relation)

        triggers = []
        for trigger in _triggers:
            temp_trigger = dict()
            temp_trigger['text'] = sentence[trigger[0]:trigger[1]]
            temp_trigger['trigger_type'] = trigger[2]
            temp_trigger['probability'] = trigger[3]
            temp_trigger['start'] = trigger[0]
            temp_trigger['end'] = trigger[1]
            triggers.append(temp_trigger)

        temp_line['event_mentions'] = []
        trigger_index_set = set()
        index = 0
        for role in _roles:

            trigger_index = role[0]
            argument = temp_line["entity_mentions"][role[1]]
            if trigger_index not in trigger_index_set:
                temp_role = dict()
                trigger = triggers[trigger_index]
                temp_role['id'] = index
                temp_role['event_type'] = trigger['trigger_type']
                temp_role['trigger'] = {
                    "text": trigger['text'],
                    "start": trigger['start'],
                    "end": trigger['end']
                }
                temp_role['arguments'] = [
                    {
                        "entity_id": argument['id'],
                        "text": argument['text'],
                        "role": role[2],
                        "probability":role[3]
                    }
                ]
                temp_line['event_mentions'].append(temp_role)
            else:
                for i in range(len(temp_line['event_mentions'])):
                    temp_role = temp_line['event_mentions'][i]
                    if temp_role['trigger']['text'] == triggers[trigger_index]['text']:
                        temp_line['event_mentions'][i]['arguments'].append(
                            {
                                "entity_id": argument['id'],
                                "text": argument['text'],
                                "role": role[2],
                                "probability": role[3]
                            }
                        )
                        break

            trigger_index_set.add(trigger_index)
            index += 1
        new_data.append(temp_line)

    with open(output_filepath, 'w', encoding='utf-8') as file:
        for temp in new_data:
            json_str = json.dumps(temp, ensure_ascii=False)
            file.write(json_str+'\n')


if __name__ == '__main__':
    json_filepath = '../test/sentences.txt.json'
    data = get_data(json_filepath)
    process_data2annotation(data, '../test/annotation_lines.json')
