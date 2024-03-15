import os
import json
import glob
import tqdm
import traceback
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig

from oneie.model import OneIE
from oneie.config import Config
from oneie.util import save_result
from oneie.data import IEDatasetEval
from oneie.convert import json_to_cs


def load_model(model_path, device=1, gpu=False, beam_size=5,language="chinese"):
    print("Loading the model from {}".format(model_path))
    map_location = "cuda:{}".format(device) if gpu else "cpu"
    state = torch.load(model_path, map_location=map_location)

    config = state["config"]
    if type(config) is dict:
        config = Config.from_dict(config)
    # config.bert_cache_dir = os.path.join(cur_dir, 'bert')
    
    vocabs = state["vocabs"]
    valid_patterns = state["valid"]
    
    # if language=="chinese":
    #     config.bert_config='/media/dell/xiehou/project/opinion/PublicOpinion_Flask/oneie/logs/bert-base-chinese'
    #     config.bert_cache_dir='/media/dell/xiehou/project/opinion/PublicOpinion_Flask/oneie/logs/bert-base-chinese'
    # else:
    #     config.bert_config='/media/dell/xiehou/project/opinion/PublicOpinion_Flask/oneie/logs/bert-base-cased'
    #     config.bert_config='/media/dell/xiehou/project/opinion/PublicOpinion_Flask/oneie/logs/bert-base-cased'

    # recover the model
    model = OneIE(config, vocabs, valid_patterns)
    model.load_state_dict(state["model"])
    model.beam_size = beam_size
    if gpu:
        model.cuda(device)
    tokenizer = BertTokenizer.from_pretrained(config.bert_cache_dir, do_lower_case=False)

    return model, tokenizer, config


def predict_document(
    path,
    model,
    tokenizer,
    config,
    batch_size=20,
    max_length=256,
    gpu=False,
    input_format="txt",
    language="english",
):
    """
    :param path (str): path to the input file.
    :param model (OneIE): pre-trained model object.
    :param tokenizer (BertTokenizer): BERT tokenizer.
    :param config (Config): configuration object.
    :param batch_size (int): Batch size (default=20).
    :param max_length (int): Max word piece number (default=128).
    :param gpu (bool): Use GPU or not (default=False).
    :param input_format (str): Input file format (txt or ltf, default='txt).
    :param langauge (str): Input document language (default='english').
    """
    test_set = IEDatasetEval(
        path,
        max_length=max_length,
        gpu=gpu,
        input_format=input_format,
        language=language,
    )
    test_set.numberize(tokenizer)
    # document info
    info = {
        "doc_id": test_set.doc_id,
        "ori_sent_num": test_set.ori_sent_num,
        "sent_num": len(test_set),
    }
    # prediction result
    result = []
    for batch in DataLoader(
        test_set, batch_size=batch_size, shuffle=False, collate_fn=test_set.collate_fn
    ):
        graphs = model.predict(batch)
        for graph, tokens, sent_id, token_ids in zip(
            graphs, batch.tokens, batch.sent_ids, batch.token_ids
        ):
            graph.clean(
                relation_directional=config.relation_directional,
                symmetric_relations=config.symmetric_relations,
            )
            result.append((sent_id, token_ids, tokens, graph))
    res={}
    for sent_id, token_ids, tokens, graph in result:
        output = {
            "sent_id": sent_id,
            "token_ids": token_ids,
            "tokens": tokens,
            "graph": graph.to_dict(),
        }
        print(output)
        print("-"*50)
        _tokens=output["tokens"]
        _sentence=''.join(_tokens)
        _entities=[{"start":ent[0],"end":ent[2],"text":''.join(_tokens[ent[0]:ent[1]]),"type":ent[2]} for ent in output["graph"]["entities"]]
        _triggers=[{"start":tri[0],"end":tri[2],"text":''.join(_tokens[tri[0]:tri[1]]),"type":tri[2]} for tri in output["graph"]["triggers"]]
        _relations=[{"subject":_entities[rel[0]]["text"],"object":_entities[rel[1]]["text"],"relation":rel[2]} for rel in output["graph"]["relations"]]
        _roles=[{"trigger":_triggers[rel[0]]["text"],"argument":_entities[rel[1]]["text"],"role":rel[2]} for rel in output["graph"]["roles"]]
        res={"tokens":_tokens,
               "sentence":_sentence,
               "entities":_entities,
               "triggers":_triggers,
               "entity-relations":_relations,
               "event-role-relations":_roles
               }
    return res

def predict(language,input_text):
    if language=="chinese":
        model_path='/media/dell/xiehou/project/opinion/PublicOpinion_Flask/oneie/logs/cn/best.role.mdl'
    else:
        model_path='/media/dell/xiehou/project/opinion/PublicOpinion_Flask/oneie/logs/en/best.role.mdl'
    device=0
    beam_size=5
    batch_size=10
    max_len=256
    gpu=True
    format="input_text"
    if gpu:
        torch.cuda.set_device(device)
    # load the model from file
    model, tokenizer, config = load_model(
        model_path, device, gpu, beam_size,language
    )
    res=predict_document(input_text,model,tokenizer,config,batch_size=batch_size,max_length=max_len,gpu=gpu,input_format=format,language=language)
    return res