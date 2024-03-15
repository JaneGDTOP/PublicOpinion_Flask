# from nltk import sent_tokenize, word_tokenize
# import nltk
# import os

# nltk.download("punkt")

# with open("./test/test.txt", "r", encoding="utf-8") as file:
#     data = file.readlines()

# test_tokens = []
# for line in data:
#     sentence = line.strip()
#     tokens = word_tokenize(sentence)
# import json

# sentences_info = []
# with open("./test/test.txt.json", "r", encoding="utf-8") as file:
#     for line in file:
#         temp = json.loads(line)
#         sentences_info.append({"token": temp["tokens"], "graph": temp["graph"]})

# with open("./test/result.json", "w", encoding="utf-8") as file:
#     json_str = json.dumps(sentences_info, ensure_ascii=False, indent=4)
#     file.write(json_str)
from transformers import BertTokenizer, BertConfig

tokenizer = BertTokenizer.from_pretrained('/media/ubuntu/projects/pretrainModel/bert-base-chinese', do_lower_case=False)

res=tokenizer.tokenize("金正日提前会晤奥尔布赖 特，也 使得奥卿在平壤的访问行程大幅调动，她原定和北韩国防委员会第 一副委员长兼人民军总 政治局局长赵明录、最高人民会议常任委 员长金永南、外交部长白南纯举行会谈，以及观赏世界 一流水准 的平壤杂技团表演等等，也被迫延后到24号进行。")
print(res)
