import json
import numpy as np

def transformer_standard_data(filepath,tri2id,role2id):
    dates = []
    sentIds=set()
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            item = {}
            item["id"]=line["id"]
            sentIds.add(line["id"])
            item["content"] = line["content"]
            item["event"]={"ti":set(),"tc":set(),"ai":set(),"ac":set()}
            for event in line["events"]:
                t_s,t_e=event["trigger"]["span"]
                t_c=tri2id[event["type"]]
                item["event"]["ti"].add((t_s,t_e))
                item["event"]["tc"].add((t_s,t_e,t_c))
                for k,v in event["args"].items():
                    for arg in v:
                        a_s,a_e=arg["span"]
                        a_c=role2id[k]
                        item["event"]["ai"].add((a_s,a_e,t_c))
                        item["event"]["ac"].add((a_s,a_e,t_c,a_c))
            dates.append(item)
    return dates,sentIds

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        else:
            return super(MyEncoder, self).default(obj)

def write_result(filepath,data):
    with open(filepath,'w',encoding='utf-8') as file:
        for t in data:
            t_str=json.dumps(t,ensure_ascii=False,cls=MyEncoder)
            file.write(t_str+'\n')

def read_oneid_pred(json_filepath):
    lines=[]
    with open(json_filepath,'r',encoding='utf-8') as file:
        for line in file:
            lines.append(json.loads(line))
    return lines

def oneie_pred2stand(lines,ids,tri2id,role2id):
    dates=[]
    for line in lines:
        item={}
        if line['sent_id'] not in ids:
            continue
        item['id']=line['sent_id']
        item["content"] = "".join(line["tokens"])

        item["event"]={"ti":set(),"tc":set(),"ai":set(),"ac":set()}
        for trigger in line["pred"]["triggers"]:
            item["event"]["ti"].add((trigger[0],trigger[1]))
            item["event"]["tc"].add((trigger[0],trigger[1],tri2id[trigger[2]]))
        for role in line["pred"]["roles"]:
            item["event"]["ai"].add((line["pred"]["entities"][role[1]][0],line["pred"]["entities"][role[1]][1],tri2id[line["pred"]["triggers"][role[0]][2]]))
            item["event"]["ac"].add((line["pred"]["entities"][role[1]][0],line["pred"]["entities"][role[1]][1],tri2id[line["pred"]["triggers"][role[0]][2]],role2id[role[2]]))
                        
        dates.append(item)
    return dates

def read_tag2id(json_filepath):
    with open(json_filepath,'r',encoding='utf-8')as file:
        tag2id=json.load(file)
    return tag2id

def read_data(json_filepath,ids=None):
    data=[]
    with open(json_filepath,'r',encoding='utf-8') as file:
        for line in file:
            line=json.loads(line)
            item={}
            item['id']=line['id']
            if ids!=None and line["id"] in ids:
                item['content']=line['content']
                item['event']={"ti":set(),"tc":set(),"ai":set(),"ac":set()}
                for ti in line['event']['ti']:
                    item['event']['ti'].add((ti[0],ti[1]))
                for tc in line['event']['tc']:
                    item['event']['tc'].add((tc[0],tc[1],tc[2]))  
                for ai in line['event']['ai']:
                    item['event']['ai'].add((ai[0],ai[1],ai[2]))
                for ac in line['event']['ac']:
                    item['event']['ac'].add((ac[0],ac[1],ac[2],ac[3]))
                data.append(item)
            elif ids==None:
                item['content']=line['content']
                item['event']={"ti":set(),"tc":set(),"ai":set(),"ac":set()}
                for ti in line['event']['ti']:
                    item['event']['ti'].add((ti[0],ti[1]))
                for tc in line['event']['tc']:
                    item['event']['tc'].add((tc[0],tc[1],tc[2]))  
                for ai in line['event']['ai']:
                    item['event']['ai'].add((ai[0],ai[1],ai[2]))
                for ac in line['event']['ac']:
                    item['event']['ac'].add((ac[0],ac[1],ac[2],ac[3]))
                data.append(item)
            else:
                continue
    return data

def calculate_f1(r, p, c):
    if r == 0 or p == 0 or c == 0:
        return 0, 0, 0
    r = c / r
    p = c / p
    f1 = (2 * r * p) / (r + p)
    return f1, r, p


if __name__=="__main__":
    # step 1
    '''
    tri2id=read_tag2id('./data/ace05cn/event2id.json')
    role2id=read_tag2id('./data/ace05cn/role2id.json')

    oneie_pred_json_filepath='./logs/20231018_221440/result.test.json'
    oneie_pred=read_oneid_pred(oneie_pred_json_filepath)

    stand_dates,sentIds=transformer_standard_data('./data/ace05cn/test.json',tri2id,role2id)
    write_result('./pred/test_stand.json',stand_dates)

    pred_dates=oneie_pred2stand(oneie_pred,sentIds,tri2id,role2id)
    write_result('./pred/pred_stand.json',pred_dates)
    '''
    # step 2
    # '''
    test_stand_filepath='./pred/test_stand.json'
    test_pred_filepath='./pred/pred_stand.json'
    test_stand=read_data(test_stand_filepath)
    test_pred=read_data(test_pred_filepath)

    results = {k + "_" + t: 0 for k in ["ti", "tc", "ai", "ac"] for t in ["r", "p", "c"]}
    for stand,pred in zip(test_stand,test_pred):
        if stand['id']!=pred['id']:
            break
        tuple_label=stand['event']  
        pred_label=pred['event']
        for key in ["ti","tc","ai","ac"]:
            results[key + "_r"] += len(tuple_label[key])
            results[key + "_p"] += len(pred_label[key])
            results[key + "_c"] += len(pred_label[key] & tuple_label[key])
    
    ti_f1, ti_r, ti_p = calculate_f1(results["ti_r"], results["ti_p"], results["ti_c"])
    tc_f1, tc_r, tc_p = calculate_f1(results["tc_r"], results["tc_p"], results["tc_c"])
    ai_f1, ai_r, ai_p = calculate_f1(results["ai_r"], results["ai_p"], results["ai_c"])
    ac_f1, ac_r, ac_p = calculate_f1(results["ac_r"], results["ac_p"], results["ac_c"])
    
    print("{}\t{}\t{}\t{}".format("classify ", 'F1', "Precision", "Recall"))
    print("Trigger  I\t{:3.4f}\t{:3.4f}\t{:3.4f}".format(ti_f1, ti_p, ti_r))
    print("Trigger  C\t{:3.4f}\t{:3.4f}\t{:3.4f}".format(tc_f1, tc_p, tc_r))
    print("Argument I\t{:3.4f}\t{:3.4f}\t{:3.4f}".format(ai_f1, ai_p, ai_r))
    print("Argument C\t{:3.4f}\t{:3.4f}\t{:3.4f}".format(ac_f1, ac_p, ac_r))
    # '''