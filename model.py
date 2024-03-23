import torch
import torch.nn as nn
from transformers import BartModel,BartTokenizerFast,BertTokenizer
tokenizer=BertTokenizer.from_pretrained('/media/dell/pretrainModel/fnlp/bart-base')
tokenizer.add_special_tokens(
                    {'additional_special_tokens': ["<trigger>", "</trigger>"]})
label_id_dict={'Causal':0,'Follow':1,'NoRel':2}
id_label_dict={0:'Causal',1:'Follow',2:'NoRel'}
label_to_token_id={0:[7149,11370],1:[21002,23182],2:[10931,5770]}

class PERE(nn.Module):
    def __init__(self,device):
        super(PERE, self).__init__()
        self.device=device
        self.model=BartModel.from_pretrained('./fnlp/bart-base-chinese')
        self.model.resize_token_embeddings(len(tokenizer))
        self.clf_lossfun=nn.CrossEntropyLoss(reduction='mean')
        self.mask_lossfun=nn.CrossEntropyLoss(reduction='mean')
        self.fc=nn.Linear(768,3)
        self.relu=nn.LeakyReLU(0.2)
        self.dropout=nn.Dropout(0.2)
        self.softmax=nn.Softmax()
        self.linear=nn.Linear(768,tokenizer.vocab_size)
    def forward(self,enc_input_ids,enc_mask_ids,dec_input_ids,dec_mask_ids,mask_positions,targets,token_ids,training=True):
        contexts=self.model(enc_input_ids,attention_mask=enc_mask_ids)
        contexts_hidden=contexts.encoder_last_hidden_state
        #contexts=contexts.last_hidden_state

        decoder_contexts=self.model.decoder(input_ids=dec_input_ids,attention_mask=dec_mask_ids,
                encoder_hidden_states=contexts_hidden,encoder_attention_mask=enc_mask_ids)
        decoder_contexts=decoder_contexts.last_hidden_state
        mask_vector=list()
        for i,mask_position in enumerate(mask_positions):
            mask_vector.append(torch.cat([decoder_contexts[i,mask_position[0]:mask_position[1]+1,:]]))
        mask_vector=torch.stack(mask_vector)
        logits=self.dropout(self.relu(self.fc(torch.mean(mask_vector,dim=1))))
        predictions = self.linear(mask_vector)
        #predicted_ids = predictions.argmax(dim=-1).tolist()
        if training:
            loss_clf=self.clf_lossfun(logits,targets.long())
            loss_mask=self.mask_lossfun(predictions.view(-1, predictions.shape[-1]),token_ids.view(-1))
            total_loss=0.7*loss_clf+0.3*loss_mask
            #total_loss=loss_clf
            return total_loss
        else:
            return logits,predictions

