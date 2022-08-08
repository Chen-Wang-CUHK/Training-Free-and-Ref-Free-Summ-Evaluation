import os
import json
from collections import OrderedDict
from nltk.tokenize import sent_tokenize

from resources import BASE_DIR
from utils import replace_xml_special_tokens_and_preprocess


class CorpusReader:
    def __init__(self,base_path):
        self.base_path = base_path

    def __call__(self,year):
        # changed by wchen
        assert '08' == year or '09' == year or '2010' == year or '2011' == year or year == 'cnndm'
        if year != 'cnndm':
            if year in ['08', '09']:
                data_path = os.path.join(self.base_path,'data','input_docs','UpdateSumm{}_test_docs_files'.format(year))
                model_path = os.path.join(self.base_path,'data','human_evaluations','UpdateSumm{}_eval'.format(year),'manual','models')
            else:
                # year in [2010, 2011]
                data_path = os.path.join(self.base_path, 'data', 'input_docs', 'GuidedSumm{}_test_docs_files'.format(year))
                model_path = os.path.join(self.base_path, 'data', 'human_evaluations', 'GuidedSumm{}_eval'.format(year), 'manual', 'models')

            docs_dic = self.readDocs(data_path)
            models_dic = self.readModels(model_path)
        else:
            data_file = os.path.join(self.base_path,'data','s_wms_style_files','cnndm_merged_filtered.jsonl')
            docs_dic, models_dic = self.readCNNDM(data_file)

        corpus = []
        for topic in docs_dic:
            entry = []
            entry.append(topic)
            entry.append(docs_dic[topic])
            entry.append(models_dic[topic])
            corpus.append(entry)
        return corpus

    def readModels(self,mpath):
        model_dic = OrderedDict()

        for model in sorted(os.listdir(mpath)):
            topic = self.uniTopicName(model)
            if topic not in model_dic:
                model_dic[topic] = []
            sents = self.readOneModel(os.path.join(mpath,model))
            model_dic[topic].append((os.path.join(mpath,model),sents))

        return model_dic

    def readOneModel(self,mpath):
        ff = open(mpath,'r', encoding='cp1252')
        sents = []
        for line in ff.readlines():
            if line.strip() != '':
                line = replace_xml_special_tokens_and_preprocess(mpath, line)
                if line.strip() != '':
                    sents.append(line.strip())
        ff.close()
        # changed by wchen sents --> sent_tokenize(' '.join(sents))
        return sent_tokenize(' '.join(sents))

    def uniTopicName(self,name):
        doc_name = name.split('-')[0][:5]
        block_name = name.split('-')[1][0]
        return '{}.{}'.format(doc_name,block_name)

    def readDocs(self,dpath):
        data_dic = OrderedDict()

        for tt in sorted(os.listdir(dpath)):
            if tt[0] == '.':
                continue
            for topic in sorted(os.listdir(os.path.join(dpath,tt))):
                topic_docs = []
                doc_names = sorted(os.listdir(os.path.join(dpath,tt,topic)))
                for doc in doc_names:
                    entry = self.readOneDoc(os.path.join(dpath,tt,topic,doc))
                    topic_docs.append((os.path.join(dpath,tt,topic,doc),entry))
                data_dic[self.uniTopicName(topic)] = topic_docs

        return data_dic

    def readOneDoc(self,dpath):
        ff = open(dpath,'r', encoding='cp1252')
        flag = False
        text = []
        for line in ff.readlines():
            if '<TEXT>' in line:
                flag = True
            elif '</TEXT>' in line:
                break
            elif flag and line.strip().lower() != '<p>' and line.strip().lower() != '</p>':
                line = replace_xml_special_tokens_and_preprocess(dpath, line)
                if line.strip() != '':
                    text.append(line.strip())

        ff.close()

        return sent_tokenize(' '.join(text))

    def readCNNDM(self, cnndm_json_file):
        data_dic = OrderedDict()
        model_dic = OrderedDict()
        fr = open(cnndm_json_file, encoding='utf-8')
        cnndm_lines = fr.readlines()
        for line in cnndm_lines:
            """
            {
            'id': str,
            'article': str,
            'sys_summs': {'ml':str, 'ml+rl':str, 'seq2seq':str, 'pointer':str},
            'summs_scores': {'ml+rl':{'overall': float, 'grammar': float, 'redundancy': float},
                            'ml': ...,
                            'seq2seq': ...,
                            'pointer': ...
                            }
            'reference':{'text': str, 'scores':{'overall': float, 'grammar': float, 'redundancy': float}}
            }
            """
            one_data = json.loads(line.strip())
            topic = one_data['id']
            one_doc_line = replace_xml_special_tokens_and_preprocess('cnndm/'+topic+'/doc', one_data['article']).strip()
            one_doc_sents = sent_tokenize(one_doc_line)
            # sys summs
            data_dic[topic] = [(topic, one_doc_sents)]
            # ref summs
            one_ref_line = one_data['reference']['ref_text']
            one_ref_line = replace_xml_special_tokens_and_preprocess('cnndm/'+topic+'/doc', one_ref_line).strip()
            one_ref_sents = sent_tokenize(one_ref_line)
            model_dic[topic] = [(topic, one_ref_sents)]
        return data_dic, model_dic


if __name__ == '__main__':
    reader = CorpusReader(BASE_DIR)
    data = reader('08')

    for topic,docs,models in data:
        print('\n---topic {}, docs {}, models {}'.format(topic,docs[0],models[0]))
