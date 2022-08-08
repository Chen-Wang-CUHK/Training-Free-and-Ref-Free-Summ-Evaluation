import os
import json
from collections import OrderedDict
from nltk.tokenize import sent_tokenize

from resources import BASE_DIR, CNNDM_SUMM_SYSTEMS
from utils import replace_xml_special_tokens_and_preprocess

class PeerSummaryReader:
    def __init__(self,base_path):
        self.base_path = base_path

    def __call__(self,year):
        # changed by wchen
        assert '08' == year or '09' == year or '2010' == year or '2011' == year or 'cnndm' == year
        if year != 'cnndm':
            if year in ['08', '09']:
                data_path = os.path.join(self.base_path,'data','human_evaluations','UpdateSumm{}_eval'.format(year), 'manual','peers')
            else:
                # year in [2010, 2011]
                data_path = os.path.join(self.base_path, 'data', 'human_evaluations', 'GuidedSumm{}_eval'.format(year), 'manual', 'peers')
            summ_dic = self.readPeerSummary(data_path)
        else:
            data_file = os.path.join(self.base_path, 'data', 's_wms_style_files', 'cnndm_merged_filtered.jsonl')
            summ_dic = self.readCNNDMSummary(data_file)

        return summ_dic

    def readPeerSummary(self,mpath):
        peer_dic = OrderedDict()

        for peer in sorted(os.listdir(mpath)):
            topic = self.uniTopicName(peer)
            if topic not in peer_dic:
                peer_dic[topic] = []
            sents = self.readOnePeer(os.path.join(mpath,peer))
            peer_dic[topic].append((os.path.join(mpath,peer),sents))

        return peer_dic

    def readOnePeer(self,mpath):
        ff = open(mpath,'r',encoding='latin-1')
        sents = []
        # changed by wchen for reading from 'manual' folder
        annot_start = False
        peer_start = False
        for line in ff.readlines():
            orig_line = line
            line = line.strip()
            if annot_start and line == '</text>':
                break
            if peer_start:
                assert line.startswith('<line>') and line.endswith('</line>')
                line = line[len('<line>'):-len('</line>')]
                if line.strip() != '':
                    line = replace_xml_special_tokens_and_preprocess(mpath, line)
                    if line.strip() != '':
                        sents.append(line)
            if line == '<annotation>':
                annot_start = True
            if annot_start and (line == '<text>' or '<text length=' in line):
                peer_start = True

        ff.close()
        # changed by wchen sents --> sent_tokenize(' '.join(sents))
        return sent_tokenize(' '.join(sents))

    def uniTopicName(self,name):
        doc_name = name.split('-')[0][:5]
        block_name = name.split('-')[1][0]
        return '{}.{}'.format(doc_name,block_name)

    def readCNNDMSummary(self, cnndm_json_file):
        peer_dic = OrderedDict()
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
            if topic not in peer_dic:
                peer_dic[topic] = []
            for sys_name in sorted(one_data['sys_summs'].keys()):
                one_summ = one_data['sys_summs'][sys_name]
                one_summ = replace_xml_special_tokens_and_preprocess('cnndm/'+topic+'/summ/'+sys_name, one_summ).strip()
                one_summ = sent_tokenize(one_summ)
                peer_dic[topic].append((os.path.join(topic,sys_name), one_summ))
        return peer_dic



if __name__ == '__main__':
    peerReader = PeerSummaryReader(BASE_DIR)
    summ = peerReader('08')

    for topic in summ:
        print('topic {}, summ {}'.format(topic,summ[topic]))