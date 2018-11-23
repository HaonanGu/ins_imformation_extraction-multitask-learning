# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import json
import os
# from Doc.Document import Document
import tensorflow as tf
import gensim
import random
import numpy as np
import sklearn.metrics as skmetrics
import math
import copy
import string
import cPickle
from TreeBuilderNeuralNetwork import TreeBuilderNeuralNetwork
from TreeBuilderWithSubtitle import TreeBuilderWithSubtitle
from RoughClassifier import RoughClassifier
from StanfordNLP.StanfordNLP import StanfordNLP
# from BilstmCrf import BilstmCrf

__metaclass__ = type

class MultiTask:pass

class BilstmCrfMultitask:

    def __init__(self,name_scope=''.join(random.sample(string.ascii_letters + string.digits, 10)),
                 model_path='Classifier_model/BilstmCrf_models/baoxianzeren_and_zerenmianchu/',
                 w2v_path='zhwiki_finance_simple.sg_50d.word2vec'):

        self._input_size=50
        self._rc_number=31
        self._lstm_hidden_size=512
        self._category_num=5
        self._batch_size=10
        self._learning_rate=5e-4
        self._echo_num=200
        self._keep_prob_rate=0.7

        self.name_scope=name_scope
        self.model_path = model_path
        self.w2v = self._load_embedding(w2v_path)
        # self.build_model(name_scope=name_scope)

        self.print_no_found_words = False

    def _load_embedding(self,path):
        word_vectors = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)
        return word_vectors

    def _bilstm(self):
        with tf.name_scope("bilstm_layer"):
            lstm_cell = tf.contrib.rnn.LSTMCell(self._lstm_hidden_size)
            lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=self.keep_prob)
            lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=self.keep_prob)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw,cell_bw=lstm_cell_bw,
                                                                        inputs=self.x_text,
                                                                        initial_state_fw=None,
                                                                        initial_state_bw=None,
                                                                        sequence_length=self.original_sequence_lengths,
                                                                        dtype=tf.float32)
            context_rep = tf.concat([output_fw, output_bw], axis=-1)
            return context_rep

    def _fully_connection(self,xcon,y_labels,original_sequence_lengths,ntime_steps,name_space):
        with tf.name_scope(name_space+'_fully_connection'):
            w = tf.Variable(tf.truncated_normal([self._lstm_hidden_size*2+self._rc_number, self._category_num], stddev=0.1),
                dtype=tf.float32)
            b = tf.Variable(tf.constant(0.1, shape=[self._category_num]), dtype=tf.float32)
            y_fc = tf.nn.dropout(tf.matmul(xcon, w) + b, self.keep_prob)
            scores = tf.reshape(y_fc, [-1, ntime_steps, self._category_num])
            return self._crf_loss(scores,y_labels,original_sequence_lengths,name_space+'_crf_loss_layer')

    def _crf_loss(self,scores,y_labels,original_sequence_lengths,name_space):
        with tf.variable_scope(name_space,reuse=tf.AUTO_REUSE):
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(scores, y_labels, original_sequence_lengths)
            loss=tf.reduce_mean(-log_likelihood)
            train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(loss)
            #for test, usage
            viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(scores, transition_params,original_sequence_lengths)
            return {'train_op':train_op,'transition_params':transition_params,'loss':loss,
                    'viterbi_sequence':viterbi_sequence,'viterbi_score':viterbi_score}

    def build_model(self,model_name_spaces,view_tensorboard=False):

        with tf.name_scope("model_input_layers"):
            self.x_text = tf.placeholder(tf.float32, [None, None, self._input_size])
            self.x_rc = tf.placeholder(tf.float32, [None, None, self._rc_number])
            self.original_sequence_lengths= tf.placeholder(tf.int32, [None])
            for name in model_name_spaces.keys():
                exec 'self.'+str(name)+' = MultiTask()'
                exec 'self.'+str(name)+ '.y_labels = tf.placeholder(tf.int32, [None, None])'

            self.keep_prob = tf.placeholder(tf.float32, [])

        with tf.name_scope("model_bulid_layers"):
            bilstm=self._bilstm()
            ntime_steps = tf.shape(bilstm)[1]
            bilstm=tf.reshape(bilstm, [-1, 2 * self._lstm_hidden_size])
            xcon=tf.concat([bilstm,tf.reshape(self.x_rc,[-1,self._rc_number])],1)

            for name in model_name_spaces.keys():
                bz_id=model_name_spaces.get(name)
                self._category_num = 2 * len(bz_id) + 1

                y_labels=eval('self.'+str(name)+'.y_labels')
                rt=self._fully_connection(xcon, y_labels, self.original_sequence_lengths, ntime_steps, name_space=name)
                train_op,transition_params,loss,viterbi_sequence,viterbi_score\
                    =rt['train_op'],rt['transition_params'],rt['loss'],rt['viterbi_sequence'],rt['viterbi_score']
                exec 'self.'+str(name)+'.train_op=train_op'
                exec 'self.' + str(name) + '.transition_params=transition_params'
                exec 'self.' + str(name) + '.loss=loss'
                exec 'self.' + str(name) + '.viterbi_sequence=viterbi_sequence'
                exec 'self.' + str(name) + '.viterbi_score=viterbi_score'

            print('the multi-task BiLSTM-CRF model is built.')
        if view_tensorboard:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                writer = tf.summary.FileWriter('log/', tf.get_default_graph())
                writer.close()
            try:
                # os.system("gnome-terminal -e 'tensorboard --logdir=log'")
                os.system("gnome-terminal -e 'bash -c \"tensorboard --logdir=log; exec bash\"'")
                print('\033[1;30;42mPlease click the link, displayed on terminal, to open tensorboard.\033[0m')
            except:
                print('Error occurred when open terminal, you can input the code in terminal:\033[1;35m tensorboard --logdir=log \033[0m ')

    @staticmethod
    def build_all_tree(json_path='Corpus_data/Corpus.json'):
        TS = TreeBuilderWithSubtitle()
        trees, subtitle_names = TS.build_tree_with_subtitle_from_json(json_path=json_path)
        return trees,subtitle_names

    @staticmethod
    def _add_tags(taga,tagb):
        '''
        'o'+'o'='o'
        'B-X'+'o'='B-X'
        'I-X'+'o'='I-X'
        其他都不行

        taga: 'o o o B-X I-X o o  o   o   o  o o'
        taga: 'o o o  o   o  o o B-Y I-Y I-Y o o'
        '''
        taga=taga.split()
        tagb=tagb.split()
        if len(taga)!=len(tagb):raise ValueError('taga and tagb do not have same length')
        else:
            r=[]
            for i,j in zip(taga,tagb) :
                if i=='O' and j=='O':  ##'O' is Ou, not 0
                    r.append('O')
                elif i=='O' and j!='O':
                    r.append(j)
                elif i != 'O' and j == 'O':
                    r.append(i)
                elif i != 'O' and j != 'O':
                    raise ValueError('taga and tagb can not add')
        return ' '.join(r)

    @staticmethod
    def build_dataset(trees,bz_id,neg_num=500):
        # bz_id = {'39': 'SYRS', '40':'MRSYR'}

        negative_data=[]
        positive_data=[]
        jd=0
        for tree in trees:
            for t in tree:
                annotation=t['other_feature']['annotation']
                if annotation==None:
                    continue
                else:
                    append_to_previous=False
                    tag_pos=True
                    keys=[a.get('key') for a in annotation]
                    for i in bz_id.keys():
                        if keys.count(i)>0:
                            a=annotation[keys.index(i)]
                            tmp={'subtitle_predit_feature':t['subtitle']['predit_feature'],
                                 'subtitle_predit_name':t['subtitle']['predit_name'],
                                 'text':t['text'],
                                 'text_with_number':t['text_with_number'],
                                 'depth':t['depth'],
                                 'annotation':a,
                                 'append_to_previous':append_to_previous}
                            positive_data.append(tmp)
                            append_to_previous = True
                            tag_pos = False
                    if tag_pos:
                        tmp = {'subtitle_predit_feature': t['subtitle']['predit_feature'],
                               'subtitle_predit_name': t['subtitle']['predit_name'],
                               'text': t['text'],
                               'text_with_number': t['text_with_number'],
                               'depth': t['depth'],
                               'annotation': None,
                               'append_to_previous': append_to_previous}
                        negative_data.append(tmp)
            jd+=1
            print('\rloading data from trees : %d%% '%(jd*100/len(trees)),end='')

        print('\nbuilding dataset...')

        data=[]
        for d in positive_data:
            key=d['annotation']['key']
            B='B-'+bz_id[key]
            I='I-'+bz_id[key]
            Ou='O'

            start=d['annotation']['wchar_start']
            end = d['annotation']['wchar_end']
            text=d['text']
            if text.count(' '):
                if text.index(' ')<start:
                    start-=text[:start].count(' ')
                    end-=text[:start].count(' ')
                elif start<text.index(' ')<end:
                    end -= text[start:end].count(' ')

            segments = StanfordNLP.segment(text)
            text_seg = RoughClassifier.segment_sentence(text, copy.deepcopy(segments)).split()

            text_ahead=text[:start]
            text_behind=text[end:]

            ind1=ind2=0
            for i in range(len(segments)):
                if ''.join(text_seg[:i])==text_ahead:
                    ind1=i
                    break
            ran=range(len(segments)-1)
            ran.reverse()
            for i in ran:
                if ''.join(text_seg[i:])==text_behind:
                    ind2=i
                    break

            taged_text=text_seg[ind1:ind2]
            BIO=[]
            if len(taged_text)==0:
                continue
            elif len(taged_text)==1:BIO=[B]
            elif len(taged_text)>=2:BIO=[B]+[I]*(len(taged_text)-1)

            qian=[Ou]*len(text_seg[:ind1])
            hou=[Ou]*len(text_seg[ind2:])

            data.append([[u' '.join(text_seg),d['subtitle_predit_feature']],u' '.join(qian+BIO+hou),d['append_to_previous']])

        while [i[2]for i in data].count(True):
            index= list([i[2] for i in data]).index(True)
            taga=data[index-1][1]
            tagb=data[index][1]
            try:
                addtag=BilstmCrfMultitask._add_tags(taga,tagb)
            except:
                data[index][2] = False
                continue
            data[index - 1][1]=addtag
            data.pop(index)

        ##negative
        random.seed(100)
        if neg_num == -1: neg_num = BilstmCrfMultitask.auto_neg_num(len(data))
        if neg_num>len(negative_data): neg_num=len(negative_data)
        negative_data=random.sample(negative_data,neg_num)

        data_n=[]
        for t in negative_data:
            text = t['text']
            segments = StanfordNLP.segment(text)
            text_seg = RoughClassifier.segment_sentence(text, copy.deepcopy(segments)).split()
            predit_feature = t['subtitle_predit_feature']
            label = ' '.join(['O'] * len(text_seg))
            data_n.append([[u' '.join(text_seg), list(predit_feature)], label])


        print('dataset size --> pos: %d, neg: %d'%(len(data),len(data_n)))

        return [i[:2]for i in data]+data_n

    def get_feed_dict_train(self,one_batch,tags,branch_name,train_test='train'):
        w2v_dict=self.w2v.vocab
        # original_lengths=[len(d[1].split()) for d in one_batch]
        original_lengths=[]
        one_batch_tmp=[]
        one_batch_labels_tmp=[]
        for d in one_batch:
            text_list=d[0][0].split()
            label=d[1].split()

            no_in_dict_index=[i for i in range(len(text_list)) if text_list[i] not in w2v_dict]
            no_in_dict_index.reverse()
            for i in no_in_dict_index:
                if self.print_no_found_words==True:
                    print('有不在字典中的词，已跳过，请添加： ',text_list.pop(i))
                else: text_list.pop(i)
                label.pop(i)
            original_lengths.append(len(text_list))
            one_batch_tmp.append([text_list,d[0][1]])
            one_batch_labels_tmp.append(' '.join(label))

        max_len=max(original_lengths)
        x_text=[]
        x_rc=[]
        y_labels=[]
        for a,b in zip(one_batch_tmp,one_batch_labels_tmp):
            # tag_zero = np.zeros([len(tags)])
            a1=self.w2v[a[0]]
            if max_len>len(a1):
                a1_pad=np.zeros([max_len-len(a1),self._input_size])
                a1=np.array(list(a1) + list(a1_pad),dtype=np.float32)
            x_text.append(a1)
            x_rc.append(np.array([np.reshape(a[1], [len(a[1])]) for _ in range(max_len)]))

            label=[tags[i] for i in b.split()]
            while(len(label)<max_len):
                label.append(tags[u'O'])

            y_labels.append(list(label))

        x_text=np.array(x_text,dtype=np.float32)
        x_rc=np.array(x_rc,dtype=np.float32)
        original_lengths=np.array(original_lengths,dtype=np.int32)
        y_labels=np.array(y_labels,dtype=np.int32)

        keep_prob_rate=0.7
        if train_test=='train':
            if branch_name=='times':keep_prob_rate=0.7
            elif branch_name=='ages' or branch_name=='payments':keep_prob_rate=0.6
        else:
            keep_prob_rate = 1.0

        feed = eval('{self.x_text:x_text,self.x_rc:x_rc,self.original_sequence_lengths:original_lengths,self.'
                    + branch_name + '.y_labels:y_labels,self.keep_prob:keep_prob_rate}')

        # if train_test=='train':
        #     feed=eval('{self.x_text:x_text,self.x_rc:x_rc,self.original_sequence_lengths:original_lengths,self.'
        #               +branch_name+'.y_labels:y_labels,self.keep_prob:self._keep_prob_rate}')
        #     # feed={self.x_text:x_text,self.x_rc:x_rc,self.original_sequence_lengths:original_lengths,
        #     #       self.y_labels:y_labels,self.keep_prob:self._keep_prob_rate}
        # else:
        #     feed = eval('{self.x_text:x_text,self.x_rc:x_rc,self.original_sequence_lengths:original_lengths,self.'
        #                 + branch_name + '.y_labels:y_labels,self.keep_prob:1.0}')
        #     # feed={self.x_text:x_text,self.x_rc:x_rc,self.original_sequence_lengths:original_lengths,
        #     #       self.y_labels:y_labels,self.keep_prob:1.0}
        return feed

    @staticmethod
    def _chunk_tags(data):
        data=data['test']+[i for abatch in data['train_batchs'] for i in abatch]

        s=u''
        for i in data:
            s=s+i[1]+' '
        keys=list(set(s.strip().split()))
        Ou_key=keys.pop(keys.index(u'O'))
        keys=[Ou_key]+sorted(keys,key=lambda x:(x.split('-')[1],x.split('-')[0]))
        values=range(0,len(keys))
        tags=dict(zip(keys, values))
        return tags

    @staticmethod
    def get_batchs(data,batch_size):
        num_batchs=math.ceil(len(data)/float(batch_size))
        random.shuffle(data)
        r=[]
        for i in range(int(num_batchs)):
            d=data[i*batch_size:(i+1)*batch_size]
            r.append(d)
        return r

    @staticmethod
    def mertics(y_true_n,y_pred_n):
        metrics_score = {}
        metrics_score_flatten = {}
        ## y_true_n 序列的真实标注; y_pred_n 序列的预测标注
        ## y_true_n 格式 [[0,0,1,2,2,0,0,0],[0,0,3,4,4,4,0,0],[0,0,1,2,2,2,2,0],...]
        ## y_pred_n 格式 [[0,0,1,2,2,2,0,0],[0,0,3,4,4,4,0,0],[0,0,1,2,2,2,2,0],...]
        y_true_n_flatten=[i for val in y_true_n for i in val]
        y_pred_n_flatten = [i for val in y_pred_n for i in val]

        if len(y_true_n)!=len(y_pred_n) or len(y_true_n_flatten)!=len(y_pred_n_flatten):
            raise ValueError('len(y_true_n)!=len(y_pred_n), or len(y_true_n_flatten)!=len(y_pred_n_flatten)')
        if len(y_true_n)!=len(y_pred_n) or len(y_true_n_flatten)!=len(y_pred_n_flatten):
            raise ValueError('len(y_true_n)!=len(y_pred_n), or len(y_true_n_flatten)!=len(y_pred_n_flatten)')
        y_true_n = [list(i) for i in y_true_n]
        y_pred_n = [list(i) for i in y_pred_n]

        d=[]
        label_n=1
        y_true_label=[]
        for i in y_true_n:
            if i not in d:
                d.append(i)
                y_true_label.append(label_n)
                label_n+=1
            else:
                y_true_label.append(d.index(i)+1)
        y_pred_label=[d.index(i)+1 if i in d else 0 for i in y_pred_n]

        # metrics_score['accuracy']=sum([1 if i==j else 0 for i,j in zip(y_true_n,y_pred_n)])/float(len(y_true_n))
        metrics_score['accuracy'] = skmetrics.accuracy_score(y_true_label, y_pred_label)  # 准确率
        metrics_score['precision'] = skmetrics.precision_score(y_true_label, y_pred_label,average='macro')  # 精确率(查准率)
        metrics_score['recall'] = skmetrics.recall_score(y_true_label, y_pred_label,average='macro')  # 召回率(查全率)
        metrics_score['f1'] = skmetrics.f1_score(y_true_label, y_pred_label,average='macro')

        metrics_score_flatten['accuracy_flatten'] = skmetrics.accuracy_score(y_true_n_flatten, y_pred_n_flatten)  # 准确率
        metrics_score_flatten['precision_flatten'] = skmetrics.precision_score(y_true_n_flatten, y_pred_n_flatten,average='macro')  # 精确率(查准率)
        metrics_score_flatten['recall_flatten'] = skmetrics.recall_score(y_true_n_flatten, y_pred_n_flatten,average='macro')  # 召回率(查全率)
        metrics_score_flatten['f1_flatten'] = skmetrics.f1_score(y_true_n_flatten, y_pred_n_flatten,average='macro')

        return [metrics_score,metrics_score_flatten]

    def restore(self,path_model):
        saver = tf.train.Saver()
        sess=tf.InteractiveSession()

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(path_model))
        return sess

    @staticmethod
    def restore_one_model(model_path,name_scope):
        with open(model_path+'/id_info.txt') as f:
            try:
                tags=json.loads(f.readline())
                bz_id=json.loads(f.readline())
            except:
                raise IOError('请检查id_info.txt是否存在或已损坏,确保前两段保存了tags和bz_id。检查路径：'+model_path)
        tf.reset_default_graph()
        model = BilstmCrf(name_scope=name_scope)
        model.category_num = 2 * len(bz_id) + 1
        model.model_path = model_path
        model.build_model()
        sess=model.restore(path_model=model_path)
        return model,sess,tags,bz_id

    def get_feed_dict_usage(self,tree):
        w2v_dict=self.w2v.vocab
        # original_lengths=[len(d[1].split()) for d in one_batch]
        data=[]
        original_lengths = []
        text_rt=[]
        skipped_ind = []
        for t,ind in zip(tree,range(len(tree))):
            text = t['text']
            segments = StanfordNLP.segment(text)
            text_seg = RoughClassifier.segment_sentence(text, copy.deepcopy(segments)).split()
            predit_feature = t['subtitle']['predit_feature']
            text_seg=[i for i in text_seg if i in w2v_dict]
            if text_seg == [] or str(t['other_feature']['tag']).lower()=='table':
                skipped_ind.append(ind)
                text_rt.append(' '.join(text))
                continue
            text_rt.append(' '.join(text_seg))
            original_lengths.append(len(text_seg))
            data.append([text_seg, list(predit_feature)])

        max_len=max(original_lengths)
        x_text=[]
        x_rc=[]
        for d in data:
            a1=self.w2v[d[0]]
            if max_len>len(a1):
                a1_pad=np.zeros([max_len-len(a1),self._input_size])
                a1=np.array(list(a1) + list(a1_pad),dtype=np.float32)
            x_text.append(a1)
            x_rc.append(np.array([np.reshape(d[1], [len(d[1])]) for _ in range(max_len)]))

        x_text=np.array(x_text,dtype=np.float32)
        x_rc=np.array(x_rc,dtype=np.float32)
        original_lengths=np.array(original_lengths,dtype=np.int32)

        feed={self.x_text:x_text,self.x_rc:x_rc,self.original_sequence_lengths:original_lengths,self.keep_prob:1.0}
        return feed,text_rt,skipped_ind

    @staticmethod
    def example_usage(model_path='Classifier_model/BilstmCrf_models/'):
        T=TreeBuilderWithSubtitle()
        tree, _ =T.build_tree_with_subtitle(file_path='train_dataset/pdfs/72d505c2-f4dc-4a77-a9dc-dc9ec3680477_terms.pdf')
        tree_times=BilstmCrf.example_usage_times(tree,model_path)
        tree_ages = BilstmCrf.example_usage_ages(tree,model_path)


        ##todo merge trees

        del T

        return tree_times,tree_ages


    @staticmethod
    def example_usage_times(tree,path_model):
        tf.reset_default_graph()
        name_scope = 'times'
        model_path=path_model + name_scope + '/'
        model_times,sess_times, tags_times, bz_id_times =BilstmCrf.restore_one_model(model_path=model_path,name_scope=name_scope)

        tree_times=BilstmCrf.test_run(model=model_times,sess=sess_times,tags=tags_times,bz_id=bz_id_times,tree=tree)

        del model_times
        return tree_times


    @staticmethod
    def test_run(model,sess,tags,bz_id,tree):
        feed_dict,text,skipped_ind=model.get_feed_dict_usage(tree=tree)

        pred_seqs=sess.run(model.viterbi_sequence,feed_dict=feed_dict)
        pred_seqs=list(pred_seqs)
        if skipped_ind!=[]:
            for ind in skipped_ind:
                pred_seqs.insert(ind,[0])

        tree_rt=[]
        tags_reverse=dict(zip(tags.values(), tags.keys()))
        bz_id_reverse=dict(zip(bz_id.values(), bz_id.keys()))
        for t,txt,p in zip(tree,text,pred_seqs):
            if sum(p[:len(txt.split())])==0:
                tree_rt.append({u'text':txt,u'text_with_number':t[u'text_with_number'],u'depth':t[u'depth'],u'bz':None})
            else:
                bz = []
                for i in range(len(txt.split())):
                    if p[i]!=0:
                        tg=tags_reverse[p[i]]
                        bz.append({u'indice':i,u'tag':[tg.split(u'-')[1],bz_id_reverse[tg.split('-')[1]]]})
                if len(bz)<2:
                    bz[0][u'indice']=[bz[0][u'indice']]
                else:
                    bz_concat=[]
                    for e in bz:
                        if e[u'tag'][1] not in [i[u'tag'][1] for i in bz_concat]:
                            bz_concat.append({u'indice':[e[u'indice']],u'tag':[e[u'tag'][0],e[u'tag'][1]]})
                        else:
                            for index in range(len(bz_concat)):
                                if e[u'tag'][1]==bz_concat[index][u'tag'][1]: break
                            bz_concat[index][u'indice']=bz_concat[index][u'indice']+[e[u'indice']]
                    bz=bz_concat
                tree_rt.append({u'text': txt, u'text_with_number': t[u'text_with_number'], u'depth': t[u'depth'], u'bz': bz})
        return tree_rt

    @staticmethod
    def build_treepkl_data(input_name='Corpus_data/Corpus93.json',output_name='trees93.pkl'):
        print('Doing ',input_name,' ----> ',output_name)
        trees, subtitles=BilstmCrf.build_all_tree(json_path=input_name)
        cPickle.dump(trees, open(output_name, 'wb'))
        print(output_name, ' done')

    @staticmethod
    def auto_neg_num(len_trees):
        '''负样本占总样本数的1/3'''
        return int(len_trees/2)

    @staticmethod
    def build_all_dataset(trees,model_name_spaces,neg_num=-1):

        data={}
        for name in model_name_spaces:
            print('\nbegin build dataset-'+name)
            bz_id=model_name_spaces[name]
            data[name] = BilstmCrfMultitask.build_dataset(trees, bz_id=bz_id, neg_num=neg_num)
        return data

    @staticmethod
    def mix_dataset(data,batch_size):
        train_batchs={}
        for name in data:
            dataone=data[name]
            random.seed(100)
            random.shuffle(dataone)
            train_data_batchs=BilstmCrfMultitask.get_batchs(dataone[:int(len(dataone)*0.9)],batch_size)
            test_data=dataone[int(len(dataone)*0.9):]
            train_batchs[name]={}
            train_batchs[name]['train_batchs']=train_data_batchs
            train_batchs[name]['test']=test_data
        return train_batchs

    def train(self,data,model_name_spaces):

        train_data=[]
        test_data={}
        tags={}
        train_info=MultiTask()
        test_info=MultiTask()
        for name in data:
            tags[name] = BilstmCrfMultitask._chunk_tags(data[name])
            for onebatch in data[name]['train_batchs']:
                train_data.append([name,onebatch])
            test_data[name]=data[name]['test']

            exec 'train_info.' + name+'=MultiTask()'
            exec 'test_info.' + name + '=MultiTask()'
        self.tags=tags
        random.shuffle(train_data)
        print('tags=', tags)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        saver = tf.train.Saver(max_to_keep=self._echo_num)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for echo in range(self._echo_num):
                print('\necho:', echo)

                for name in data.keys():  ##每echo清空
                    exec 'train_info.' + name + '.train_loss' + '=[]'
                    exec 'train_info.' + name + '.train_y_preds' + '=[]'
                    exec 'train_info.' + name + '.train_y_trues' + '=[]'
                    exec 'test_info.' + name + '.test_loss' + '=[]'
                    exec 'test_info.' + name + '.test_y_preds' + '=[]'
                    exec 'test_info.' + name + '.test_y_trues' + '=[]'
                n=0
                for one_batch in train_data:
                    branch_name=one_batch[0]
                    feed_dict = self.get_feed_dict_train(one_batch[1],tags=tags[branch_name],
                                                         branch_name=branch_name, train_test='train')

                    train_op = eval('self.'+branch_name+'.train_op')
                    loss = eval('self.'+branch_name+'.loss')
                    y_labels = eval('self.'+branch_name+'.y_labels')
                    viterbi_sequence = eval('self.'+branch_name+'.viterbi_sequence')

                    _,loss_n,y_true_n,y_pred_n=sess.run([train_op, loss, y_labels, viterbi_sequence], feed_dict=feed_dict)  ##todo 不同name dropout设置不同

                    exec 'train_info.' + branch_name+'.train_loss.append(loss_n)'
                    exec 'train_info.' + branch_name + '.train_y_trues+=list(y_true_n)'
                    exec 'train_info.' + branch_name + '.train_y_preds+=list(y_pred_n)'
                    n+=1
                    print('\r进度%4.2f%%, train_loss: %6.3f' % (n * 100.0 / len(train_data), loss_n), end='')
                print('')

                for name in data.keys():
                    loss_list=eval('train_info.'+name+'.train_loss')
                    print('%s_train_avg_loss: %6.3f' % (name,sum(loss_list) / len(loss_list)),end='   ')

                for name in data.keys():
                    y_trues=eval('train_info.'+name+'.train_y_trues')
                    y_preds=eval('train_info.'+name+'.train_y_preds')
                    m1, m2 = BilstmCrfMultitask.mertics(y_trues, y_preds)
                    print('\n'+name+'_train\'s mertics=', m1)
                    print(name+'_train\'s mertics_flatten=', m2)

                    feed_dict_test = self.get_feed_dict_train(test_data[name],tags=tags[name],branch_name=name,train_test='test')
                    y_labels = eval('self.' + name + '.y_labels')
                    viterbi_sequence = eval('self.' + name + '.viterbi_sequence')

                    y_trues_test, y_preds_test = sess.run([y_labels, viterbi_sequence],feed_dict=feed_dict_test)
                    m1, m2 = BilstmCrfMultitask.mertics(y_trues_test, y_preds_test)
                    print(name+'_test\'s mertics=', m1)
                    print(name+'_test\'s mertics_flatten=', m2)

                saver.save(sess, self.model_path + "/model.ckpt", global_step=echo)
            print('Training Done!')


    @staticmethod
    def train_multitask(pkls,model_name_spaces,neg_num=-1):
        ##neg_num 为 负样本数量，当neg_num==-1时，启动自动选取机制

        trees=[]
        for p in pkls:
            trees_one = cPickle.load(open(p, 'rb'))
            trees+=trees_one

        ###start build model
        tf.reset_default_graph()
        T=BilstmCrfMultitask()
        T.model_path='Classifier_model/BilstmCrf_models/multi-task/'
        T.build_model(model_name_spaces=model_name_spaces,view_tensorboard=True)
        ###end build model

        data=BilstmCrfMultitask.build_all_dataset(trees,model_name_spaces,neg_num=neg_num)
        data_mix=BilstmCrfMultitask.mix_dataset(data,T._batch_size)
        T.print_no_found_words=False
        T.train(data_mix,model_name_spaces)

        with open(T.model_path+'id_info.txt','w') as f:
            f.write(json.dumps(T.tags)+'\n'+json.dumps(model_name_spaces))
        print('done')

    @staticmethod
    def example_train():
        model_name_spaces = {
            'times': {'35': 'YYQSC', '37': 'BXQJSC', '42': 'BXSGTZSX', '46': 'BXJSQHDSX', '47': 'FZQKXBXJSQHDSX',
                      '48': 'JJPFDFSX', '49': 'LXPFDSX', '50': 'KQDBFXXPFDSX', '65': 'HTJCCLSC', '69': 'GSXSHTJCQDSX',
                      '70': 'GSWQJCDHTSXSC', '72': 'SSSXSC', '74': 'KXQSC', '98': 'BDDKZCQX'},
            'ages': {'13': 'TBRDNLXZ', '103': 'BXQJZGNL', '105': 'BBXRNLDX', '106': 'BBXRNLGX', '107': 'XBNLGX'},
            'payments': {'111': 'JFFSYCX', '112': 'JFFSFQPL', '113': 'JFFSFQNX', '114': 'JFFSTZHT', '115': 'JFFSZJ',
                         '116': 'JFFSFQNLSX', '117': 'JFFSBDQBDE', '118': 'JFFSFQ', '119': 'YZFJHTYQJF'}
            }  ##todo 正确的

        print('====== multitask training ======')
        BilstmCrfMultitask.train_multitask(pkls=['trees160.pkl','trees93.pkl'],model_name_spaces=model_name_spaces,neg_num=-1)


if __name__ == '__main__':
    # model_name_spaces = {'times':{'35':'YYQSC','37':'BXQJSC','42':'BXSGTZSX','46':'BXJSQHDSX','47': 'FZQKXBXJSQHDSX',
    #                               '48':'JJPFDFSX','49':'LXPFDSX','50':'KQDBFXXPFDSX','65':'HTJCCLSC','69':'GSXSHTJCQDSX',
    #                               '70': 'GSWQJCDHTSXSC', '72': 'SSSXSC','74': 'KXQSC', '98': 'BDDKZCQX'},
    #                      'ages':{'13':'TBRDNLXZ','103':'BXQJZGNL','105':'BBXRNLDX','106':'BBXRNLGX','107':'XBNLGX'},
    #                      'payments':{'111':'JFFSYCX','112':'JFFSFQPL','113':'JFFSFQNX','114':'JFFSTZHT','115': 'JFFSZJ',
    #                                  '116': 'JFFSFQNLSX', '117': 'JFFSBDQBDE', '118': 'JFFSFQ', '119': 'YZFJHTYQJF'},
    #                      'files': {1:1,2:2},                   ##todo 正确的
    #                      'nouninterpretation': {1:1,2:2},      ##todo 正确的
    #                      'cancelinsurance': {1:1,2:2}}         ##todo 正确的
    #
    # tf.reset_default_graph()
    # T = BilstmCrfMultitask()
    # T.model_path = 'Classifier_model/BilstmCrf_models/' + 'payments' + '/'
    # T.build_model(model_name_spaces=model_name_spaces,view_tensorboard=True)


    BilstmCrfMultitask.example_train()



    print('')



