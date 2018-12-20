# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 20:44:32 2018

@author: luyimin
"""
import tensorflow as tf

MAX_LEN = 50
SOS_ID = 1     #词汇表中<sos>的ID

def MakeDataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    #根据空格将单词编号切开并放入一个一维向量
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    #将字符串形式的编号转换为整数
    dataset = dataset.map(lambda string: tf.string_to_number(string, tf.int32))
    #统计每个单词的数量，并与句子内容一起放入Dataset
    dataset = dataset.map(lambda x: (x, tf.size(x)))
    return dataset

def MakeSrcTrgDataset(src_path, trg_path, batch_size):
    #分别读取源语言数据和目标语言数据
    src_data = MakeDataset(src_path)
    trg_data = MakeDataset(trg_path)

    dataset = tf.data.Dataset.zip((src_data, trg_data))

    #删除内容为空的句子和长度过长的句子
    def FilterLength(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)

        src_len_ok = tf.logical_and(
            tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN)
        )
        trg_len_ok = tf.logical_and(
            tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN)
        )
        return tf.logical_and(src_len_ok, trg_len_ok)
    dataset = dataset.filter(FilterLength)

    def MakeTrgInput(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
        return ((src_input, src_len), (trg_input, trg_label, trg_len))
    dataset = dataset.map(MakeTrgInput)

    #随机打乱数据
    dataset = dataset.shuffle(10000)

    #规定填充后输出的数据维度
    padded_shapes = (
        (tf.TensorShape([None]), tf.TensorShape([])),
        (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([]))
    )

    batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
    return batched_dataset