import tensorflow as tf

SRC_TRAIN_DATA = "train.en"
TRG_TRAIN_DATA = "train.zh"
CHECKPOINT_PATH = "./model/seq2seq/seq2seq_ckpt"
HIDDEN_SIZE = 1024
NUM_LAYERS = 2
SRC_VOCAB_SIZE = 10000
TRG_VOCAB_SIZE = 4000
BATCH_SIZE = 100
NUM_EPOCH = 5
KEEP_PROB = 0.8
MAX_GRAD_NORM = 5
SHARE_EMB_AND_SOFTMAX = True

class NMTModel(object):
    def __init__(self):
        #定义编码器和解码器所使用的LSTM结构
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)]
        )
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)]
        )

        #为源语言和目标语言定义词嵌入
        self.src_embedding = tf.get_variable(
            "src_emb", shape=[SRC_VOCAB_SIZE, HIDDEN_SIZE]
        )
        self.trg_embedding = tf.get_variable(
            "trg_emb", shape=[TRG_VOCAB_SIZE, HIDDEN_SIZE]
        )

        #定义softmax层变量
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable("weight", shape=[HIDDEN_SIZE, TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable("softmax_bias", [TRG_VOCAB_SIZE])

