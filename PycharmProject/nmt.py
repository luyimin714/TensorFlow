import sys
import codecs
import tensorflow as tf
from PaddingBatch import MakeSrcTrgDataset

#读取chickpoint路径
CHECKPOINT_PATH_TEST = "../model/seq2seq/seq2seq_ckpt-0"

SRC_TRAIN_DATA = "train.en"
TRG_TRAIN_DATA = "train.zh"
CHECKPOINT_PATH = "../model/seq2seq/seq2seq_ckpt"
HIDDEN_SIZE = 1024
NUM_LAYERS = 2
SRC_VOCAB_SIZE = 10000
TRG_VOCAB_SIZE = 4000
BATCH_SIZE = 100
NUM_EPOCH = 5
KEEP_PROB = 0.8
MAX_GRAD_NORM = 5
SHARE_EMB_AND_SOFTMAX = True

#词汇表中<sos>和<eos>的ID。解码过程中需要用<sos>作为第一部的输入，并将检查是否是<eos>
SOS_ID = 1
EOS_ID = 2

# 词汇表文件
SRC_VOCAB = "en.vocab"
TRG_VOCAB = "zh.vocab"

class NMTModel(object):
    def __init__(self):
        #定义编码器和解码器所使用的LSTM结构
        # self.enc_cell = tf.nn.rnn_cell.MultiRNNCell(    #单向编码器
        #     [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)]
        # )
        self.enc_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)  #前向
        self.enc_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)  #后向
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

    #在forward函数中定义前向计算图
    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        batch_size = tf.shape(src_input)[0]

        #将输入输出单词编号转换为词向量
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

        #在词向量上进行dropout
        src_emb = tf.nn.dropout(src_emb, KEEP_PROB)
        trg_emb = tf.nn.dropout(trg_emb, KEEP_PROB)

        #使用dynamic_rnn构造编码器
        # with tf.variable_scope("encoder"):
        #     enc_outputs, enc_state = tf.nn.dynamic_rnn(self.enc_cell,
        #                                                src_emb,
        #                                                sequence_length=src_size,
        #                                                dtype=tf.float32)

        #使用bidirectional_dynamic_rnn构造双向循环网络
        enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(self.enc_cell_fw,
                                                                 self.enc_cell_bw,
                                                                 src_emb,
                                                                 sequence_length=src_size,
                                                                 dtype=tf.float32)
        #将两个LSTM的输出拼接为1个张量
        enc_outputs = tf.concat([enc_outputs[0], enc_outputs[1]], -1)

        #使用dynamic_rnn构造解码器
        # with tf.variable_scope("decoder"):
        #     dec_outputs, dec_state = tf.nn.dynamic_rnn(self.dec_cell,
        #                                                trg_emb,
        #                                                sequence_length=trg_size,
        #                                                initial_state=enc_state)

        with tf.variable_scope("decoder"):
            #选择注意力权重的计算模型 BahdanauAttention是使用一个隐藏层的前馈神经网络
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(HIDDEN_SIZE,
                                                                       enc_outputs,
                                                                       memory_sequence_length=src_size)
            #将解码器的循环神经网络和注意力机制一起封装成更高层的循环神经网络
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(self.dec_cell,
                                                                 attention_mechanism,
                                                                 attention_layer_size=HIDDEN_SIZE)
            dec_outputs, _ = tf.nn.dynamic_rnn(attention_cell,
                                               trg_emb,
                                               sequence_length=trg_size,
                                               dtype=tf.float32)

        #计算解码器每一步的损失
        output = tf.reshape(dec_outputs, shape=[-1, HIDDEN_SIZE])
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(trg_label, [-1]),
                                                              logits=logits)

        #计算损失时，需要将填充位置的权重设置为0，避免无效位置的预测干扰模型训练
        label_weights = tf.sequence_mask(trg_size,
                                         maxlen=tf.shape(trg_label)[1],
                                         dtype=tf.float32)
        label_weights = tf.reshape(label_weights, [-1])
        cost = tf.reduce_sum(loss * label_weights)
        cost_per_token = cost / tf.reduce_sum(label_weights)

        #定义反向传播操作
        trainable_variables = tf.trainable_variables()

        #控制梯度大小，定义优化方法和训练步骤
        grads = tf.gradients(cost / tf.to_float(batch_size),
                             trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
        return cost_per_token, train_op

    def inference(self, src_input):
        #虽然输入只有一个句子，但dynamic_rnn要求输入时batch的形式，需要将输入句子整理为大小为1的batch
        src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

        #使用dynamic_rnn构造编码器
        with tf.variable_scope("encoder"):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(self.enc_cell,
                                                       src_emb,
                                                       sequence_length=src_size,
                                                       dtype=tf.float32)
        #设置解码的最大步数
        MAX_DEC_LEN = 100
        with tf.variable_scope("decoder/rnn/multi_rnn_cell"):
            #使用一个变长的TensorArray来存储生成的句子
            init_array = tf.TensorArray(dtype=tf.int32,
                                        size=0,
                                        dynamic_size=True,
                                        clear_after_read=False)
            #填入第一个单词<sos>作为解码器输入
            init_array = init_array.write(0, SOS_ID)
            #构建初始的循环状态。循环状态包括循环神经网络的隐藏状态，
            #保存句子的TensorArray，以及记录解码步数的一个step
            init_loop_var = (enc_state, init_array, 0)

            #tf.while_loop的循环条件
            #循环直到解码器输出<eos>，或者达到最大步数
            def continue_loop_condition(state, trg_ids, step):
                return tf.reduce_all( #计算一个张量在维度上元素的“逻辑和”。
                    tf.logical_and(tf.not_equal(trg_ids.read(step), EOS_ID),
                                   tf.less(step, MAX_DEC_LEN - 1))
                )

            def loop_body(state, trg_ids, step):
                #读取最后一步输出的单词，并读取其词向量
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding,
                                                 trg_input)

                #这里不使用dynamic_rnn，而是直接调用dec_cell向前计算一步
                dec_outputs, next_state = self.dec_cell.call(state=state,
                                                             inputs=trg_emb)
                #计算每个可能的输出单词对应的logit，并选取logit最大的值作为这一步的输出
                output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
                logits = (tf.matmul(output, self.softmax_weight) + self.softmax_bias)
                next_id = tf.argmax(logits, axis=1, output_type=tf.int32)
                #将这一步输出的单词写入循环状态的trg_id中
                trg_ids = trg_ids.write(step + 1, next_id[0])
                return next_state, trg_ids, step + 1

            #执行tf.while_loop，返回最终状态
            state, trg_ids, step = tf.while_loop(continue_loop_condition,
                                                 loop_body,
                                                 init_loop_var)
            return trg_ids.stack()


#训练一个epoch，并返回全局步数
def run_epoch(sess, cost_op, traing_op, saver, step):
    while True:
        try:
            cost, _ = sess.run([cost_op, traing_op])
            if step % 10 == 0:
                print("After %d steps, per token cost is %.3f" % (step, cost))
            if step % 200 == 0:
                saver.save(sess, CHECKPOINT_PATH, global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step

def main():
    # with tf.variable_scope("nmt_model", reuse=None):
    #     model = NMTModel()
    #
    # test_sentence = [90, 13, 9, 689, 4, 2]
    #
    # output_op = model.inference(test_sentence)
    #
    # saver = tf.train.Saver()
    # with tf.Session() as sess:
    #     saver.restore(sess, CHECKPOINT_PATH_TEST)
    #     #读取翻译结果
    #     output = sess.run(output_op)
    #     print(output)
    #
    #     # 根据中文词汇表，将翻译结果转换为中文文字。
    #     with codecs.open(TRG_VOCAB, "r", "utf-8") as f_vocab:
    #         trg_vocab = [w.strip() for w in f_vocab.readlines()]
    #     output_text = ''.join([trg_vocab[x] for x in output])
    #
    #     # 输出翻译结果。
    #     print(output_text.encode('utf8').decode(sys.stdout.encoding))

    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    with tf.variable_scope("nmt_model", reuse=None, initializer=initializer):
        train_model = NMTModel()

    #定义输入数据
    data = MakeSrcTrgDataset(SRC_TRAIN_DATA, TRG_TRAIN_DATA, BATCH_SIZE)
    iterator = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()

    #定义前向计算图
    cost_op, train_op = train_model.forward(src, src_size,
                                            trg_input, trg_label, trg_size)

    saver = tf.train.Saver()
    step = 0

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            sess.run(iterator.initializer)
            step = run_epoch(sess, cost_op, train_op, saver, step)

if __name__ == '__main__':
    #tf.reset_default_graph()
    main()