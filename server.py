# -*- encoding: utf8 -*-

import sql
import jieba
import jieba.posseg as seg
import numpy as np
jieba.load_userdict("myDict.txt")

# 词典
vocabulary = set([])

def getAimAns(im):
    res = []
    n = len(im)
    for i in range(n):
        if im[i][3] == 0:
            if i+1 >= n or im[i+1][3] == 0:
                res.append(im[i])
    return res


def get_tf(words, idf):
    tf = dict({})
    for word in words:
        tf[word] = tf.setdefault(word, 0) + 1
    return [float(tf[word])*idf[word] for word in words]


def get_idf(ims, ques_cnt):
    df = dict({})
    for i in range(len(ims)):
        words = ims[i][5].split(' ')
        for word in set(words):
            df[word] = df.setdefault(word, 0) + 1
    # convert to idf
    for (key, val) in df.items():
        df[key] = np.log(float(ques_cnt)/val)
    return df


def get_word_id(words):
    word_id = []
    for word in words:
        word_id.append(vocabulary.index(word))
    return word_id


def get_tfidf(ims, ques_cnt):
    tfidf = []
    word_id = []
    idf = get_idf(ims, ques_cnt)
    for im in ims:
        words = im[5].split(' ')
        tfidf.append(get_tf(words, idf))
        word_id.append(get_word_id(words))
    return tfidf, word_id


def get_stop_words():
    stop_words = []
    with open('stop_words') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            stop_words.append(line)
    return stop_words


def getTrainData():
    global vocabulary
    ims = sql.fetchall(sql.get_allIM_word())
    ims = getAimAns(ims)
    stop_words = get_stop_words()

    new_ims = []
    ques_cnt = len(ims)

    for im in ims:
        # print im[1], im[3], im[4]
        cut_words = seg.cut(im[4])
        cut_words = [list(word)[0].encode('utf8') for word in list(cut_words)]
        words = [word for word in cut_words if
                 word not in stop_words and not word.isdigit() and word.strip() != '' and len(
                     word) != 1]
        if len(words) == 0: continue
        vocabulary |= set(words)
        # new_ims.append(list(im).append(words))
        tmp = list(im)
        tmp.append(' '.join(words))
        new_ims.append(tmp)

    vocabulary = list(vocabulary)

    tfidf, word_id = get_tfidf(new_ims, ques_cnt)
    return (tfidf, word_id, new_ims)


def train(tfidf, word_id):
    global vocabulary





if __name__ == "__main__":
    (tfidf, word_id, ims) = getTrainData()
    train(tfidf, word_id)
    print ims[0][4], ims[0][5]

    print "tfidf: ", len(tfidf)
    print 'word_id', len(word_id)
    print "ims:   ", len(ims)

    print 'voca', len(vocabulary)












