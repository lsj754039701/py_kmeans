# -*- encoding: utf8 -*-

import sql
import jieba
import jieba.posseg as seg
import numpy as np
import kmeans
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

    words_set = set([])

    for im in ims:
        # print im[1], im[3], im[4]
        cut_words = seg.cut(im[4])
        cut_words = [list(word)[0].encode('utf8') for word in list(cut_words)]
        words = [word for word in cut_words if
                 word not in stop_words and not word.isdigit() and word.strip() != '' and len(
                     word) != 1]
        if len(words) == 0: continue
        word_join = ' '.join(words)
        if word_join in words_set:
            continue
        else:
            words_set.add(word_join)
        vocabulary |= set(words)
        tmp = list(im)
        tmp.append(word_join)
        new_ims.append(tmp)

    vocabulary = list(vocabulary)

    tfidf, word_id = get_tfidf(new_ims, ques_cnt)
    return (tfidf, word_id, new_ims)


def train(tfidf, word_id, k):
    global vocabulary
    dataSet = []
    for i in range(len(tfidf)):
        data = [0] * len(vocabulary)
        for j in range(len(tfidf[i])):
            data[word_id[i][j]] = tfidf[i][j]
        dataSet.append(data)
    clust, cent = kmeans.kmeans(dataSet, k)
    return clust


if __name__ == "__main__":
    (tfidf, word_id, ims) = getTrainData()

    k = 6
    # for k in range(3,11):
    #     print "k = ", k
    #     clust = train(tfidf, word_id, k)
    #     # print clust
    #     print sum(clust[:, 1])

    clust = train(tfidf, word_id, k)

    for i in range(k):
        print '*'* 20,'type : %d' % i, '*'*20
        subclust = np.nonzero(clust[:, 0] == i)[0]
        subclust = list(np.array(subclust)[0])
        for x in subclust:
            print ims[x][4]


    print "tfidf: ", len(tfidf)
    print 'word_id', len(word_id)
    print "ims:   ", len(ims)

    print 'voca', len(vocabulary)












