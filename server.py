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


# 用编辑距离计算相识性
def is_sim2(ims, words):
    for i in range(len(ims)):
        im = ims[i]
        str1 = im[5].decode('utf8')
        str2 = ' '.join(words).decode('utf8')
        dist = levenshtein(str1, str2)
        if dist <= 3 and dist <= min(len(str1), len(str2))/3:
            return i
    return -1


def get_vec(word1, word2):
    voca = list(set(word1) | set(word2))
    vec1 = [0] * len(voca); vec2 = [0] * len(voca)
    for w in word1:
        vec1[voca.index(w)] += 1
    for w in word2:
        vec2[voca.index(w)] += 1
    return vec1, vec2


# 用余弦距离计算相识性
def is_sim(ims, words):
    for i in range(len(ims)):
        im = ims[i]
        vec1, vec2 = get_vec(im[5].split(' '), words)
        dist = kmeans.distCos(np.mat(vec1), np.mat(vec2))
        if dist > 0.85:
            # str1 = im[5].decode('utf8')
            # str2 = ' '.join(words).decode('utf8')
            # print str1
            # print str2
            # print dist
            # print '*'*30
            return i
    return -1


def levenshtein(str1, str2):
    n1 = len(str1) + 1; n2 = len(str2) + 1
    dp = [range(i, n2+i) for i in range(n1) ]
    for i in range(1, n1):
        for j in range(1, n2):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+1)
    return dp[n1-1][n2-1]


def getTrainData():
    global vocabulary
    ims = sql.fetchall(sql.get_allIM_word())
    ims = getAimAns(ims)
    stop_words = get_stop_words()
    new_ims = []

    for im in ims:
        # print im[1], im[3], im[4]
        cut_words = seg.cut(im[4])
        cut_words = [str(word).split('/')[0] for word in list(cut_words)]

        words = [word for word in cut_words if
                 word not in stop_words and not word.isdigit() and word.strip() != '' ]
        if len(words) <= 1: continue
        # 若有相识的问题，保留较长的那个问题
        sim1 = is_sim(new_ims, words)
        sim2 = is_sim2(new_ims, words)
        if sim1 >= 0 or sim2 >= 0:
            sim_idx = max(sim1, sim2)
            if len(new_ims[sim_idx][4]) < len(im[4]):
                print sim1, sim2
                print im[4], new_ims[sim_idx][4]
                new_ims.pop(sim_idx)
            else: continue

        vocabulary |= set(words)
        word_join = ' '.join(words)
        tmp = list(im)
        tmp.append(word_join)
        new_ims.append(tmp)

    vocabulary = list(vocabulary)
    ques_cnt = len(new_ims)
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

    # for k in range(3,11):
    #     print "k = ", k
    #     clust = train(tfidf, word_id, k)
    #     # print clust
    #     print sum(clust[:, 1])
    k = 5
    clust = train(tfidf, word_id, k)

    f = open("clust.txt", 'w')
    for i in range(k):
        print '*'* 20,'type : %d' % i, '*'*20
        f.write('*'* 20  + 'type : %d' % i + '*'*20 + '\n')
        subclust = np.nonzero(clust[:, 0] == i)[0]
        subclust = list(np.array(subclust)[0])
        for x in subclust:
            print ims[x][4]
            f.write(ims[x][4].encode('utf8') + '\n')
        f.write('\n')
    f.close()


    print "tfidf: ", len(tfidf)
    print 'word_id', len(word_id)
    print "ims:   ", len(ims)
















