# -*- encoding: utf8 -*-

def get_allIM_word():
    str ="""select * from IM where uid in
    (select uid from
       (select uid, sum(type) as typeSum, count(type) as typeCnt from IM group by uid) as tmp
       where typeSum*2 != typeCnt)
    order by uid asc,time asc"""
    return str