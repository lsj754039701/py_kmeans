import MySQLdb

conn = MySQLdb.connect(host='localhost', user='root',passwd='root', db='test',  charset='utf8')
cur = conn.cursor()


def fetchall(str):
    cur.execute(str)
    return cur.fetchall()