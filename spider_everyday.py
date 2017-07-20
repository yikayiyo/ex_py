#-*- encoding:utf-8 -*-
'''每天都可以爬的'''
import requests,re,datetime
def free_ebook():
    '''图灵特价电子书'''
    url='http://www.ituring.com.cn/'
    try:
        res=requests.get(url)
        if 200==res.status_code:
            html=res.content
    except:
        html=None
    # print html
    if html:
        pattern=re.compile(r'<dt>电子书每周特价</dt>.*?href="(.*?)">(.*?)</a>.*?href="(.*?)">(.*?)</a>.*?href="(.*?)">(.*?)</a>',re.S)
        freebooks=re.search(pattern,html).groups()
        bk1_name=freebooks[1]
        bk1_link=freebooks[0]
        bk2_name=freebooks[3]
        bk2_link = freebooks[2]
        bk3_name=freebooks[5]
        bk3_link = freebooks[4]
        mes=[(bk1_link,bk1_name),(bk2_link,bk2_name),(bk3_link,bk3_name)]
        for item in mes:
            print item
        return mes
    else:
        print 'NO Info'

def write_file(mes):
    today=datetime.date.today()
    filename='mes'+str(today)
    print filename
    with open(filename,'wb') as f:
        f.write('图灵特价电子书\t'+str(today)+'\n')
        for item in mes:
            f.write(str(item[0])+'\t'+str(item[1])+'\n')

if __name__ == '__main__':
    book_mes=free_ebook()
    write_file(book_mes)