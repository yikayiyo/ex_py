# -*- encoding:utf-8 -*-
'''每天都可以爬的'''
from Tkinter import *
import requests, re, datetime


def free_ebook():
    '''图灵特价电子书'''
    url = 'http://www.ituring.com.cn/'
    try:
        res = requests.get(url)
        if 200 == res.status_code:
            html = res.content
    except:
        html = None
    # print html
    if html:
        pattern = re.compile(
            r'<dt>电子书每周特价</dt>.*?href="(.*?)">(.*?)</a>.*?href="(.*?)">(.*?)</a>.*?href="(.*?)">(.*?)</a>', re.S)
        freebooks = re.search(pattern, html).groups()
        bk1_link, bk1_name = freebooks[0], freebooks[1]
        bk2_link, bk2_name = freebooks[2], freebooks[3]
        bk3_link, bk3_name = freebooks[4], freebooks[5]
        mes = [[bk1_link,bk1_name], [bk2_link,bk2_name], [bk3_link,bk3_name]]
        return mes
    else:
        print 'NO Info'


def write_file(mes):
    title = '图灵特价电子书'
    today = datetime.date.today()
    filename = 'mes' + str(today)
    # print filename
    with open(filename, 'wb') as f:
        f.write(title + '\t' + str(today) + '\n')
        for k in mes:
            f.write(k[0] + '\t' + k[1] + '\n')


def pop_mes(mes):
    date = datetime.date.today()
    window = Tk()
    title = Label(window, text=str(date) + '特价电子书')
    title.pack()
    for k in mes:
        # lambda event, text=text: self.click_link(event, text)
        print k
        book = Label(window,text=k[1]+k[0])
        book.bind('Enter',openlink(k[0]))
        book.pack()
    window.mainloop()


def openlink(url):
    import webbrowser
    webbrowser.open(url)
    print '点击了' + url


if __name__ == '__main__':
    mes = free_ebook()
    write_file(mes)
    pop_mes(mes)
