# -*- encoding:utf-8 -*-
import json
import os
import requests
from urllib import urlencode
from multiprocessing import Pool
from time import ctime
from hashlib import md5
'''
分析XHR，爬取今日头条街拍图片
'''

def get_page_index(offset, keyword):
    # 根据offset,keyword 发送http请求，返回数据类型为json
    data = {
        'offset': offset,
        'format': 'json',
        'keyword': keyword,
        'autoload': 'true',
        'count': 20,
        'cur_tab': 1,
    }
    #get请求用urlencode来处理data
    url = 'https://www.toutiao.com/search_content/?' + urlencode(data)
    proxy = {"http": "http://127.0.0.1:8087"}
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, proxies=proxy)
        if 200 == response.status_code:
            return response.text
        return None
    except:
        print 'false request.'
        return None

#处理json数据，提取街拍项的title和images，提取images包含的image的url并下载
def parse_page_index(html):
    try:
        data = json.loads(html)
        # print data
        if data and 'data' in data.keys():
            for item in data.get('data'):
                title = item.get('title')
                image_detail = item.get('image_detail')
                print title, item.get('article_url')
                for image in image_detail:
                    url = image.get('url')
                    download_image(title, url)
    except:
        pass

def download_image(title, url):
    print 'downloading...' + url
    proxy = {"http": "http://127.0.0.1:8087"}
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers, proxies=proxy, timeout=5)
        if 200 == response.status_code:
            #不含title字段的
            if not title:
                title = 'mytitle'
            #保存数据
            save_image(title, response.content)
        return None
    except requests.exceptions.Timeout:
        print "Timeout occurred"
        return None


def save_image(title, content):
    title = title.encode('utf-8').strip()
    if len(title)>60:
        #防止文件名过长带来的错误
        title=title[:60]+'..'
    path = '/home/gao/图片/' + title
    #对每一个项目，不存在则创建对应文件夹
    if not os.path.exists(path):
        os.mkdir(path)
    #保证文件唯一，md5
    file_path = '{0}/{1}.{2}'.format(path, md5(content).hexdigest(), 'jpg')
    if not os.path.exists(file_path):
        with open(file_path, 'wb+') as f:
            f.write(content)


def main(offset):
    data = get_page_index(offset, '街拍')
    if data:
        parse_page_index(data)


if __name__ == '__main__':

    print ctime()+'\tStart!'
    groups = [x * 20 for x in range(2)]
    pool = Pool()
    pool.map(main, groups)
    print ctime()+'\tOver!'