#-*- encoding:utf-8 -*-

import requests
from lxml import etree
import re
# from selenium import webdriver
import sys
reload(sys)
sys.setdefaultencoding('utf8')

class Paper():

    def __init__(self,title=None,authors=None,tags=None):
        self.title = title
        self.authors = authors[:]
        # self.abstract = abstract
        self.tags = tags[:]



    def __str__(self):
        return '标题：{}\n作者：{}\n标签：{}'.format(self.title,','.join(self.authors),','.join(self.tags))
def getPage(url):
    res = ''
    headers = {
        'Content-Type':'text/html;charset=utf-8',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/62.0.3202.62 Chrome/62.0.3202.62 Safari/537.36',
        'Host': 'www.baidu.com',
        'Cookie': 'BAIDUID=42708892E9F780037C33C84232E0DCC0:FG=1; BIDUPSID=42708892E9F780037C33C84232E0DCC0; PSTM=1509529366; BD_UPN=123353; BD_HOME=0; Hm_lvt_f28578486a5410f35e6fbd0da5361e5f=1509529368; Hm_lpvt_f28578486a5410f35e6fbd0da5361e5f=1509529375; tipVisible=no; neverShow=no; BDRCVFR[w2jhEs_Zudc]=mk3SLVN4HKm; BD_CK_SAM=1; PSINO=2; BDSVRTM=584; H_PS_PSSID=; Hm_lvt_43115ae30293b511088d3cbe41ec099c=1509529390; Hm_lpvt_43115ae30293b511088d3cbe41ec099c=1509548711'}
    try:
        response = requests.get(url,headers=headers)
        res = response.text
    except Exception as e:
        print(e)

    return res

# 获取每一页的项目
def getPapers(text):
    res=[]
    # print text
    tree = etree.HTML(text)
    # print text
    papers = tree.xpath(u'//*[@id="bdxs_result_lists"]/div[@class="result sc_default_result xpath-log"]')
    # print(len(papers))
    for paper in papers:
        title = paper.xpath(u'./div[1]/h3/a')[0].xpath('string(.)').replace('<em>','').replace('</em>','')
        authors = paper.xpath(u'./div[1]/div[1]/span[1]/a/text()')
        # print authors
        # abstract = paper.xpath(u'./div[1]/div[2]')[0].xpath('string(.)').replace('<em>','').replace(r'\s','')
        tags = paper.xpath(u'./div[2]/div[1]/a/text()')
        p = Paper(title,authors,tags)
        # print p
        # print '-----------'
        res.append(p)
    return res


def main():
    # SERVICE_ARGS = ['--load-images=false', '--disk-cache=true']
    # browser = webdriver.PhantomJS(service_args=SERVICE_ARGS)
    data = []
    kwd = '机器学习'
    page_nums = 10
    count = 10

    for i in range(page_nums):
        print '下载第{}页'.format(i+1)
        count *= i
        url_base = 'https://www.baidu.com/s?wd={kwd}&pn={count}&tn=SE_baiduxueshu_c1gjeupa&ie=utf-8&sc_f_para=sc_tasktype%3D%7BfirstSimpleSearch%7D&sc_hit=1'.format(
            kwd=kwd, count=count)
        page_text = getPage(url_base)
        if page_text:
            # data = getPapers(page_text)
            data+=getPapers(page_text)
            print '第{}页处理完毕'.format(i+1)
        else:
            print('no content')

    print('下载数目：{}'.format(len(data)))

    # for item in data:
    #     print item
    #     print '-----------'

if __name__ == '__main__':
    main()