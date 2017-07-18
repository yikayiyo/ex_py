#-*- encoding:utf-8 -*-
import urllib
import urllib2

def error_test():
    proxy = urllib2.ProxyHandler({'http': '127.0.0.1:8087'})
    opener = urllib2.build_opener(proxy)
    urllib2.install_opener(opener)
    # path="http://cuiqingcai.com/947.html"
    path="http://ci.com/947ss.html"
    request=urllib2.Request(path)
    try:
        response = urllib2.urlopen(request,timeout=10)
        print response.read()
    except urllib2.HTTPError,e:
        print e

def cookie_test():
    url='http://weibo.com/fav?leftnav=1'
    COOKIE='自己在浏览器找'
    headers={
        'Connection': 'keep-alive', 'cookie': COOKIE,
        'Referer':'http://weibo.com/'
        ,'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
    request=urllib2.Request(url,headers=headers)
    try:
        result=urllib2.urlopen(request)
        print result.read()
    except:
        print 'error'


if __name__ == '__main__':
    #error_test()
    cookie_test()
