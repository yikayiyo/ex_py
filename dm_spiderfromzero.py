#-*- encoding:utf-8 -*-
import urllib2

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