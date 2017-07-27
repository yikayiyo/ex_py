#-*- encoding:utf-8 -*-
import requests
import urllib
from pyquery import PyQuery as pq
'''
某频道的最近10个视频
练习使用pyquery
'''
chanel_id='UCmyNVGDx2rrbeYj7U0nUmew'

ytb_url = 'https://www.youtube.com/channel/'+chanel_id+'/videos'
headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36',
    'Host':'www.youtube.com',
}

try:
    res=requests.get(ytb_url, headers=headers,verify=False)
    if 200==res.status_code:
        html=pq(res.text)
        if len(html):
            ##channels-browse-content-grid
            videos=html('#channels-browse-content-grid')
            videos=videos.children().items()
            cnt=-1
            for video in videos:
                cnt+=1
                if cnt==10:
                    break
                print video('.yt-lockup-content h3 a').text()
                print video('.yt-lockup-content h3 span').text()[:-1]
                print 'https://www.youtube.com/'+video('.yt-lockup-content h3 a').attr("href")

        else:
            print None
    else:
        print '300'
except Exception as e:
    print e