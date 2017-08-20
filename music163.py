# -*- ecoding:utf-8 -*-
'''爬些网易云的评论,可能会做做文本分析
参考: https://www.zhihu.com/question/41505181
     https://www.zhihu.com/question/36081767/answer/140287795
     http://moonlib.com/606.html
'''
import re
import json
import requests
from Crypto.Cipher import AES
# from Crypto import AES
import base64
import codecs
from time import ctime

# 头部信息
headers = {
    'Cookie': "_ntes_nnid=754361b04b121e078dee797cdb30e0fd,1486026808627; _ntes_nuid=754361b04b121e078dee797cdb30e0fd; JSESSIONID-WYYY=yfqt9ofhY%5CIYNkXW71TqY5OtSZyjE%2FoswGgtl4dMv3Oa7%5CQ50T%2FVaee%2FMSsCifHE0TGtRMYhSPpr20i%5CRO%2BO%2B9pbbJnrUvGzkibhNqw3Tlgn%5Coil%2FrW7zFZZWSA3K9gD77MPSVH6fnv5hIT8ms70MNB3CxK5r3ecj3tFMlWFbFOZmGw%5C%3A1490677541180; _iuqxldmzr_=32; vjuids=c8ca7976.15a029d006a.0.51373751e63af8; vjlast=1486102528.1490172479.21; __gads=ID=a9eed5e3cae4d252:T=1486102537:S=ALNI_Mb5XX2vlkjsiU5cIy91-ToUDoFxIw; vinfo_n_f_l_n3=411a2def7f75a62e.1.1.1486349441669.1486349607905.1490173828142; P_INFO=m15527594439@163.com|1489375076|1|study|00&99|null&null&null#hub&420100#10#0#0|155439&1|study_client|15527594439@163.com; NTES_CMT_USER_INFO=84794134%7Cm155****4439%7Chttps%3A%2F%2Fsimg.ws.126.net%2Fe%2Fimg5.cache.netease.com%2Ftie%2Fimages%2Fyun%2Fphoto_default_62.png.39x39.100.jpg%7Cfalse%7CbTE1NTI3NTk0NDM5QDE2My5jb20%3D; usertrack=c+5+hljHgU0T1FDmA66MAg==; Province=027; City=027; _ga=GA1.2.1549851014.1489469781; __utma=94650624.1549851014.1489469781.1490664577.1490672820.8; __utmc=94650624; __utmz=94650624.1490661822.6.2.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; playerid=81568911; __utmb=94650624.23.10.1490672820",
    'Referer': 'http://music.163.com/'
}


# 解密
def AES_encrypt(text, key):
    pad = 16 - len(text) % 16
    text = text + pad * chr(pad)
    encryptor = AES.new(key, AES.MODE_CBC, "0102030405060708")
    encrypt_text = encryptor.encrypt(text)
    encrypt_text = base64.b64encode(encrypt_text)
    return encrypt_text


# 根据歌单或者歌曲的url转成对应api链接,并返回歌曲或者歌单的id
def url_2_api(integer, url):
    id = re.search(r'id=(\d+)', url).group(1)
    if integer == 1:
        res = 'http://music.163.com/weapi/v1/resource/comments/A_PL_0_' + id + '/?csrf_token='
        print 'api链接为:' + res
        return (res, 'sl_' + id)
    elif integer == 2:
        res = 'http://music.163.com/weapi/v1/resource/comments/R_SO_4_' + id + '/?csrf_token='
        print 'api链接为:' + res
        return (res, 's_' + id)


def get_json(url, params, encSecKey):
    data = {
        "params": params,
        "encSecKey": encSecKey,
    }
    proxies = {"http": "http://127.0.0.1:8087", "https": "http://127.0.0.1:8087", }
    response = requests.post(url, headers=headers, data=data,proxies=proxies)
    return response.content


# 两次加密
def get_params(page):
    first_key = '0CoJUm6Qyw8W8jud'
    second_key = 16 * 'F'
    if page == 1:
        first_param = '{rid:"", offset:"0", total:"true", limit:"20", csrf_token:""}'
        h_encText = AES_encrypt(first_param, first_key)
    else:
        offset = str((page - 1) * 20)
        first_param = '{rid:"", offset:"%s", total:"false", limit:"20", csrf_token:""}' % offset
        h_encText = AES_encrypt(first_param, first_key)
    h_encText = AES_encrypt(h_encText, second_key)
    return h_encText


# 获取 encSecKey
def get_encSecKey():
    key = "257348aecb5e556c066de214e531faadd1c55d814f9be95fd06d6bff9f4c7a41f831f6394d5a3fd2e3881736d94a02ca919d952872e7d0a50ebfa1769a7a62d512f5f1ca21aec60bc3819a9c3ffca5eca9a0dba6d6f7249b06f5965ecfff3695b54e1c28f3f624750ed39e7de08fc8493242e26dbc4484a01c76f739e135637c"
    return key


# 获取一定页数的评论
def get_cmts(url_cmt, mypages):
    cmt_list = []
    # 第一行内容
    cmt_list.append(u"用户ID 用户昵称 用户头像地址 评论时间 点赞总数 评论内容\n")
    params = get_params(1)
    encSecKey = get_encSecKey()
    json_text = get_json(url_cmt, params, encSecKey)
    json_dict = json.loads(json_text)
    comments_num = int(json_dict['total'])
    if (comments_num <= 20):
        page = 1
    else:
        page = int(comments_num / 20) + 1
    print("共有%d页评论!" % page)
    print "共:", comments_num, "条!"
    if mypages == 0:
        getcmt_by_page(cmt_list, encSecKey, page, url_cmt)
    elif mypages <= page:
        getcmt_by_page(cmt_list, encSecKey, mypages, url_cmt)
    else:
        getcmt_by_page(cmt_list, encSecKey, page, url_cmt)
    return cmt_list


# 逐页抓取
def getcmt_by_page(cmt_list, encSecKey, mypages, url_cmt):
    for i in range(mypages):
        params = get_params(i + 1)
        encSecKey = get_encSecKey()
        json_text = get_json(url_cmt, params, encSecKey)
        json_dict = json.loads(json_text)
        for item in json_dict['comments']:
            comment = item['content']  # 评论内容
            likedCount = item['likedCount']  # 点赞总数
            comment_time = item['time']  # 评论时间(时间戳)
            userID = item['user']['userId']  # 评论者id
            nickname = item['user']['nickname']  # 昵称
            avatarUrl = item['user']['avatarUrl']  # 头像地址
            comment_info = unicode(userID) + u" " + nickname + u" " + avatarUrl + u" " + unicode(
                comment_time) + u" " + unicode(likedCount) + u" " + comment + u"\n"
            cmt_list.append(comment_info)
        print("第%d页抓取完毕!" % (i + 1))
    return cmt_list


# 将评论写入文本文件
def save_to_file(fn, list):
    with codecs.open(fn, 'a', encoding='utf-8') as f:
        f.writelines(list)
    print("写入文件成功!")


def test():
    print "开始时间:", ctime()
    s_or_sl = int(raw_input("1.歌单链接 2.歌曲链接\n"))
    if s_or_sl == 1:
        sl_url = raw_input("请输入歌单链接:")
        url_cmt, filename = url_2_api(s_or_sl, sl_url)

    elif s_or_sl == 2:
        s_url = raw_input("请输入歌曲链接:")
        url_cmt, filename = url_2_api(s_or_sl, s_url)
    else:
        pass
    pages = int(raw_input("请输入要爬取的页数:(0代表全部)"))

    cmts_list = get_cmts(url_cmt, pages)
    # save_to_file(filename, cmts_list)
    print "结束时间:", ctime()


if __name__ == '__main__':
    test()
