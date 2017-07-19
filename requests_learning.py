# -*- encoding:utf-8 -*-
import requests
from PIL import Image
from io import BytesIO
# my_proxy = {'http': '127.0.0.1:8087'}
# my_headers={'User-Agent':'Mozilla/5.0 (iPhone; CPU iPhone OS 9_1 like Mac OS X) AppleWebKit/601.1.46 (KHTML, like Gecko) Version/9.0 Mobile/13B143 Safari/601.1'}
# response = requests.get('https://avatars0.githubusercontent.com/u/22975826?v=4&s=460',headers=my_headers)
#print response.content

#show the photo
# i = Image.open(BytesIO(response.content))
# i.show()

#save the photo
# filename='avatar.jpg'
# with open(filename, 'wb') as fd:
#     for chunk in response.iter_content():
#         fd.write(chunk)

#post form data
# payload = {'key1': 'value1', 'key2': 'value2'}

#useful when the form has multiple elements that use the same key
# payload = (('key1', 'value1'), ('key1', 'value2'))
# r = requests.post('http://httpbin.org/post', data=payload)
# print(r.text)
import json
#
# url = 'https://api.github.com/users'
# payload = {'user_url': 'users/yikayiyo'}
# r = requests.post(url,json=payload)
#
# print r.text

#cookie
# jar = requests.cookies.RequestsCookieJar()
# jar.set('tasty_cookie', 'yum', domain='httpbin.org', path='/cookies')
# jar.set('gross_cookie', 'blech', domain='httpbin.org', path='/elsewhere')
# url = 'http://httpbin.org/cookies'
# r = requests.get(url, cookies=jar)
# print r.text

#redirection and history
#get : redirection by default. GitHub redirects all HTTP requests to HTTPS
# r = requests.get('http://github.com')
#head : no redirection by default.We can enable redirection handling with the allow_redirects parameter
# r = requests.head('http://github.com')
r = requests.head('http://github.com', allow_redirects=True)
print r.url,r.history
