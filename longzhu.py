#-*- ecoding:utf-8 -*-
'''龙珠每天10话'''
import webbrowser
file_name='longzhu'
#上次的页码写在文件中
with open(file_name) as f:
    cid=f.readlines()[0]
    # print type(id)
    cid=int(cid)
    print cid
    for i in range(10):
        longzhu_url='http://ac.qq.com/dragonball/v/cid/'+str(cid+i)
        webbrowser.open_new_tab(longzhu_url)
        print longzhu_url
with open(file_name,'w') as f:
    f.writelines(str(cid+10))