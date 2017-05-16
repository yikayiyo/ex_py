# -*- ecoding:utf-8 -*-
from survey import AnonymousSurvey
#定义一个问题,创建表示调查的对象
question = "What language did you first learn to speak?"
mysurvey = AnonymousSurvey(question)
#显示问题并存储答案
mysurvey.show_question()
print ("Enter 'q' at anytime to quit.\n")
while True:
    response = raw_input('language:')
    if response == 'q':
        break
    mysurvey.store_response(response)
#显示调查结果
print("\nThank you to everyone who participated in the survey!")
mysurvey.show_results()
