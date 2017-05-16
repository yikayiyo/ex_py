# -*- ecoding:utf-8 -*-
import unittest
from survey import AnonymousSurvey

class MyTestCase(unittest.TestCase):
    """针对AnonymousSurvey类的测试"""
    def setUp(self):
        """setup中的变量可在这个类的其它地方使用,避免多次创建"""
        question = "What language did you first learn to speak?"
        self.mysurvey = AnonymousSurvey(question)
        self.responses = ['English', 'Spanish', 'Mandarin']

    def test_store_single_response(self):
        """测试单个答案会被正确存储"""
        # question="What language did you first learn to speak?"
        # mysurvey=AnonymousSurvey(question)
        # mysurvey.store_response('English')
        self.mysurvey.store_response(self.responses[0])
        self.assertIn(self.responses[0],self.mysurvey.responses)

    def test_store_three_responses(self):
        """测试三个答案会被正确存储"""

        for response in self.responses:
            self.mysurvey.store_response(response)
        for response in self.responses:
            self.assertIn(response,self.mysurvey.responses)

if __name__ == '__main__':
    unittest.main()
