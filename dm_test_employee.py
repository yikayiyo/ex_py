#-*- ecoding:utf-8 -*-
import unittest
from dm_Employee import Employee

class MyTestCase(unittest.TestCase):
    """dm_Employee测试类"""
    def setUp(self):
        self.employee=Employee('Lei','Li',5000)

    def test_give_default_raise(self):
        self.employee.give_raise()
        self.assertEqual(10000,self.employee.salary)

    def test_give_custom_raise(self):
        self.employee.give_raise(3000)
        self.assertEqual(8000,self.employee.salary)

if __name__ == '__main__':
    unittest.main()
