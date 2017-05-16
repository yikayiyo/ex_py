#-*- ecoding:utf-8 -*-

class Employee():
    def __init__(self,firstname,lastname,salary):
        self.firstname=firstname
        self.lastname=lastname
        self.salary=salary

    def give_raise(self,custom=''):
        """默认加薪5000"""
        if custom:
            self.salary+=custom
        else:
            self.salary+=5000