#-*- ecoding:utf-8 -*-
'''生成器'''
class Bank():
    crisis = False
    def create_atm(self):
        while not self.crisis:
            yield "$100"


def test():
    hsbc = Bank()
    corner_atm = hsbc.create_atm()
    #<generator object create_atm at 0x000000000324D2D0>
    # print corner_atm
    # $100
    # print corner_atm.next()
    # # $100
    # print corner_atm.next()

if __name__ == '__main__':
    test()

