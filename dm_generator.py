#-*- ecoding:utf-8 -*-
'''生成器
Yield is lazy, it puts off computation. A function with a yield in it doesn't actually execute at all when you call it. 
The iterator object it returns uses magic to maintain the function's internal context.Each time you call next() on the 
iterator (this happens in a for-loop) execution inches forward to the next yield. (return raises StopIteration and ends 
the series.)
'''
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

