#-*- ecoding:utf-8 -*-
'''使用属性函数，改观调用setter和getter时的'''
class TT(object):
    def __init__(self):
        self.__x=None
    def getx(self):
        '''property中没有定义doc时输出：hahahaha'''
        return self.__x
    def setx(self, value):
        self.__x = value
    def delx(self):
        del self.__x
    #x = property(getx, setx, delx,'default doc')
    #x = property(getx, setx, delx)

if __name__ == '__main__':
    t=TT()
    #t.x=5
    #print(t.x)
    t.setx(5)
    print(t.getx())

    #print(TT.x.__doc__)
