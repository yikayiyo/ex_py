# -*- ecoding:utf-8 -*-
"""功能:借助栈实现括号匹配"""
from ds_lstack import *


def checkparents(text):
    parents = "()[]{}"
    open_part = "([{"
    opposite_part = {")": "(", "]": "[", "}": "{"}

    def parentheses(text):
        i, text_len = 0, len(text)
        while True:
            while i < text_len and text[i] not in parents:
                i += 1
            if i >= text_len:
                return
            if text[i] in parents:
                yield text[i], i
            i += 1

    st = LStack()
    for pr, i in parentheses(text):
        if pr in open_part:
            st.push(pr)
        elif st.pop() != opposite_part[pr]:
            print "unmatching is found at", i, "for", pr
            return False
    # 最后栈不为空，说明没有全部匹配
    if st.is_empty():
        print "all parenthese are correctly matched."
        return True
    else:
        print "there is sth wrong..."
        return False


if __name__ == '__main__':
    text = "({(1+2)*3*[ssss890f]}"
    # text="({(1+2)*3*[ssss890f]})"
    print checkparents(text)
