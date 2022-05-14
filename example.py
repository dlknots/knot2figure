# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 22:47:34 2021

@author: cyw
"""

from figure2knot2figure import f2k
# from f2kk2f import f2k

import time
'''
需要将图片放在同级文件夹或pth赋完整路径
'''
start = time.time()

# pth="s14.png"
pth="knot01.png"
# pth="knot03.jpg"
# pth="knotX.jpg"
# pth="knot_H04.jpg"
# pth="s5.png"
a=f2k(pth)
# k=a.fun1()#v1.0
# print(len(k[0]))
k=a.fun2()#v2.0 #返回值为参数曲线坐标
print(k.shape)
end = time.time()
#
print(end-start)
