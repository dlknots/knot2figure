# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 22:47:34 2021

@author: cyw
"""

from figure2knot2figure import f2k

'''
需要将图片放在同级文件夹或pth赋完整路径
'''
filename="figures/"
#需要转化的图片名
pth=filename+"knot01.png"

a=f2k(pth)
a.fun1()#v1.0 #返回值为参数曲线坐标
a.fun2()#v2.0 #返回值为参数曲线坐标


