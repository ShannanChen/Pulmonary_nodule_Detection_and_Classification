#encoding:utf-8
'''
目的：利用frocwrtdetpepchluna16.py中的getcsv（）函数生成的test候选框
与annotation和exclusive生成的文件进行对比得到fp，fn
'''

#生成对应的annotation和exclude

import  numpy as  np

#any fold csv root

oneseriesuid=''
annotation=''
exclude=''
pre=''



alist=[]

elist=[]
fs = open(oneseriesuid, 'r')
fa = open(annotation, 'r')
fe = open(exclude, 'r')
fre = open(pre, 'r')

for seri in fs.readlines():
    for ann in  fa.readlines():
        if seri==ann[0]:
            alist.append(ann)

    for ex in  fa.readlines():
        if seri==ex[0]:
            elist.append(ex)

for pr in fre.readlines():
    for i in range(0,len(alist))