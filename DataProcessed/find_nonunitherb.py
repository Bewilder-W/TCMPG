import fangzi_strcut

"""
    程序描述：
        对别名替换后的方子.xlsx进行操作，找出列F中存在有剂量无单位的方子
        列举出部分情况，结果有68个方子存在类似情况
"""

# path = r'C:\Users\吴杨\Desktop\中药复方-吴杨\TCM\别名替换后的方子.xlsx'
path = r'C:\Users\吴杨\Desktop\中药复方-吴杨\TCM\1_含有无剂量单位药材的方子-zb(1).xlsx'
sheet_name = '方子'
sheet, fangzi_object = fangzi_strcut.open_excel(path, sheet_name)

digit = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
digitlist = []

for fangzi in fangzi_object:

    herb_total = fangzi_strcut.split_column_FGH(fangzi.herb_total)
    herb_non_unit = fangzi_strcut.split_column_FGH(fangzi.herb_non_unit)

    for i, j in zip(herb_total, herb_non_unit):
        k = str(i.replace(str(j), ''))
        if k != '':
            if k[-1] in digit:
                digitlist.append(fangzi)
                break
# # 写入txt中保存
# xpath = r'C:\Users\吴杨\Desktop\中药复方-吴杨\TCM\有剂量没单位的方子(2).txt'
# with open(xpath, 'w', encoding='utf-8') as file_object:
#     file_object.write('有剂量没单位的方子（编号+列F）:')
#     file_object.write('\n')
#     for fangzi in digitlist:
#         file_object.write(str(fangzi.id) + '\t' + str(fangzi.herb_total))
#         file_object.write('\n')
#     file_object.write('总计' + str(len(digitlist)) + '个方子')


# 测试专用
for fangzi in digitlist:
    print(fangzi.id + ' ' + fangzi.herb_total)
print(len(digitlist))