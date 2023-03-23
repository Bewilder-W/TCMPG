import fangzi_strcut

"""
    程序描述：
        从别名替换后的方子.xlsx中筛选得到515个方子存在有范围的剂量
        将结果保存在含有’-‘（范围剂量）的方子.txt中。
"""

# 导入excel表格
path = r'C:\Users\吴杨\Desktop\中药复方-吴杨\TCM\别名替换后的方子.xlsx'
sheet_name = '方子'
sheet, fangzi_object = fangzi_strcut.open_excel(path, sheet_name)


char = '-'
count = 0

xpath = r'C:\Users\吴杨\Desktop\中药复方-吴杨\TCM\含有’-‘（范围剂量）的方子.txt'
with open(xpath, 'w', encoding='utf-8') as file_object:
    file_object.write('含有范围剂量的方子（编号+列F）:')
    file_object.write('\n')
    for fangzi in fangzi_object:
        herb_total = fangzi.herb_total
        if char in herb_total:
            count += 1
            file_object.write(str(fangzi.id) + '\t' + str(fangzi.herb_total))
            file_object.write('\n')
    file_object.write('总计' + str(count) + '个方子')
