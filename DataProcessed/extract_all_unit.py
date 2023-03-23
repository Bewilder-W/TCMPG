import fangzi_strcut
import re
"""
    程序描述：
        对"别名替换后的方子.xlsx"进行操作，操作对象为列F和列G。通过观察所有数据发现，大部分
        列F和列G都是对应药材名相同的，所以直接将列F的每一项剔除列G对应的那一项，可以
        得到所有的剂量单位，再通过re包删除数字剂量部分，得到单位，最后去重保存在all_unit
        列表中。但分析结果发现几个问题。
        问题：（1）部分列F中有些药材连在一起，但是列G没有对应好药材名。这部分方子应该
              需要删除或者修改。
              例如：77936	人参芦1钱桔梗2钱 牙皂5分	人参芦 牙皂
             （2）部分列F和列G中对应的药材名不相同
              例如  79313	皂荚嫩芽不限多少	皂荚
             （3）按照匹配数字得到单位的情况下，有些单位是 1两2分， 1斤3两 等等，处理后
              得到的单位是两分，斤两等
             （4）还有一些单位很奇怪
              例如  76251	茯苓1我	82298	胡粉7棋子	   82466	猪血半蛤蜊壳
             （5）还有一些不是单位
              例如： 不拘多少，随意用等等
    程序输入：path, sheet_name
    程序输出：all_unit列表
"""
# path = r'C:\Users\吴杨\Desktop\中药复方-吴杨\TCM\别名替换后的方子.xlsx'
# path = r'C:\Users\吴杨\Desktop\中药复方-吴杨\TCM\别名替换后的方子1-2.xlsx'
path = r'C:\Users\吴杨\Desktop\中药复方-吴杨\TCM\别名替换后的方子1-7.xlsx'
sheet_name = '方子'

# 得到表格对象和所有方子的对象列表
sheet, fangzi_object = fangzi_strcut.open_excel(path, sheet_name)
# print(fangzi_object[1].herb_total[0], fangzi_object[1].herb_non_unit)

# 字典保存所有单位，去重
all_unit = {}
# 字典保存单位和对应的方子编号
unit_id = {}
for fangzi in fangzi_object:

    # 别名替换后的方子.xlsx中的列FGH已经是规格化后的结果，类型为str
    # 经过split_column_FGH()操作后，结果类型为list
    herb_total = fangzi_strcut.split_column_FGH(fangzi.herb_total)
    herb_non_unit = fangzi_strcut.split_column_FGH(fangzi.herb_non_unit)

    for i, j in zip(herb_total, herb_non_unit):
        i = str(i.replace(str(j), ''))
        i = re.sub('\d+', '', i)
        i = re.sub('\.', '', i)
        if len(i) != 0:
            if i in all_unit.keys():
                all_unit[i] = all_unit[i] + 1
                unit_id[i].append(fangzi.id)
            else:
                all_unit[i] = 1
                unit_id[i] = []
                unit_id[i].append(fangzi.id)

# filepath = r'C:\Users\吴杨\Desktop\中药复方-吴杨\TCM\药材的单位.txt'
# filepath = r'C:\Users\吴杨\Desktop\中药复方-吴杨\TCM\药材的单位1-1.txt'
# filepath = r'C:\Users\吴杨\Desktop\中药复方-吴杨\TCM\药材的单位1-2.txt'
# filepath = r'C:\Users\吴杨\Desktop\中药复方-吴杨\TCM\药材的单位1-3.txt'
# filepath = r'C:\Users\吴杨\Desktop\中药复方-吴杨\TCM\药材的单位1-4.txt'
# filepath = r'C:\Users\吴杨\Desktop\中药复方-吴杨\TCM\药材的单位1-5.txt'
filepath = r'C:\Users\吴杨\Desktop\中药复方-吴杨\TCM\药材的单位1-7.txt'

with open(filepath, 'w', encoding='utf-8') as file_object:
    file_object.write('所有药材的单位集合如下(去重后) 单位+个数+部分方子编号:')
    file_object.write('\n')
    file_object.write('无单位')
    file_object.write('\n')
    k = 0
    for i in all_unit.keys():
        # 第一行是药物组成，不用写出
        if k == 0:
            k = 1
            continue
        file_object.write(str(i) + '\t' + str(all_unit[i]))
        count = 0
        for j in unit_id[i]:
            file_object.write('\t' + str(j))
            count = count + 1
            if count > 50:
                break
        file_object.write('\n')
        # file_object.write('\n')
    file_object.write('总共有' + str(len(all_unit)) + '种单位')