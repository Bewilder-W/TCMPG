# TCMPG

## 1 TCMPG-GAE

### 1.1 Introduction

In this work, we used Traditional Chinese Medicine Prescription Generation Graph Auto-Encoder(TCMPG-GAE) to discover the associations between the entire prescriptions and symptoms, and generate prescriptions in an end-to-end manner. 

We constructed our prescriptions dataset for prescription generation through manually collecting and processing. In this paper, we proposed a novel graph auto-encoder model named TCMPG-GAE. Generally speaking, we first proposed novel feature representation methods for prescription and symptom nodes, respectively. Secondly, considering the heterogeneity of prescription and symptom nodes, we designed node-type transformation matrices to project all nodes into the same vector space. Thirdly, we constructed a graph neural networks-based encoder to learn the node embeddings from prescription-symptom bipartite graph. Finally, we designed a prescription generation decoder to simulate the treatment process and reconstruct the representations of prescriptions.

- Under the `data` folder, there are all the data involved in our model. Due to the confidentiality of the data, we do not disclose the source data of TCM prescriptions. 

  > - feature_p.npz: The Feature matrix of all prescriptions with herb dose.
  > - feature_p_none_dose.npz: The Feature matrix of all prescriptions without herb dose.
  > - feature_s.npz: The Feature matrix of all symptoms with word2vec model.
  > - feature_p_none_w2v.npz: The Feature matrix of all symptoms  without word2vec model.
  > - p_s_adj_matrix.npz: Adjacency matrix of the prescription-symptom interaction network.
  > - all_prescription_symptom_pairs.csv: All the prescription-symptom pair of prescription dataset.

  **Note:** If you would like to receive details of this dataset, please contact us. Here is Prof. Du's email:pdu@tju.edu.cn

- Under the `DataProcessed` folder, there are all the codes for TCM data pre-processing. 

  > - add_dosage.py: Increase the dosage of drug composition herbs.
  > - columnFGH_classify_fangzi.py: Classify the data in the three columns of FGH.
  > - unit_conversion.py and unit_convert2g.py: Normalize dose units.
  > - find_nonunitherb.py, find_non_dosageunit.py, and find_with-_dosage.py: Analyze all dose units in the dataset.
  > - extract_all_unit.py: Extract all dose units.
  > - fangzi_struct.py: Construct a class for prescription dataset.
  > - generate_fangzi.py: Generate the format of the dataset for the model experiment.

- All the mode codes are saved in the  `src`  folder.

  > - data_preprocessed.py: Pre-process prescription datasets.
  > - Functions.py: This file include some functions for model.
  > - generate_fangzi.py: Generate the format of the dataset for the model experiment.
  > - layers.py: This file include the code of graph neural network layer.
  > - model.py: This file include the code of TCMPG-GAE model.
  > - predict_prescription.py: This file include the training code and others.
  > - main.py: Run the file to generate the experiment results.



### 1.2 Prescription Dataset

The following information was recorded for each prescription in the dataset: number, prescription name, alias, prescription source, drug composition, drug composition-herb, 【】 is not in the herb list, addition and subtraction, efficacy, indication, preparation method, usage and dosage, contraindication, clinical application, pharmacological effects, various treatises, etc. 

Among them, number, prescription name, prescription source, drug composition, drug composition-herb, 【】 is not in the herb list, efficacy, indication are closely related to the content of this paper and will be the main object of study.

Then, we will list ten examples of prescriptions for you.

|      | Name       |                Source                | Drug composition                                             | Drug composition-herb                                        | 【】 is not in the herb list                                 | Efficacy                   | Indication                                             |
| ---- | ---------- | :----------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------- | ------------------------------------------------------ |
| 1    | 阿胶散     |          《圣惠》卷七十九。          | 阿胶3分（捣碎，炒令黄燥），人参3分（去芦头），黄耆3分（锉），干姜3分（炮裂，锉），当归3分（锉，微炒），熟干地黄3分，芎半两，白茯苓半两，陈橘皮半两（汤浸去白瓤，焙），艾叶半两（微炒），赤石脂1两。 | 阿胶 人参 黄耆 干姜 当归 熟干地黄 芎 白茯苓 陈橘皮 艾叶 赤石脂 | 阿胶 人参 黄芪 干姜 当归 【熟干地黄】 【芎】 茯苓 陈皮 艾叶 赤石脂 | -                          | 产后脓血痢不止，腹内疼痛，不欲饮食，渐加羸弱。         |
| 2    | 阿胶煎丸   |    《幼幼新书》卷一引《灵苑方》。    | 伏道艾（取叶去梗，捣熟，筛去粗皮，只取艾茸，称取2两，米醋煮1伏时，候干研成膏），阿胶3两（炙），糯米（炒）1两，大附子（炮，去皮脐）1两，枳壳（去瓤，麸炒）1两。 | 伏道艾 阿胶 糯米 大附子 枳壳                                 | 【伏道艾】 阿胶 糯米 【大附子】 枳壳                         | 大补益虚损不足，滋助血海。 | 妇人血气久虚，孕胎不成。                               |
| 3    | 阿魏雷丸散 |          《圣惠》卷二十四。          | 阿魏1分（生用），雷丸半两，雄黄半两（细研），朱砂半两（细研），滑石半两，石胆1分（细研），消石半两（细研），白蔹1分，犀角屑半两，牛黄半两（细研），紫石英半两（细研，水飞过），斑蝥20枚（糯米拌炒米黄，去翅足），芫青20枚（糯米拌炒，米黄，去翅足）。 | 阿魏 雷丸 雄黄 朱砂 滑石 石胆 消石 白蔹 犀角屑 牛黄 紫石英 斑蝥 芫青 | 阿魏 雷丸 雄黄 朱砂 滑石 胆矾 【消石】 白蔹 【犀角屑】 牛黄 紫石英 斑蝥 斑蝥 | -                          | 大风出五虫癞，四色可治，唯黑虫不可治，宜先服此。       |
| 4    | 当归补血汤 |           《辨证录》卷三。           | 当归5钱，黄耆1两，荆芥（炒黑）3钱，人参3钱，白术5钱，生地5钱。 | 当归 黄耆 荆芥 人参 白术 生地                                | 当归 黄芪 荆芥 人参 白术 地黄                                | -                          | 血热妄行，九窍流血，气息奄奄，欲卧不欲见日，头晕身困。 |
| 5    | 当归连翘汤 |  《普济方》卷八十三引《卫生家宝》。  | 当归3分，黄连5分，甘草3分，连翘4分，南黄柏5分。              | 当归 黄连 甘草 连翘 南黄柏                                   | 当归 黄连 甘草 连翘 【南黄柏】                               | -                          | 眼白睛红，隐涩难开。                                   |
| 6    | 地黄煎     |          《圣惠》卷二十六。          | 生地黄汁3升，防风2两（去芦头），黄耆2两（锉），鹿角胶2两（捣碎，炒令黄燥），当归2两，丹参2两，桑寄生2两，狗脊2两，牛膝2两，羊髓1升。 | 生地黄汁 防风 黄耆 鹿角胶 当归 丹参 桑寄生 狗脊 牛膝 羊髓    | 【地黄汁】 防风 黄芪 鹿角胶 当归 丹参 桑寄生 狗脊 牛膝 【羊髓】 | 强骨髓，令人充健。         | 骨极。                                                 |
| 7    | 灵砂丹     | 《普济方》卷一四五引《保生回车论》。 | 朱砂（研细、水飞、晒干）1钱，硫黄（研极细）1钱。             | 朱砂 硫黄                                                    | 朱砂 硫黄                                                    | -                          | 伤寒后发疟。                                           |
| 8    | 前胡散     |          《圣惠》卷八十九。          | 前胡半两（去芦头），白茯苓1分，陈橘皮半两（汤浸去白瓤，焙），桂心1分，白术1分，人参1分（去芦头），细辛1分，甘草1分（炙微赤，锉）。 | 前胡 白茯苓 陈橘皮 桂心 白术 人参 细辛 甘草                  | 前胡 茯苓 陈皮 肉桂 白术 人参 细辛 甘草                      | -                          | 小儿肺脏伤冷，鼻流清涕。                               |
| 9    | 温中白术丸 |         《普济方》卷二○六。          | 白术2两半，半夏2两，干姜1两，丁香半两。                      | 白术 半夏 干姜 丁香                                          | 白术 半夏 干姜 丁香                                          | -                          | 胃寒呕哕。                                             |
| 10   | 枳壳汤     |        《圣济总录》卷二十五。        | 枳壳（去瓤，麸炒）1两半，厚朴（去粗皮，生姜汁炙）1两，白术1两，人参1两，赤茯苓（去黑皮）1两。 | 枳壳 厚朴 白术 人参 赤茯苓                                   | 枳壳 厚朴 白术 人参 茯苓                                     | -                          | 伤寒后，心腹气滞胀满，不能饮食。                       |



### 1.3 How to run

The program is written in **Python 3.7** and to run the code we provide, you need to install the `requirements.txt` by inputting the following command in command line mode:

```shell
pip install -r requirements.txt 
```

| Requirements | Release     |
| ------------ | ----------- |
| CUDA         | 9.0         |
| Python       | 3.7.0       |
| mxnet-cu90   | 1.5.0       |
| dgl-cu90     | 0.4.3.post1 |
| matplotlib   | 3.5.1       |
| gensim       | 4.1.2       |
| Django       | 3.2.13      |
| networkx     | 2.6.3       |
| numpy        | 1.21.5      |
| openpyxl     | 3.0.9       |
| pandas       | 1.0.3       |
| scikit-learn | 0.24.2      |
| torch        | 1.1.0       |
| scipy        | 1.7.3       |



And use the below command to run the `main.py`:

```shell
python main.py
```



## 2 TCMPG Platform

To put our model TCMPG-GAE into the application, we constructed a TCMPG web platform where users could obtain the reference prescription by inputting a symptom set. The technologies we adopted included Html, CSS, JavaScript, and Django. We provided four pages for users totally, including `Home`, `Prescription Generation`, `Introduction`, and `Dataset`.  The TCMPG platform is freely accessible at [http://tcm.pufengdu.org](http://tcm.pufengdu.org/).
