# TCMPG

## 1 TCMPG-GAE

### 1.1 Introduction

In this work, we used Traditional Chinese Medicine Prescription Generation Graph Auto-Encoder(TCMPG-GAE) to discover the associations between the entire prescriptions and symptoms, and generate prescriptions in an end-to-end manner.

- Under the "data" folder, there are all the data involved in our model. 

- Under the `Experiment result` folder, the `/Case Studies.xlsx` contains the symptom sets of 60 test prescriptions and the results in K = (5, 10, 20). The `/Figures` folder contains the experiment results about Precision, Recall, HR, and NDCG under 5-fold cross-validation.

- All the codes are saved in the  `TCMPG-GAE`  folder.



### 1.2 How to run

The program is written in **Python 3.7** and to run the code we provide, you need to install the `requirements.txt` by inputting the following command in command line mode:

```shell
pip install -r requirements.txt 
```

And use the below command to run the `main.py`:

```shell
python main.py
```



## 2 TCMPG Platform

### 2.1 Introduction

To put our model TCMPG-GAE into the application, we constructed a TCMPG web platform where users could obtain the reference prescription by inputting a symptom set. The technologies we adopted included Html, CSS, JavaScript, and Django. We provided four pages for users totally, including `Home`, `Prescription Generation`, `Introduction`, and `Dataset`.  The TCMPG platform is freely accessible at [http://tcm.pufengdu.org](http://tcm.pufengdu.org/).

All the code files are saved in `/TCMPG Platform/server` folder.

### 2.2 How to run

If you want to run this code in your development environment, you should first install [Visual Studio Code](https://code.visualstudio.com/) or other code editors. And the program is written in **Python 3.8.5** and to run the code we provide, you need to install the `requirements.txt` by inputting the following command in command line mode:

```shell
pip install -r requirements.txt 
```

**Tips:** you should enter the directory `/TCMPG Platform/server` first.



After you have done all the environment configuration, you can use the following command to start the project.

```shell
python manage.py runserver
```

Finally, you can enter the TCMPG platform through [127.0.0.1](http://127.0.0.1:8000/).

