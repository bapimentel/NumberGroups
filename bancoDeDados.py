# -*- coding: utf-8 -*-
"""
Created on Wed Nov 08 11:58:33 2017

@author: Bruno Pimentel
"""

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()

import numpy as np
#import sys
from sklearn import preprocessing

diretorio = 0
iterator_ids_bases = 0
iterator_nome_bases = 0
n_bases = 0
proximo = 0

def carregar_bases(diretory): 
    global diretorio
    diretorio = diretory
    ro.r.assign('diretory', diretory)
    ro.r('''
         library("ParamHelpers")
         library("mlr")    
         library("OpenML")
         setwd(diretory)
         df = listOMLDataSets(number.of.missing.values = 0, number.of.instances = c(100,500), number.of.features = c(2, 200), number.of.classes = c(2, 100), status = "active")
         sub.df = df#df[1:219,]
         tam = nrow(sub.df)
         dir.create(path = "datasource")
         nomesArq = list.files(paste0(diretory,"/datasource")) 
         if(length(nomesArq) < tam){             
             for(i in 1:nrow(sub.df)) {
                     did = sub.df[i, "data.id"]
                     dataset = OpenML::getOMLDataSet(data.id = did)
                     name = sub.df[i, "name"]
                     if(!is.element(name,nomesArq)){
                         #filename = paste0("datasource/", did, "_", name, ".csv")
                         #filename = paste0("datasource/", name, ".csv")
                         filename = paste0("datasource/", name, "_", did, ".csv")
                         #nomesArq = c(nomesArq, paste(did, "_", name, ".csv", sep=""))
                         #nomesArq = c(nomesArq, paste(name, ".csv", sep=""))
                         write.csv(x = dataset$data, file = filename)
                     }
             }
         }
         nomesArq = list.files(paste0(diretory,"/datasource")) 
    ''')
    global iterator_ids_bases
    ids_bases = ro.r('sub.df["data.id"]')[0]
    #iterator_ids_bases = iter(ids_bases)
    iterator_ids_bases = ids_bases
    global iterator_nome_bases
    #nomeBases = ro.r('nomesArq[45:219]')
    nomeBases = ro.r('nomesArq')
    #print(nomeBases)
    #iterator_nome_bases = iter(nomeBases)
    iterator_nome_bases = nomeBases
    global n_bases
    #n_bases = ro.r('length(nomesArq)')[0]
    n_bases = len(nomeBases)

#def get_next_data_OpenML():
#    id_data = next(iterator_ids_bases)
#    ro.r.assign('id_data', id_data)
#    #a = ro.r('id_data')
#    #print(a)
#    ro.r('dataset = OpenML::getOMLDataSet(data.id = id_data)')
#    ro.r('data_base = dataset$data')
#    data_base = ro.r('data_base')
#    LOriginal = ro.r('data_base[,ncol(data_base)]')
#    nClusters = int(max(LOriginal))
#    return data_base, LOriginal, nClusters

def get_next_data():
    #nome = next(iterator_nome_bases)
    global proximo
    nome = iterator_nome_bases[proximo]
    proximo = proximo + 1
    #nome = "iris_61.csv"
    #print(nome)    
    ro.r.assign('nomeArq', nome)
    global diretorio
    ro.r.assign('diretory', diretorio)
    ro.r('data_base = read.csv(file.path(paste0(diretory,"/datasource/"), nomeArq))')
    ro.r('data_base = data_base[,-1]')    
    LOriginal = ro.r('data_base[,ncol(data_base)]')
    nClusters = ro.r('length(unique(data_base[,ncol(data_base)]))')[0]
    ro.r('data_base = data_base[,-ncol(data_base)]')
    data_base = np.transpose(ro.r('data_base'))
    #print(data_base)
    data_base = preprocess_data(data_base)
    #print(data_base)
    #nClusters = int(max(LOriginal))
    if nClusters == len(data_base):
        nClusters = 2
        
    return nome, data_base, LOriginal, nClusters

def get_next_data_file(diretorio):
    global proximo
    ro.r.assign('diretory', diretorio)
    ro.r('''
         nomesArq = list.files(paste0(diretory,"/datasource")) 
         ''')
    nomeBases = ro.r('nomesArq')
    nome = nomeBases[proximo]
    proximo = proximo+1
    ro.r.assign('nomeArq', nome)    
    ro.r('data_base = read.csv(file.path(paste0(diretory,"/datasource/"), nomeArq))')
    ro.r('data_base = data_base[,-1]')    
    LOriginal = ro.r('data_base[,ncol(data_base)]')
    nClusters = ro.r('length(unique(data_base[,ncol(data_base)]))')[0]
    ro.r('data_base = data_base[,-ncol(data_base)]')
    data_base = np.transpose(ro.r('data_base'))
    
    return  nome, data_base, LOriginal, nClusters

def nBases(diretorio):
    ro.r.assign('diretory', diretorio)
    ro.r('''
         nomesArq = list.files(paste0(diretory,"/datasource")) 
         ''')
    nomeBases = ro.r('nomesArq')
    return len(nomeBases)

def preprocess_data(data_base):
    # variaveis categoricas
#    lb_make = LabelEncoder()
#    df = pd.DataFrame(data_base)
#    obj_df = df.select_dtypes(include=['object']).copy()
#    head = obj_df.head()
#    nCat = len(head)
#    for i in range(nCat):
#        head_i = head[i]
#        data_base[head_i] = lb_make.fit_transform(data_base[head_i])
#    print(data_base)
#    sys.exit()
    
    # normalizacao
    #data_base = preprocessing.scale(data_base) # mean=0 sd=1
    min_max_scaler = preprocessing.MinMaxScaler() # [0,1]
    data_base = min_max_scaler.fit_transform(data_base)
    return data_base   

def reset():
    global proximo
    proximo = 0