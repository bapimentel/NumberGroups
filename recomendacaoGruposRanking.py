# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 12:26:55 2019

@author: Bruno Pimentel
"""


import bancoDeDados

import sys

import numpy as np
from datetime import datetime
import time
import math
import random
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import rpy2.robjects as ro
ro.__path__
from rpy2.robjects import pandas2ri
pandas2ri.activate()

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.linear_model import SGDClassifier, SGDRegressor, Perceptron, HuberRegressor
from sklearn.neural_network import MLPRegressor


def getNumGruposBases(diretory, exceto, filtro_ng):
#    bancoDeDados.carregar_bases(diretory)
#    nBases = bancoDeDados.n_bases
    nBases = bancoDeDados.nBases(diretorio)
#    bancoDeDados.reset()  
    numGrupos = {}
    todosDados = {}
    for indice_base in range(nBases): 
#        nome, dados, LOriginal, nClusters = bancoDeDados.get_next_data()
        nome, dados, LOriginal, nClusters = bancoDeDados.get_next_data_file(diretorio)
#        print nome
        nomeOriginal = nome[:-4]
        if nomeOriginal not in exceto and (filtro_ng == -1 or nClusters == filtro_ng):
            numGrupos[nomeOriginal] = nClusters
            todosDados[nomeOriginal] = dados
    return numGrupos, todosDados

def getNomeBases(nomesAll, fold_i):
    nomes = []
    for a in nomesAll:
        if a not in fold_i:
            nomes.append(a)
    return nomes

def getNumGruposBases2(numGruposAll, fold_i):
    num = []
    nomes = numGruposAll.keys()
    for a in nomes:
        if a not in fold_i:
            num.append(numGruposAll[a])
    return num
    

def carregar_metafeatures(diretorio, tipo, bases_teste, selecionadas, bases_todas):
    metafeatures = [] # Agrupada por tipo. Cada tipo tem uma matriz (b X f), 
    #onde b eh o numero de bases e f o numero de features (obs.: cada tipo tem um f diferente)
    ro.r.assign('tipo', tipo)
    ro.r.assign('diretorio', diretorio+"/metafeatures")
    ro.r('nomesArq = list.files(diretorio)')
    nomeBases = ro.r('nomesArq')
    #print(nomeBases)
    for nome in nomeBases:
        #print(nome)
        ro.r.assign('nome', nome+"") 
        ro.r('mf = read.csv(file=paste0(diretorio,"/",nome,"/",tipo,".csv"), header = FALSE, sep = ",")')
        mf = ro.r('as.vector(as.matrix(mf))')
        if not nome in bases_teste and nome in bases_todas:
            mf_selecao = []
            if len(selecionadas)!=0:
                for i in range(len(selecionadas)):
                    mf_selecao.append(mf[selecionadas[i]])
            else:
                mf_selecao = mf
            mf = mf_selecao
            metafeatures.append([nome, mf])
    return metafeatures

def carregar_metafeatures_fold(mf, fold):
    n = len(mf)
    mf_result = []
    for i in range(n):
        if mf[i][0] not in fold:
            mf_result.append([mf[i][0], mf[i][1]])
    return mf_result

def pegar_metafeatures_nome(metafeatures, nome):
    mf = -1
    tam = len(metafeatures)
    for i in xrange(tam):
        mf = metafeatures[i]
        if(mf[0]==nome):
            break
    return mf[1]

def recomendarNGrupos(baseDados, metodo, indice, min_nc, max_nc): 
    ro.r.assign('x', baseDados)
    ro.r.assign('metodo', metodo)
    ro.r.assign('indice', indice)
    ro.r.assign('min_nc', min_nc)
    ro.r.assign('max_nc', max_nc)
#    print indice
    ro.r('''         
        library(NbClust) 
        res<-NbClust(x, distance = "euclidean", method = metodo, index = indice, min.nc = min_nc, max.nc = max_nc)
        a = res$Best.nc[1] 
        b = array(res$All.index)       
         ''')
    ng = ro.r('a')[0]
    index = list(ro.r('b'))
    return ng, index

def recomendarNGrupos_Elbow(baseDados, min_nc, max_nc):
    wcss = []
#    min_nc = 1
    for i in range(min_nc, max_nc+1):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(baseDados)
        wcss.append(kmeans.inertia_)
    distances = []
    p1x = min_nc
    p1y = wcss[0]
    p2x = max_nc
    p2y = wcss[max_nc-min_nc]
    for i in range(max_nc-1):
        x = min_nc + i
        y = wcss[i]
        x_diff = p2x - p1x
        y_diff = p2y - p1y
        num = abs(y_diff*x - x_diff*y + p2x*p1y - p2y*p1x)
        distances.append(num)    
    ng = int(distances.index(max(distances))+min_nc)
    index = wcss
#    print ng, index
    return ng, index

def recomendarNGrupos_Information(baseDados, metodo, min_nc, max_nc):
    valores = []
#    min_nc = 1
    for i in range(min_nc, max_nc+1):
        EM = GaussianMixture(n_components=i)
        EM.fit(baseDados)
        valor = 0
        if metodo == 'bic':
            valor = EM.bic(baseDados)
        elif metodo == 'aic':
            valor = EM.aic(baseDados)
        valores.append(valor)
    ng = int(valores.index(min(valores))+min_nc)
    index = valores
#    print  ng, index
    return ng, index
    

def pegar_nome_bases(metafeatures):
    nomes = []
    tam = len(metafeatures)
    for i in range(tam):
        mf = metafeatures[i]
        nomes.append(mf[0])
    return nomes 

def criar_k_fold(k, vetor):
    folds = []
    ro.r.assign('diretorio', diretorio+"/datasource")
    ro.r('nomesArq = list.files(diretorio)')
    nomeBases = vetor#ro.r('nomesArq')
    n = float(len(nomeBases))/float(k)
    n = math.ceil(n)
    #print(n)
    n = int(n)
    for i in range(k-1):
        lista = np.random.choice(range(len(nomeBases)), size=n, replace=False)
        fold = [nomeBases[i] for i in lista]
        folds.append(fold)
        nomeBases = np.delete(nomeBases, lista)
    folds.append(nomeBases)
    return folds

def pegar_X(metafeatures):
    X = []
    n = len(metafeatures)
    for i in range(n):
        mf = np.asanyarray(metafeatures[i][1])
        X.append(mf)  
    scaler = MinMaxScaler()
#    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

#Com oversampling 
def pegar_X_Y_balanceado(metafeatures, Ys):
    X = []
    Y = []
    dicNG = {}
    for i in range(len(metafeatures)):
        dicNG[metafeatures[i][0]] = Ys[metafeatures[i][0]]  
#    print dicNG
    maisFrequente = max(set(dicNG.values()), key = dicNG.values().count)
    count = Counter(dicNG.values())
    maiorFrequencia = count[maisFrequente]
    count_freqs = {}
#    print count, maisFrequente, maiorFrequencia
#    sys.exit(1)
    for i in count:
        count_freqs[i] = 0
    for i in range(len(metafeatures)):
        mf = np.asanyarray(metafeatures[i][1])
        X.append(mf) 
        Y.append(dicNG[metafeatures[i][0]])
        count_freqs[dicNG[metafeatures[i][0]]] = count_freqs[dicNG[metafeatures[i][0]]] + 1
#    print count_freqs
#    sys.exit(1)
    grupos = {}
    for i in count:
        grupo_i = []        
        for ii in range(len(metafeatures)):
            if dicNG[metafeatures[ii][0]] == i:
                grupo_i.append(ii)
        grupos[i] = grupo_i
#    print grupos
#    sys.exit(1)
    for i in count:
        while count_freqs[i] < maiorFrequencia:
#            print count_freqs
            indice_mf = random.choice(grupos[i])
            mf = np.asanyarray(metafeatures[indice_mf][1])
            X.append(mf) 
            Y.append(dicNG[metafeatures[indice_mf][0]])
            count_freqs[dicNG[metafeatures[indice_mf][0]]] = count_freqs[dicNG[metafeatures[indice_mf][0]]] + 1
#    sys.exit(1)
            
#    print len(X), len(Y)
            
#    scaler = MinMaxScaler()
#    scaler = StandardScaler()
#    X = scaler.fit_transform(X)
            
    maximos, minimos = getMaxMin(X)
    n = len(X)
    p = len(X[0])
    Xnovo = []
    for a in range(n):
        x = []
        for b in range(p):
            x.append(min(1,(X[a][b]-minimos[b])/(maximos[b]-minimos[b])))
        Xnovo.append(x)
    X = Xnovo
            
    
#    print X
    
    return X, Y

def getMaxMin(dados):
#    T = []
    n = len(dados)
    p = len(dados[0])
    minimos =  []
    maximos = []
    for i in range(p):
        a = [dados[x][i] for x in range(n)]
        a_min = min(a)
        a_max = max(a)
        minimos.append(a_min)
        maximos.append(a_max)
    return maximos, minimos

def pegar_Y(todosDados, min_nc, max_nc, nome_index):
    Y = {}
    tempos_Ys = {}
    indices = {}
    for key in todosDados:
#        print key
        start_time = time.time()
        
        #Elbow method
        if nome_index == 'elbow':
            ng_indice, index = recomendarNGrupos_Elbow(todosDados[key], min_nc, max_nc)
        #Information method
        elif nome_index == 'bic' or nome_index == 'aic':
            ng_indice, index = recomendarNGrupos_Information(todosDados[key], nome_index, min_nc, max_nc)
        #Internal valuation index
        else:        
            ng_indice, index = recomendarNGrupos(todosDados[key], "kmeans", nome_index, min_nc, max_nc)
#        ng_indice, index = recomendarNGrupos(todosDados[key], "average", "db", min_nc, max_nc)
#        ng_indice, index = recomendarNGrupos(todosDados[key], "average", "dunn", min_nc, max_nc)
#        ng_indice, index = recomendarNGrupos(todosDados[key], "average", "kl", min_nc, max_nc)
#        ng_indice, index = recomendarNGrupos(todosDados[key], "average", "sdindex", min_nc, max_nc)
#        ng_indice, index = recomendarNGrupos(todosDados[key], "average", "silhouette", min_nc, max_nc)
#        ng_indice = recomendarNGrupos(todosDados[key], "kmeans", "gap", min_nc, max_nc)
#        ng_indice = recomendarNGrupos(todosDados[key], "kmeans", "pseudot2", min_nc, max_nc)
        
        #Majoritario
#        ng_indice1, index1 = recomendarNGrupos(todosDados[key], "average", "silhouette", min_nc, max_nc)
#        ng_indice2, index2 = recomendarNGrupos(todosDados[key], "average", "db", min_nc, max_nc)
#        ng_indice3, index3 = recomendarNGrupos(todosDados[key], "average", "dunn", min_nc, max_nc)
#        ng_indice4, index4 = recomendarNGrupos(todosDados[key], "average", "kl", min_nc, max_nc)
#        ng_indice5, index5 = recomendarNGrupos(todosDados[key], "average", "sdindex", min_nc, max_nc) 
#        array = [ng_indice1, ng_indice2, ng_indice3, ng_indice4, ng_indice5]
#        array2 = [index1, index2, index3, index4, index5]
##        print array
##        ng_indice = np.random.randint(2, size=1)[0]
#        ng_indice = max(set(array), key=array.count)
##        print ng_indice
#        index = np.mean(array2)        
#        print index       
        
        
        delta_time = time.time() - start_time
        Y[key] = ng_indice
#        print len(Y),
        
        tempos_Ys[key] = delta_time
        indices[key] = index
            
#    Y = numGrupos.values()   
    return Y, tempos_Ys, indices



def calcular_NG(mf_teste, clf, X, Y, r_ideal, indices):
    start_time = time.time()
#    print len(X), len(Y)
    clf.fit(X, Y) 
#    print len(X[0])
#    print len(Y)
#    print [list(mf_teste)]
    ng = clf.predict([list(mf_teste)])[0]
#    print r_recomendado
    delta_time = time.time() - start_time    
#    r_ideal = numGrupos[nomeBase_teste]
#    print r_ideal, r_recomendado
#    dist = np.abs(r_ideal - r_recomendado)
    ng = np.abs(ng)
    
#    ng = int(dist)-2
#    index = indices[ng]
    
#    print dist
    return ng, delta_time#, index

def calcularMetricas(desempenho_fold, Y_real_fold):
    
    mape_all = []
    mrae_all = []
    rrmse_all = []
    
    media_Y_real = np.mean(Y_real_fold) 
    
#    print desempenho_fold
    
    n_datasets = len(desempenho_fold)
    if type(desempenho_fold[0])==list:
        if type(desempenho_fold[0][0])==list:
    #        print 'lista'
            n_mfs = len(desempenho_fold[0])
            n_models = len(desempenho_fold[0][0])          
            
            for i in range(n_mfs):
                mape_models = []
                mrae_models = []
                rrmse_models = []
                for j in range(n_models):
                    mape = 0.0
                    mrae = 0.0
                    soma1 = 0.0
                    soma2 = 0.0
                    for k in range(n_datasets):
#                        print Y_real_fold[k], desempenho_fold[k][i][j], media_Y_real
                        mape = mape + float(np.abs(Y_real_fold[k] - desempenho_fold[k][i][j]))/Y_real_fold[k]
                        mrae = mrae + float(np.abs(Y_real_fold[k] - desempenho_fold[k][i][j] + 0.00001))/(np.abs(Y_real_fold[k] - media_Y_real) + 0.01)
#                        print float(np.abs(Y_real_fold[k] - desempenho_fold[k][i][j] + 0.00001))/np.abs(Y_real_fold[k] - media_Y_real + 0.1)
                        soma1 = soma1 + (Y_real_fold[k] - desempenho_fold[k][i][j])**2.0
                        soma2 = soma2 + (Y_real_fold[k] - media_Y_real)**2.0
                    mape = mape/n_datasets
                    mrae = mrae/n_datasets
                    rrmse = (soma1/(soma2 + 0.00001))**0.5
                    mape_models.append(mape)
                    mrae_models.append(mrae)
                    rrmse_models.append(rrmse)
                mape_all.append(mape_models)
                mrae_all.append(mrae_models)
                rrmse_all.append(rrmse_models)
        else:
            n_mfs = len(desempenho_fold[0])
            for i in range(n_mfs):
                mape = 0.0
                mrae = 0.0
                soma1 = 0.0
                soma2 = 0.0
                for k in range(n_datasets):
                    mape = mape + float(np.abs(Y_real_fold[k] - desempenho_fold[k][i]))/Y_real_fold[k]
                    mrae = mrae + float(np.abs(Y_real_fold[k] - desempenho_fold[k][i] + 0.00001))/(np.abs(Y_real_fold[k] - media_Y_real) + 0.01)
                    soma1 = soma1 + (Y_real_fold[k] - desempenho_fold[k][i])**2.0
                    soma2 = soma2 + (Y_real_fold[k] - media_Y_real)**2.0
                mape = mape/n_datasets
                mrae = mrae/n_datasets
                rrmse = (soma1/(soma2 + 0.00001))**0.5
                mape_all.append(mape)
                mrae_all.append(mrae)
                rrmse_all.append(rrmse)
           
    else:
        mape = 0.0
        mrae = 0.0
        soma1 = 0.0
        soma2 = 0.0
        for k in range(n_datasets):
#            print Y_real_fold[k], desempenho_fold[k], media_Y_real
            mape = mape + float(np.abs(Y_real_fold[k] - desempenho_fold[k]))/Y_real_fold[k]
            mrae = mrae + float(np.abs(Y_real_fold[k] - desempenho_fold[k] + 0.00001))/(np.abs(Y_real_fold[k] - media_Y_real) + 0.01)
#            print float(np.abs(Y_real_fold[k] - desempenho_fold[k] + 0.00001))/np.abs(Y_real_fold[k] - media_Y_real + 0.00001)
            soma1 = soma1 + (Y_real_fold[k] - desempenho_fold[k])**2.0
            soma2 = soma2 + (Y_real_fold[k] - media_Y_real)**2.0
        mape = mape/n_datasets
        mrae = mrae/n_datasets
#        print mape, mrae
        rrmse = (soma1/(soma2 + 0.00001))**0.5
        mape_all.append(mape)
        mrae_all.append(mrae)
        rrmse_all.append(rrmse)
        
#    print mape_all
#    print mrae_all
#    print rrmse_all
    
    return mape_all, mrae_all, rrmse_all

def printTempo(tempo, string):
#    n = len(tempo)
#    print string+' ='
#    for i in range(n):
#            m = len(tempo[0])
#            for j in range(m):
#                print '%.8f' % tempo[i][j],
#            print ''
    print string+' ='
    m = len(tempo[0])
    for j in range(m):
        n = len(tempo)    
        for i in range(n):  
            if i != n-1:
                print '%.8f' % tempo[i][j],
            else:
                 print '%.8f' % tempo[i][j]
#        print ''



        
        
diretorio = "C:/Users/Bruno/Documents"
filtro_ng = -1 # -1 para todas as bases, qualquer que seja o numero de grupos
numGrupos, todosDados = getNumGruposBases(diretorio, [], filtro_ng)
#print len(numGrupos), len(todosDados)


tipo = "distancia"

names = ["Nearest Neighbors",
         "Linear SVM",
         "RBF SVM", 
         "Decision Tree",
         "Random Forest",
         "AdaBoost",         
         "Naive Bayes",
         "Linear Regression",
         "Logistic Regression",
         "SGD Classifier"
         ]

#classifiers = [
#    KNeighborsClassifier(n_neighbors=5),    
#    SVC(kernel="linear", max_iter=30),#30
#    SVC(max_iter=30),#30
#    DecisionTreeClassifier(),
#    RandomForestClassifier(),
#    AdaBoostClassifier(n_estimators=20),#20'''    
#    GaussianNB(),
#    Lasso(),
#    LogisticRegression(max_iter=50),#50
#    SGDClassifier(max_iter=10)#10
#]

#classifiers = [
#    KNeighborsRegressor(n_neighbors=5),
#    SVC(kernel='linear', max_iter=30),
#    SVR(max_iter=50),
#    DecisionTreeRegressor(),
#    RandomForestRegressor(),
#    AdaBoostRegressor(),
#    Perceptron(),
##    BayesianRidge(),
#    Lasso(),
#    LogisticRegression(),
#    SGDClassifier(max_iter=10)#10
##    LinearRegression(fit_intercept=False),
##    HuberRegressor(max_iter=300)
#]

classifiers = [
    KNeighborsRegressor(n_neighbors=5),
    SVC(kernel='linear', max_iter=30),
    SVR(max_iter=50),
    RandomForestRegressor(),
    DecisionTreeRegressor(),
    AdaBoostRegressor(),
    Perceptron(),
    Lasso(),
    LogisticRegression(),
    SGDClassifier(max_iter=10)
]

nomes_bases = numGrupos.keys()

metafeatures_teste_artigo = carregar_metafeatures(diretorio, "artigo", [], [], nomes_bases)
metafeatures_teste_clustering = carregar_metafeatures(diretorio, "Clustering", [], [], nomes_bases)
metafeatures_teste_distancia = carregar_metafeatures(diretorio, "distancia", [], [], nomes_bases)
metafeatures_teste_avaliacao = carregar_metafeatures(diretorio, "Vukicevic", [], [], nomes_bases)
metafeatures_teste_proposto = carregar_metafeatures(diretorio, "proposto3", [], [], nomes_bases)
metafeatures_teste_ng = carregar_metafeatures(diretorio, "ng", [], [], nomes_bases)
#metafeatures_teste_all = carregar_metafeatures(diretorio, "all", [], [], nomes_bases)

mfs_teste = [metafeatures_teste_artigo, metafeatures_teste_clustering, 
             metafeatures_teste_distancia, metafeatures_teste_avaliacao, 
             metafeatures_teste_proposto, metafeatures_teste_ng]
#mfs_teste = [metafeatures_teste_ng]

#nomes_bases = pegar_nome_bases(carregar_metafeatures(diretorio, tipo, [], []))
#nomes_bases = numGrupos.keys()

#print len(metafeatures_teste_artigo)

min_nc=2
max_nc=10
#nome_index = 'elbow'
nome_index = 'bic'
#nome_index = 'aic'
#nome_index = "db"
#nome_index = "dunn"
#nome_index = "kl"
#nome_index = "sdindex"
#nome_index = "silhouette"
if nome_index == 'elbow':
    min_nc = 1
#----------------
Ys, tempos_Ys, indices = pegar_Y(todosDados, min_nc, max_nc, nome_index)
#print Ys.values()
#print indices
#----------------
#Ys = numGrupos #Numero de grupos originais
#tempos_Ys = {}
#for i in Ys:
#    tempos_Ys[i] = 0
#---------------
count_Ys = Counter(Ys.values())
print count_Ys
print np.mean(Ys.values()), max(set(Ys.values()), key = Ys.values().count), sum(count_Ys.values())


np.random.seed(10)
numRep = 2
numFolds = 10

desempenho_final_standard_mape = []
desempenho_final_standard_mrae = []
desempenho_final_standard_rrmse = []
desempenho_final_random = []
desempenho_final_majority_mape = []
desempenho_final_majority_mrae = []
desempenho_final_majority_rrmse = []
desempenho_final_ari_standard = []
desempenho_final_ari_majority = []
desempenho_final_ranking_mape = []
desempenho_final_ranking_mrae = []
desempenho_final_ranking_rrmse = []
desempenho_final_ari = []
desempenho_final_mape = []
desempenho_final_mrae = []
desempenho_final_rrmse = []
tempo_final = []
tempo_final_Y = []
ranking_final = []
importances_final = []
desempenho_final_literatura = []
for rep in range(numRep):
    print '>>> repeticao '+str(rep+1)
    f = criar_k_fold(numFolds, nomes_bases)
    desempenho_global_standard_mape = []
    desempenho_global_standard_mrae = []
    desempenho_global_standard_rrmse = []
    desempenho_global_random = []
    desempenho_global_majority_mape = []
    desempenho_global_majority_mrae = []
    desempenho_global_majority_rrmse = []
    desempenho_global_ari_standard = []
    desempenho_global_ari_majority = []
    desempenho_global_ari = []
    desempenho_global_mape = []
    desempenho_global_mrae = []
    desempenho_global_rrmse = []
    desempenho_global_ranking_mape = []
    desempenho_global_ranking_mrae = []
    desempenho_global_ranking_rrmse = []
    tempo_global = []
    tempo_global_Y = []
    ranking_global = []
    importances_global = []
    desempenho_global_literatura = []
    for i in range(len(f)):  
        print 'fold '+str(i+1)+' --- '+str(datetime.now())
        fold_i = f[i]
        #print fold_i        
        
#        metafeatures_distancia = carregar_metafeatures(diretorio, "distancia", fold_i, [], nomes_bases)
#        #print pegar_nome_bases(metafeatures_distancia)
#        #metafeatures_proposto1 = carregar_metafeatures(diretorio, "proposto1", fold_i)        
#        #metafeatures_proposto2 = carregar_metafeatures(diretorio, "proposto2", fold_i)
#        metafeatures_artigo = carregar_metafeatures(diretorio, "artigo", fold_i, [], nomes_bases)
#        metafeatures_proposto3 = carregar_metafeatures(diretorio, "proposto3", fold_i, [], nomes_bases)
#        metafeatures_avaliacao = carregar_metafeatures(diretorio, "Vukicevic", fold_i, [], nomes_bases)
#        #metafeatures_artigo = carregar_metafeatures(diretorio, "artigo", fold_i)
        
        metafeatures_artigo = carregar_metafeatures_fold(metafeatures_teste_artigo, fold_i)
        metafeatures_clustering = carregar_metafeatures_fold(metafeatures_teste_clustering, fold_i)
        metafeatures_distancia = carregar_metafeatures_fold(metafeatures_teste_distancia, fold_i)
        metafeatures_avaliacao = carregar_metafeatures_fold(metafeatures_teste_avaliacao, fold_i)
        metafeatures_proposto3 = carregar_metafeatures_fold(metafeatures_teste_proposto, fold_i)
        metafeatures_ng = carregar_metafeatures_fold(metafeatures_teste_ng, fold_i)
#        metafeatures_all = carregar_metafeatures_fold(metafeatures_teste_all, fold_i)        
        
        
#        print len(fold_i), len(metafeatures_artigo)
        
        mfs = [metafeatures_artigo, metafeatures_clustering, metafeatures_distancia, 
               metafeatures_avaliacao, metafeatures_proposto3, metafeatures_ng]
#        mfs = [metafeatures_ng]
        
        
        nomesBases = getNomeBases(nomes_bases, fold_i)
        #------------------
        Y = [Ys[a] for a in nomesBases] #Calcula usando indices
        #------------------
#        Y = getNumGruposBases2(numGrupos, fold_i) #Pega os originais
        #------------------
#        print Y
        tempo_Y = [tempos_Ys[a] for a in nomesBases]
#        print np.mean(tempo_Y)
#        print len(Y)
        
        media_Y = np.mean(Y)
        moda_Y = max(set(Y), key = Y.count)
        
        desempenhos_fold = []
        desempenhos_fold = []
        desempenhos_fold = []
        desempenho_standard = []
        desempenho_standard = []
        desempenho_standard = []
        desempenho_random = []
        desempenho_majority = []
        desempenho_majority = []
        desempenho_majority = []
        desempenho_ari_standard = []
        desempenho_ari_majority = []
        desempenhos_fold_ari = []
        tempo_fold = []
        tempo_fold_Y = []
        ranking_fold = []
        importances_fold = []
        desempenho_fold_literatura = []  
        Y_real_fold = []
        Y_real_fold_valor = []
        indices_fold = []
        indices_fold_valor = []
        
#        desempenho_standard.append(np.mean(Y))
#        desempenho_majority.append(max(set(Y), key = Y.count))       
        
        for j in range(len(fold_i)):        
            nomeBase_teste = fold_i[j]  
#            print nomeBase_teste 
            
            desempenho_literatura = []
#            print todosDados.keys()
#            ng_silhoutte = recomendarNGrupos(todosDados[nomeBase_teste], "kmeans", "silhouette", min_nc, max_nc)
#            ng_db = recomendarNGrupos(todosDados[nomeBase_teste], "kmeans", "db", min_nc, max_nc)
#            ng_sdbw = recomendarNGrupos(todosDados[nomeBase_teste], "kmeans", "sdbw", min_nc, max_nc)
#            ng_gamma = recomendarNGrupos(todosDados[nomeBase_teste], "kmeans", "dunn", min_nc, max_nc)
#            print ng_silhoutte, ng_db, ng_sdbw, ng_gamma
#            r_ideal = numGrupos[nomeBase_teste] #ng = a priori
            r_ideal = Ys[nomeBase_teste]    #ng = usando indice 
#            ng_literatura = [ng_silhoutte, ng_db, ng_sdbw, ng_gamma]
            ng_literatura = [0]*4
#            ng_literatura = [np.abs(ng-r_ideal) for ng in ng_literatura]
            ng_literatura = [np.abs(ng) for ng in ng_literatura]
#            print ng_literatura
            desempenho_literatura = ng_literatura
            desempenho_fold_literatura.append(desempenho_literatura)
            
            
#            desempenho_standard.append(np.abs(media_Y - r_ideal))
#            desempenho_majority.append(np.abs(moda_Y - r_ideal))
            desempenho_standard.append(np.abs(media_Y))
            desempenho_majority.append(np.abs(moda_Y))
            
            Y_real_fold.append(r_ideal)
            Y_real_fold_valor.append(indices[nomeBase_teste][int(r_ideal)-min_nc])
                        
#            desempenhos_k = [[]]*len(mfs)
            desempenhos_k = []
            desempenhos_k_ari = [[]]*len(mfs)
            tempo_k = [[]]*len(mfs)  
            indices_k = []
            indices_k_valor = []
            ranking_k = [[]]*len(mfs) 
#            print ranking_k
#            Y = pegar_Y(diretorio, mfs[0])
            for l in range(len(mfs)):
                ng_k = []
                src_k_ari = []
                tempo_m = []
                index_k = []
#                X = pegar_X(mfs[l]) 
                X, Y = pegar_X_Y_balanceado(mfs[l], Ys)                
                #mf_teste = pegar_metafeatures_nome(mfs[l], nomeBase_teste)
                #mf_teste = pegar_metafeatures_nome(mfs_all[l], nomeBase_teste)
                mf_teste = pegar_metafeatures_nome(mfs_teste[l], nomeBase_teste)
                # iterate over classifiers
                for name, clf in zip(names, classifiers):
                    #print name
                    ng_rec, delta_time = calcular_NG(mf_teste, clf, X, Y, r_ideal, indices[nomeBase_teste])                    
                    ng = int(ng_rec)-min_nc
                    index = indices[nomeBase_teste][ng]
                    #print(src)
#                    print dist
                    ng_k.append(ng_rec)
                    tempo_m.append(delta_time)  
                    index_k.append(index)
                    desempenhos_k_ari[l] = 0
#                desempenhos_k[l] = src_k
#                print src_k
                desempenhos_k.append(ng_k)
                tempo_k[l] = tempo_m 
                if nome_index == 'db' or nome_index == 'sdindex' or nome_index == 'elbow' or nome_index == 'bic' or nome_index == 'aic':
#                    print 'min'
                    indices_k.append(ng_k[index_k.index(min(index_k))])
                    ranking_k[l] = np.argsort(index_k)
                    indices_k_valor.append(min(index_k))
                else:
#                    print 'max'
                    indices_k.append(ng_k[index_k.index(max(index_k))])
                    ranking_k[l] = np.argsort([-1.0*q for q in index_k])
                    indices_k_valor.append(max(index_k))
#                print(importances_k)
#                print desempenhos_k
#                print ranking_k
            desempenhos_fold.append(desempenhos_k)
            desempenhos_fold_ari.append(desempenhos_k_ari)
            tempo_fold.append(tempo_k)
            indices_fold.append(indices_k)
            ranking_fold.append(ranking_k)
            indices_fold_valor.append(indices_k_valor)
            
#        print indices_fold
            
        mape, mrae, rrmse = calcularMetricas(desempenhos_fold, Y_real_fold)
        mape_standard, mrae_standard, rrmse_standard = calcularMetricas(desempenho_standard, Y_real_fold)
        mape_majority, mrae_majority, rrmse_majority = calcularMetricas(desempenho_majority, Y_real_fold)
        #mape_ranking, mrae_ranking, rrmse_ranking = calcularMetricas(indices_fold, Y_real_fold)  
        mape_ranking, mrae_ranking, rrmse_ranking = calcularMetricas(indices_fold_valor, Y_real_fold_valor)  
        
        
        
#        print desempenhos_fold
#        print np.mean(desempenho_standard)
        
#        print mrae_standard, mrae_majority 
        
        desempenho_global_standard_mape.append(np.mean(mape_standard))
        desempenho_global_standard_mrae.append(np.mean(mrae_standard))
        desempenho_global_standard_rrmse.append(np.mean(rrmse_standard))
        
        desempenho_global_random.append(np.mean(desempenho_random))
        
        desempenho_global_majority_mape.append(np.mean(mape_majority))
        desempenho_global_majority_mrae.append(np.mean(mrae_majority))
        desempenho_global_majority_rrmse.append(np.mean(rrmse_majority))
        
        
        desempenho_global_ari_standard.append(np.mean(desempenho_ari_standard))
        desempenho_global_ari_majority.append(np.mean(desempenho_ari_majority))
        
        desempenho_global_ranking_mape.append(mape_ranking)
        desempenho_global_ranking_mrae.append(mrae_ranking)
        desempenho_global_ranking_rrmse.append(rrmse_ranking)
        
#        print desempenho_global_ranking_mape
        
#        desempenhos_fold = np.array(desempenhos_fold)
#        print np.mean(desempenhos_fold, axis=0)
        desempenho_global_mape.append(mape)
        desempenho_global_mrae.append(mrae)
        desempenho_global_rrmse.append(rrmse)
#        print desempenho_global_mape
        
        desempenhos_fold_ari = np.array(desempenhos_fold_ari)
        #print np.mean(desempenhos_fold_ari, axis=0)
        desempenho_global_ari.append(np.mean(desempenhos_fold_ari, axis=0))
        #print desempenho_global_ari
        
        tempo_fold = np.array(tempo_fold)
        tempo_global.append(np.mean(tempo_fold, axis=0))
        #print tempo_global        
        
        tempo_global_Y.append(np.mean(tempo_Y))
        
        desempenho_fold_literatura = np.array(desempenho_fold_literatura)
#        print np.mean(desempenho_fold_literatura, axis=0)
        desempenho_global_literatura.append(np.mean(desempenho_fold_literatura, axis=0))
#        print desempenho_global_literatura
        
        ranking_fold = np.array(ranking_fold)
        ranking_global.append(np.argsort(np.mean(ranking_fold, axis=0)))
#        print ranking_global
     
    desempenho_final_standard_mape.append(np.mean(desempenho_global_standard_mape))
    desempenho_final_standard_mrae.append(np.mean(desempenho_global_standard_mrae))
    desempenho_final_standard_rrmse.append(np.mean(desempenho_global_standard_rrmse))
    
#    print desempenho_final_standard_mape
#    print desempenho_final_standard_mrae
#    print desempenho_final_standard_rrmse
    
    desempenho_final_ari_standard.append(np.mean(desempenho_global_ari_standard))  
    desempenho_final_random.append(np.mean(desempenho_global_random))
    
    desempenho_final_majority_mape.append(np.mean(desempenho_global_majority_mape))
    desempenho_final_majority_mrae.append(np.mean(desempenho_global_majority_mrae))
    desempenho_final_majority_rrmse.append(np.mean(desempenho_global_majority_rrmse))
    
    desempenho_final_ari_majority.append(np.mean(desempenho_global_ari_majority))
    
    
    desempenho_final_ranking_mape.append(np.mean(desempenho_global_ranking_mape,axis=0))
    desempenho_final_ranking_mrae.append(np.mean(desempenho_global_ranking_mrae,axis=0))
    desempenho_final_ranking_rrmse.append(np.mean(desempenho_global_ranking_rrmse,axis=0))
    
#    print 'desempenho_final_ranking_mape=', list(desempenho_final_ranking_mape)
#    print 'desempenho_final_ranking_mrae=', list(desempenho_final_ranking_mrae)
#    print 'desempenho_final_ranking_rrmse=', list(desempenho_final_ranking_rrmse)
    
#    media_desempenho_global_standard = np.mean(desempenho_global_standard)
#    desvio_desempenho_global_standard = np.std(desempenho_global_standard)
#    media_desempenho_global_majority = np.mean(desempenho_global_majority)
#    desvio_desempenho_global_majority = np.std(desempenho_global_majority)
#    print '\nmedia_standard = %.4f  desvio_standard = %.4f' % (media_desempenho_global_standard, desvio_desempenho_global_standard)
#    print '\nmedia_majority = %.4f  desvio_majority = %.4f' % (media_desempenho_global_majority, desvio_desempenho_global_majority)
     
    
#    print desempenho_global_mape
    resultado_global_mape=[]
    for i in range(numFolds):
        resultado_global_mape.append(desempenho_global_mape[i])
    media_desempenho_global_mape = np.mean(resultado_global_mape,axis=0)
    desempenho_final_mape.append(media_desempenho_global_mape)
#    desvio_desempenho_global = np.std(resultado_global,axis=0)
#    print '\nmedia_desempenho_global_mape =\n '+str(media_desempenho_global_mape)
#    print 'desvio_desempenho_global = '+str(desvio_desempenho_global)
    
    resultado_global_mrae=[]
    for i in range(numFolds):
        resultado_global_mrae.append(desempenho_global_mrae[i])
    media_desempenho_global_mrae = np.mean(resultado_global_mrae,axis=0)
    desempenho_final_mrae.append(media_desempenho_global_mrae)
#    desvio_desempenho_global = np.std(resultado_global,axis=0)
#    print '\nmedia_desempenho_global_mrae =\n '+str(media_desempenho_global_mrae)
#    print 'desvio_desempenho_global = '+str(desvio_desempenho_global)
    
    resultado_global_rrmse=[]
    for i in range(numFolds):
        resultado_global_rrmse.append(desempenho_global_rrmse[i])
    media_desempenho_global_rrmse = np.mean(resultado_global_rrmse,axis=0)
    desempenho_final_rrmse.append(media_desempenho_global_rrmse)
#    desvio_desempenho_global = np.std(resultado_global,axis=0)
#    print '\nmedia_desempenho_global_rrmse =\n '+str(media_desempenho_global_rrmse)
#    print 'desvio_desempenho_global = '+str(desvio_desempenho_global)
    
    resultado_global_ari=[]
    for i in range(numFolds):
        resultado_global_ari.append(desempenho_global_ari[i])
    media_desempenho_global_ari = np.mean(resultado_global_ari,axis=0)
    desempenho_final_ari.append(media_desempenho_global_ari)
#    desvio_desempenho_global = np.std(resultado_global,axis=0)
#    print '\nmedia_desempenho_global_ari =\n '+str(media_desempenho_global_ari)
#    print 'desvio_desempenho_global = '+str(desvio_desempenho_global)
    
    resultado_global=[]
    for i in range(numFolds):
        resultado_global.append(tempo_global[i])
    media_desempenho_global_tempo = np.mean(resultado_global,axis=0)
    tempo_final.append(media_desempenho_global_tempo)
##    desvio_desempenho_global = np.std(resultado_global,axis=0)
#    print '\nmedia_tempo_global =\n '+str(media_desempenho_global_tempo)
##    print 'desvio_desempenho_global = '+str(desvio_desempenho_global)
    
    tempo_final_Y.append(np.mean(tempo_global_Y))
    
    resultado_global_ranking=[]
    for i in range(numFolds):
        resultado_global_ranking.append(ranking_global[i])
    media_desempenho_global_ranking = np.mean(resultado_global_ranking,axis=0)
    ranking_final.append(np.argsort(media_desempenho_global_ranking))
    
#    resultado_global=[]
#    for i in range(numFolds):
#        resultado_global.append(importances_global[i])
#    media_desempenho_global = np.mean(resultado_global,axis=0)
#    importances_final.append(media_desempenho_global)
##    desvio_desempenho_global = np.std(resultado_global,axis=0)
#    print '\nmedia_importances_global =\n '+str(media_desempenho_global)
##    print 'desvio_desempenho_global = '+str(desvio_desempenho_global)
    
    resultado_global=[]
    for i in range(numFolds):
        resultado_global.append(desempenho_global_literatura[i])
    media_desempenho_global_literatura = np.mean(resultado_global,axis=0)
    desempenho_final_literatura.append(media_desempenho_global_literatura)
#    print '\nmedia_desempenho_global_literatura =\n '+str(media_desempenho_global_literatura)

media_desempenho_final_standard_mape = np.mean(desempenho_final_standard_mape)
media_desempenho_final_standard_mrae = np.mean(desempenho_final_standard_mrae)
media_desempenho_final_standard_rrmse = np.mean(desempenho_final_standard_rrmse)
desvio_desempenho_final_standard_mape = np.std(desempenho_final_standard_mape)
desvio_desempenho_final_standard_mrae = np.std(desempenho_final_standard_mrae)
desvio_desempenho_final_standard_rrmse = np.std(desempenho_final_standard_rrmse)
print '\nmedia_standard_final_mape = %.4f  desvio_standard_final_mape = %.4f' % (media_desempenho_final_standard_mape, desvio_desempenho_final_standard_mape)
print '\nmedia_standard_final_mrae = %.4f  desvio_standard_final_mrae = %.4f' % (media_desempenho_final_standard_mrae, desvio_desempenho_final_standard_mrae)
print '\nmedia_standard_final_rrmse = %.4f  desvio_standard_final_rrmse = %.4f' % (media_desempenho_final_standard_rrmse, desvio_desempenho_final_standard_rrmse)
#media_desempenho_final_standard_ari = np.mean(desempenho_final_ari_standard)
#desvio_desempenho_final_standard_ari = np.std(desempenho_final_ari_standard)
#print '\nmedia_standard_final_ari = %.4f  desvio_standard_final_ari = %.4f' % (media_desempenho_final_standard_ari, desvio_desempenho_final_standard_ari)

#media_desempenho_final_random = np.mean(desempenho_final_random)
#desvio_desempenho_final_random = np.std(desempenho_final_random)
#print '\nmedia_random_final = %.4f  desvio_random_final = %.4f' % (media_desempenho_final_random, desvio_desempenho_final_random)

media_desempenho_final_majority_mape = np.mean(desempenho_final_majority_mape)
media_desempenho_final_majority_mrae = np.mean(desempenho_final_majority_mrae)
media_desempenho_final_majority_rrmse = np.mean(desempenho_final_majority_rrmse)
desvio_desempenho_final_majority_mape = np.std(desempenho_final_majority_mape)
desvio_desempenho_final_majority_mrae = np.std(desempenho_final_majority_mrae)
desvio_desempenho_final_majority_rrmse = np.std(desempenho_final_majority_rrmse)
print '\nmedia_majority_final_mape = %.4f  desvio_majority_final_mape = %.4f' % (media_desempenho_final_majority_mape, desvio_desempenho_final_majority_mape)
print '\nmedia_majority_final_mrae = %.4f  desvio_majority_final_mrae = %.4f' % (media_desempenho_final_majority_mrae, desvio_desempenho_final_majority_mrae)
print '\nmedia_majority_final_rrmse = %.4f  desvio_majority_final_rrmse = %.4f' % (media_desempenho_final_majority_rrmse, desvio_desempenho_final_majority_rrmse)
#media_desempenho_final_majority_ari = np.mean(desempenho_final_ari_majority)
#desvio_desempenho_final_majority_ari = np.std(desempenho_final_ari_majority)
#print '\nmedia_majority_final_ari = %.4f  desvio_majority_final_ari = %.4f' % (media_desempenho_final_majority_ari, desvio_desempenho_final_majority_ari)

media_desempenho_final_mape = np.mean(desempenho_final_mape,axis=0)
media_desempenho_final_mrae = np.mean(desempenho_final_mrae,axis=0)
media_desempenho_final_rrmse = np.mean(desempenho_final_rrmse,axis=0)
desvio_desempenho_final_mape = np.std(desempenho_final_mape,axis=0)
desvio_desempenho_final_mrae = np.std(desempenho_final_mrae,axis=0)
desvio_desempenho_final_rrmse = np.std(desempenho_final_rrmse,axis=0)
#print '\nmedia_desempenho_final =\n '+str(media_desempenho_final)
#print 'desvio_desempenho_final =\n '+str(desvio_desempenho_final)
n_mf = len(media_desempenho_final_mape)
#print n_mf
media_desempenho_final_mape = list(media_desempenho_final_mape)
media_desempenho_final_mrae = list(media_desempenho_final_mrae)
media_desempenho_final_rrmse = list(media_desempenho_final_rrmse)
desvio_desempenho_final_mape = list(desvio_desempenho_final_mape)
desvio_desempenho_final_mrae = list(desvio_desempenho_final_mrae)
desvio_desempenho_final_rrmse = list(desvio_desempenho_final_rrmse)
#print media_desempenho_final
#for i in range(n_mf):
#    media_desempenho_final_mape[i] = list(media_desempenho_final_mape[i])
#    media_desempenho_final_mrae[i] = list(media_desempenho_final_mrae[i])
#    media_desempenho_final_rrmse[i] = list(media_desempenho_final_rrmse[i])
#    
#    media_desempenho_final_standard_mape[i].append(media_desempenho_final_standard_mape)
#    media_desempenho_final_standard_mrae[i].append(media_desempenho_final_standard_mrae)
#    media_desempenho_final_standard_rrmse[i].append(media_desempenho_final_standard_rrmse)
#    
#    media_desempenho_final_majority_mape[i].append(media_desempenho_final_majority_mape)
#    media_desempenho_final_majority_mrae[i].append(media_desempenho_final_majority_mrae)
#    media_desempenho_final_majority_rrmse[i].append(media_desempenho_final_majority_rrmse)
#    
#    desvio_desempenho_final_mape[i] = list(desvio_desempenho_final_mape[i])
#    desvio_desempenho_final_mrae[i] = list(desvio_desempenho_final_mrae[i])
#    desvio_desempenho_final_rrmse[i] = list(desvio_desempenho_final_rrmse[i])
#    
#    desvio_desempenho_final_standard_mape[i].append(desvio_desempenho_final_standard_mape)
#    desvio_desempenho_final_standard_mrae[i].append(desvio_desempenho_final_standard_mrae)
#    desvio_desempenho_final_standard_rrmse[i].append(desvio_desempenho_final_standard_rrmse)
#    
#    desvio_desempenho_final_majority_mape[i].append(desvio_desempenho_final_majority_mape)
#    desvio_desempenho_final_majority_mrae[i].append(desvio_desempenho_final_majority_mrae)
#    desvio_desempenho_final_majority_rrmse[i].append(desvio_desempenho_final_majority_rrmse)
#printTempo(media_desempenho_final_mape, 'media_desempenho_final_mape')
#printTempo(desvio_desempenho_final_mape, 'desvio_desempenho_final_mape')
#printTempo(media_desempenho_final_mrae, 'media_desempenho_final_mrae')
#printTempo(desvio_desempenho_final_mrae, 'desvio_desempenho_final_mrae')
#printTempo(media_desempenho_final_rrmse, 'media_desempenho_final_rrmse')
#printTempo(desvio_desempenho_final_rrmse, 'desvio_desempenho_final_rrmse')

#media_desempenho_final_ari = np.mean(desempenho_final_ari,axis=0)
#desvio_desempenho_final_ari = np.std(desempenho_final_ari,axis=0)
#print '\nmedia_desempenho_final_ari =\n '+str(media_desempenho_final_ari)
#print 'desvio_desempenho_final_ari =\n '+str(desvio_desempenho_final_ari)
media_desempenho_final_literatura = np.mean(desempenho_final_literatura,axis=0)
desvio_desempenho_final_literatura = np.std(desempenho_final_literatura,axis=0)
#print '\nmedia_desempenho_final_literatura =\n '+str(media_desempenho_final_literatura)
#print 'desvio_desempenho_final_literatura =\n '+str(desvio_desempenho_final_literatura)

media_desempenho_final_tempo_Y = np.mean(tempo_final_Y)
desvio_desempenho_final_tempo_Y = np.std(tempo_final_Y)
#print '\nmedia_desempenho_final_tempo_Y = %.8f\n' % media_desempenho_final_tempo_Y
#print 'desvio_desempenho_final_tempo_Y = %.8f\n' % desvio_desempenho_final_tempo_Y
#printTempo(media_desempenho_final_tempo_Y, 'media_desempenho_final_tempo_Y')
#printTempo(desvio_desempenho_final_tempo_Y, 'desvio_desempenho_final_tempo_Y')

media_desempenho_final_tempo = np.mean(tempo_final,axis=0)
desvio_desempenho_final_tempo = np.std(tempo_final,axis=0)
#print '\nmedia_desempenho_final_tempo =\n '+str(media_desempenho_final_tempo)
#print 'desvio_desempenho_final_tempo =\n '+str(desvio_desempenho_final_tempo)
media_desempenho_final_tempo = list(media_desempenho_final_tempo)
desvio_desempenho_final_tempo = list(desvio_desempenho_final_tempo)
#print media_desempenho_final
for i in range(n_mf):
    media_desempenho_final_tempo[i] = list(media_desempenho_final_tempo[i])
    media_desempenho_final_tempo[i].append(media_desempenho_final_tempo_Y)
    desvio_desempenho_final_tempo[i] = list(desvio_desempenho_final_tempo[i])
    desvio_desempenho_final_tempo[i].append(desvio_desempenho_final_tempo_Y)
#printTempo(media_desempenho_final_tempo, 'media_desempenho_final_tempo')
#printTempo(desvio_desempenho_final_tempo, 'desvio_desempenho_final_tempo')

media_importances_final = np.mean(importances_final,axis=0)
desvio_importances_final = np.std(importances_final,axis=0)
#print '\nmedia_importances_final =\n '+str(media_importances_final)
#print 'desvio_importances_final =\n '+str(desvio_importances_final)

media_desempenho_final_ranking = np.mean(ranking_final,axis=0)
desvio_desempenho_final_ranking = np.std(ranking_final,axis=0)
print '\nmedia_desempenho_final_ranking =\n '+str(media_desempenho_final_ranking)
print 'desvio_desempenho_final_ranking =\n '+str(desvio_desempenho_final_ranking)

#nomes_mf = ['Statistical', 'Clustering', 'Distance', 'Evaluation', 'Correlation', 'NB']
for a in range(len(media_desempenho_final_ranking)):
    for b in range(len(media_desempenho_final_ranking[0])):
        print '& $%.4f$' % (media_desempenho_final_ranking[a][b]+1.0),
    print '\\\ '
    for b in range(len(desvio_desempenho_final_ranking[0])):
        print '& $(%.4f)$' % desvio_desempenho_final_ranking[a][b],
    print '\\\ \hline '
    


media_desempenho_final_ranking_mape = np.mean(desempenho_final_ranking_mape,axis=0)
media_desempenho_final_ranking_mrae = np.mean(desempenho_final_ranking_mrae,axis=0)
media_desempenho_final_ranking_rrmse = np.mean(desempenho_final_ranking_rrmse,axis=0)
desvio_desempenho_final_ranking_mape = np.std(desempenho_final_ranking_mape,axis=0)
desvio_desempenho_final_ranking_mrae = np.std(desempenho_final_ranking_mrae,axis=0)
desvio_desempenho_final_ranking_rrmse = np.std(desempenho_final_ranking_rrmse,axis=0)
#print '\nmedia_desempenho_final =\n '+str(media_desempenho_final)
#print 'desvio_desempenho_final =\n '+str(desvio_desempenho_final)
n_mf = len(media_desempenho_final_mape)
#print n_mf
media_desempenho_final_ranking_mape = list(media_desempenho_final_ranking_mape)
media_desempenho_final_ranking_mrae = list(media_desempenho_final_ranking_mrae)
media_desempenho_final_ranking_rrmse = list(media_desempenho_final_ranking_rrmse)
desvio_desempenho_final_ranking_mape = list(desvio_desempenho_final_ranking_mape)
desvio_desempenho_final_ranking_mrae = list(desvio_desempenho_final_ranking_mrae)
desvio_desempenho_final_ranking_rrmse = list(desvio_desempenho_final_ranking_rrmse)

print '\nmedia_desempenho_final_ranking_mape=', media_desempenho_final_ranking_mape
print '\nmedia_desempenho_final_ranking_mrae=',media_desempenho_final_ranking_mrae
print '\nmedia_desempenho_final_ranking_rrmse=',media_desempenho_final_ranking_rrmse
print '\ndesvio_desempenho_final_ranking_mape=',desvio_desempenho_final_ranking_mape
print '\ndesvio_desempenho_final_ranking_mrae=',desvio_desempenho_final_ranking_mrae
print '\ndesvio_desempenho_final_ranking_rrmse=',desvio_desempenho_final_ranking_rrmse



linha_mape = [media_desempenho_final_ranking_mape[a] for a in range(n_mf)]
linha_mape.append(media_desempenho_final_standard_mape)
linha_mape.append(media_desempenho_final_majority_mape)
linha_mrae = [media_desempenho_final_ranking_mrae[a] for a in range(n_mf)]
linha_mrae.append(media_desempenho_final_standard_mrae)
linha_mrae.append(media_desempenho_final_majority_mrae)
linha_rrmse = [media_desempenho_final_ranking_rrmse[a] for a in range(n_mf)]
linha_rrmse.append(media_desempenho_final_standard_rrmse)
linha_rrmse.append(media_desempenho_final_majority_rrmse)

linha_mape_desvio = [desvio_desempenho_final_ranking_mape[a] for a in range(n_mf)]
linha_mape_desvio.append(desvio_desempenho_final_standard_mape)
linha_mape_desvio.append(desvio_desempenho_final_majority_mape)
linha_mrae_desvio = [desvio_desempenho_final_ranking_mrae[a] for a in range(n_mf)]
linha_mrae_desvio.append(desvio_desempenho_final_standard_mrae)
linha_mrae_desvio.append(desvio_desempenho_final_majority_mrae)
linha_rrmse_desvio = [desvio_desempenho_final_ranking_rrmse[a] for a in range(n_mf)]
linha_rrmse_desvio.append(desvio_desempenho_final_standard_rrmse)
linha_rrmse_desvio.append(desvio_desempenho_final_majority_rrmse)

print 'MAPE ',
for a in range(n_mf+2):
    print '& $%.4f$' % linha_mape[a],
print '\\\ '
for a in range(n_mf+2):
    print '& $(%.4f)$' % linha_mape_desvio[a],
print '\\\ \hline '

print 'MRAE ',
for a in range(n_mf+2):
    print '& $%.4f$' % linha_mrae[a],
print '\\\ '
for a in range(n_mf+2):
    print '& $(%.4f)$' % linha_mrae_desvio[a],
print '\\\ \hline '

print 'RRMSE ',
for a in range(n_mf+2):
    print '& $%.4f$' % linha_rrmse[a],
print '\\\ '
for a in range(n_mf+2):
    print '& $(%.4f)$' % linha_rrmse_desvio[a],
print '\\\ \hline '

 

#numSelececao = 11 
#for j in range(len(classifiers)):
#    print '\n--- SRC (%i) ---' % (j+1)
#    for i in range(numSelececao):
#        print media_desempenho_final[i][j] 
# 
#for j in range(len(classifiers)):      
#    print '\n--- Tempo (%i) ---' % (j+1)
#    for i in range(numSelececao):
#        print media_desempenho_final_tempo[i][j] 
