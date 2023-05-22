"""""""""""
(c) Mikami 2023
"""""""""""
import numpy as np
import os
import csv
import random
import subprocess
import copy
import time
from math import sqrt

####### constants #######
NGEN = 500
NDIM = 2
NPOPULATION = 20
NPARENTS = NDIM + 1
NCHILDREN = int(4 * NDIM)
LEARNING_RATE = 1 / (5 * NDIM) #C_α
SIGMA2 = 1/(NPARENTS - 1) #分散
ALPHA = 1
MU_ALPHA = NPARENTS
GENE_MAX = 2
GENE_MIN = -2
SEED = 32


#########################
def setup(): 
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

def ackley(individual):

    fitness =\
        20 - 20 * np.exp(-0.2 * sqrt(((individual**2).sum())/ NDIM)) + np.e - np.exp((np.cos(2 * np.pi * individual).sum()) / NDIM)

    return fitness



def evaluate(individuals):
    fitnesses = np.zeros(len(individuals))

    for i in range(len(individuals)):
        fitnesses[i] = ackley(individuals[i])

    return fitnesses


def select_parents(n_parents, population, population_fitnesses):

    parents = np.zeros((n_parents,NDIM))
    parents_fitnesses = np.zeros(n_parents)

    for i in range(n_parents):
        index = np.random.randint(0, len(population))
        parents[i] = population[index]
        parents_fitnesses[i] = population_fitnesses[index]
        population = np.delete(population, index, axis=0)
        population_fitnesses = np.delete(population_fitnesses, index)

    return parents, parents_fitnesses, population, population_fitnesses


def selection(children):
    # 子個体の評価
    children_fitnesses = evaluate(children)
    # ソート
    sorted_index_children = np.argsort(children_fitnesses)

    children_survived = np.zeros((NPARENTS,NDIM))
    children_fitnesses_survived = np.zeros((NPARENTS))

    for i in range(NPARENTS):
        children_survived[i] = children[[sorted_index_children[i]]]
        children_fitnesses_survived[i] = children_fitnesses[[sorted_index_children[i]]]
    
    return children_survived, children_fitnesses_survived, sorted_index_children, children_fitnesses

def calc_center(individuals):
    
    center = individuals.sum(axis=0)
    center = center / len(individuals)

    return center



def AREX_crossover(parents, parents_fitnesses,expansion_rate):
    children = np.zeros((NCHILDREN, NDIM))

    # parentsのソート
    sorted_index_parents = np.argsort(parents_fitnesses)

    # 交叉中心降下
    m = np.zeros(NDIM)
    for i in range(NPARENTS):
        m += 2*(NPARENTS+1-i) / (NPARENTS*(NPARENTS+1)) * parents[sorted_index_parents[i]]

    # 親の中心
    parents_center = calc_center(parents)
    
    # 交叉用乱数の準備
    e_random = np.random.normal(loc=0.0, scale=sqrt(SIGMA2), size=(NCHILDREN, NPARENTS))

    for i in range(NCHILDREN):
        sum_e_y = np.zeros(NDIM)
        for j in range(NPARENTS):
            # e = np.random.normal(0.0, scale=sqrt(SIGMA2))
            sum_e_y += e_random[i][j] * (parents[j] - parents_center)

        children[i] = m + expansion_rate * sum_e_y

    return children, e_random


def AREX_expand_adaptation(expansion_rate, e_random, sorted_index_children):
    # Σ<e_j>^2_mu_alpha
    e_random_rankavg = np.zeros(NPARENTS)

    for j in range(NPARENTS):
        for i in range(MU_ALPHA):
            e_random_rankavg[j] += e_random[sorted_index_children[i]][j]
        e_random_rankavg[j] = e_random_rankavg[j] / MU_ALPHA

    e_random_rankavg_2 = e_random_rankavg**2
    sum_1 = e_random_rankavg_2.sum()

    # Σ<e_j>mu_alpha
    sum_2 = e_random_rankavg.sum()

    L_cdp = expansion_rate ** 2 * (NPARENTS - 1) * (sum_1 - (sum_2 **2)/NPARENTS)
    L_avg = (expansion_rate ** 2) * SIGMA2 * ((NPARENTS - 1)**2) / MU_ALPHA

    # update expansion rate
    expansion_rate = max(expansion_rate * sqrt((1 - LEARNING_RATE) + LEARNING_RATE * L_cdp / L_avg), 1.0)
    
    return expansion_rate



def main():
    #計算時間の測定
    time_start = time.time()

    #シード値の設定等
    setup()

    #結果出力フォルダの作成
    os.makedirs('result', exist_ok=True)

    best_fitness = 100 # 最良個体のfitness
    expansion_rate = ALPHA

    times_per_gen = []

    gen = 0

    # 初期集団の生成
    population = np.random.rand(NPOPULATION,NDIM) * (GENE_MAX - GENE_MIN) + GENE_MIN

    # 初期集団の評価
    population_fitnesses = evaluate(population)

    # 目的関数順に並び替え
    sorted_index_pop = np.argsort(population_fitnesses)


    # 最良個体を保存しておく
    best_fitness = population_fitnesses[sorted_index_pop[0]]

    # 出力フォルダの作成
    os.makedirs('result/Generation'+ str(gen), exist_ok=True)
    
    # 最良個体の評価値を出力
    with open('result/elite.csv', 'a', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow([gen, best_fitness])
    f.close

    
    # 集団の評価値を出力
    file_path=os.path.join('result/Generation'+str(gen), "pop_fitness.csv")
    file=open(file_path,"w",newline="")
    writer=csv.writer(file,lineterminator="\n")
    for i in range(len(population)):
        writer.writerow([i, population_fitnesses[i]])
    file.close()
    
    
    times_per_gen.append(time.time() - time_start) 
    
    # 2世代目以降の処理
    for gen in range(1,NGEN+1):
        # 親個体の選択
        parents, parents_fitnesses, population, population_fitnesses = \
            select_parents(NPARENTS, population, population_fitnesses)

        # 親個体を用いて交叉
        children, e_random = AREX_crossover(parents, parents_fitnesses, expansion_rate)

        # 生存個体の選択
        children_survived, children_fitnesses_survived, sorted_index_children ,children_fitnesses= selection(children)

        # 子個体から生存選択し，集団に戻す
        population = np.append(population, children_survived)
        population_fitnesses =\
                np.append(population_fitnesses, children_fitnesses_survived)
        
        population = population.reshape(NPOPULATION, NDIM)
        population_fitnesses = population_fitnesses.reshape(NPOPULATION)


        # AREXの拡張率更新
        expansion_rate = AREX_expand_adaptation(expansion_rate, e_random, sorted_index_children)
        
        # 最上位個体を保存しておく
        if best_fitness > children_fitnesses[sorted_index_children[0]]:
            best_fitness = children_fitnesses[sorted_index_children[0]]

        # 結果の出力
        os.makedirs('result/Generation'+ str(gen), exist_ok=True)
        # save the best fitness
        with open('result/elite.csv', 'a', newline = '') as f:
            writer = csv.writer(f)
            writer.writerow([gen, best_fitness])
        f.close
        
        # 集団の評価値を出力
        file_path=os.path.join('result/Generation'+str(gen), "pop_fitness.csv")
        file=open(file_path,"w",newline="")
        writer=csv.writer(file,lineterminator="\n")
        for i in range(len(population)):
            writer.writerow([i, population_fitnesses[i]])
        file.close()
        
        # すべての子個体を出力
        filename = os.path.join('result/Generation'+str(gen), 'children.csv')
        file=open(filename,"w")
        writer=csv.writer(file,lineterminator="\n")
        writer.writerows(children)
        file.close()

        # 生存個体を出力
        filename = os.path.join('result/Generation'+str(gen), 'children_survived.csv')
        file=open(filename,"w")
        writer=csv.writer(file,lineterminator="\n")
        writer.writerows(children_survived)
        file.close()

        times_per_gen.append(time.time() - times_per_gen[gen-1])

    print('---------- optimize completed ----------')

    # 計算時間記録用データの出力
    with open(os.path.join('result', "times.csv"), "w") as f:
        writer = csv.writer(f)
        for i in range(len(times_per_gen)):
            writer.writerow([times_per_gen[i]])
    
    print('--------- finish all jobs ---------')


if __name__ == '__main__':

    main()




