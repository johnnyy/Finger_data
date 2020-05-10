#!/usr/bin/env python
# coding: utf-8

# In[1]:


class Hash:
    def __init__(self, num_hashs, num_bits):
        self.text_in_hashs = []
        self.num_bits = num_bits
        self.num_hashs = num_hashs
        self.size_bucket = 2**num_bits

        self.text_in_hashs = list(map(lambda x:list(map(lambda x1: "",list(range(self.size_bucket)))),list(range(self.num_hashs))))
        
class Hash_Boolean:
    def __init__(self, num_hashs, num_bits):
        self.text_in_hashs = []
        self.num_bits = num_bits
        self.num_hashs = num_hashs
        self.size_bucket = 2**num_bits            
        self.logical_position = list(map(lambda x:list(map(lambda x1: False,list(range(self.size_bucket)))),list(range(self.num_hashs))))


# In[2]:


def generate_list_int(tuple_):
    line = tuple_[0]
    bits = tuple_[1]
    
    values_bit = wrap(line.replace(" ",""),bits)
#    print(len(values_bit))
    return list(map(lambda x1: int(x1,2),values_bit))


def set_value_in_hash(string_text,index,position):
    global hashs
    
    hashs.text_in_hashs[index][position]+=string_text
    
def generate_word(tuple_):
    list_index = tuple_[0]
    id = tuple_[1]
    id_min = tuple_[2]
    #id = 1
    return list(map(lambda pos_hash,index_hash: set_value_in_hash("{}_{} ".format(id,id_min),index_hash,pos_hash) if(pos_hash > 0) else None ,list_index,list(range(len(list_index)))))


# In[3]:



def gendata1(tuple_):
    list_pos_in_hash = tuple_[0]
    list_not_index = tuple_[1]
    
    global hashs
    # list_not_index = [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
    #    13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
    #    26,  27,  28,  29,  30,  31,  43,  50, 173, 180, 32,  33,  34,  39,  40,  41,  42,  51,  52,  61, 162, 171, 172,
    #   181, 182, 183, 184, 189, 190, 191]
    #list_not_index=[0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 21,
     #               25, 86, 90,16, 20, 91, 95]
        
  #  list_not_index= []
    for i,pos in enumerate(list_pos_in_hash):
        if(pos != 0 and i not in list_not_index):
            yield hashs.text_in_hashs[i][pos]
           # print(hashs.text_in_hashs[i][pos])
   # return list(map(lambda i: hashs.text_in_hashs[i[0]][i[1]] if (i[1] != 0 and i[0] not in list_not_index) else '' , enumerate(list_pos_in_hash)))
            
                            
                            
                            
                            
                            
      


# In[4]:



import numpy as np
import datetime
from textwrap import wrap
import os

def func_count_byKey_return_max(x):
    
    if(x != []):
      #  time1 = datetime.datetime.now()

        x_,y_ = np.unique(x,return_counts=True)
        y_max = np.max(y_)
        a = []
        for x,y in zip(x_,y_):
            if(y == y_max):
                #a.append(x.split('_')[0])
                yield x.split('_')[0]

       # a = " ".join(a)
      
    
    #a =  " ".join([ i[0].split('_')[0] for i in list(filter(lambda x: x[1] == y_max,list(zip(x_,y_)) )) ])
       # time2 = datetime.datetime.now()
        #print("time:", time2 - time1)
        #return time2 -time1
    #else:
     #   return ""
    
def func_rank_elements(persons):
    persons_list, count_ = np.unique(persons,return_counts=True)
    return persons_list[np.argsort(-1*count_)]


# In[5]:


from functools import reduce

def realize_rank(tuple_):
    bits = tuple_[1]
    line = tuple_[0]
    index_not_found = tuple_[2]
    #time1 = datetime.datetime.now()
    
    list_pos = generate_list_int((line,bits))
#    print(len(list_pos))

#    time2 = datetime.datetime.now()
 #   print("time:", time2 - time1)

  #  time1 = datetime.datetime.now()
 #   print(len(list_pos))
#    print(len(list_pos[0]))

    a = " ".join(list(gendata1((list_pos,index_not_found))))
   # time2 = datetime.datetime.now()
   # print("time1:", time2 - time1)
   # time1 = datetime.datetime.now()

    a = a.split()
        
   # time2 = datetime.datetime.now()
   # print("time2:", time2 - time1)
  #  time1 = datetime.datetime.now()

    a = list(func_count_byKey_return_max(a))
   # time2 = datetime.datetime.now()
   # print("time3:", time2 - time1)
    return a
    # return a


# In[ ]:



import datetime  
from functools import reduce
from itertools import chain
import multiprocessing
import pandas as pd

listNS = [18,16,12]
listND = [5,5,7]
list_quality = [0,15,20,30,40]
list_bits = [24]

pool_index = multiprocessing.Pool(4)
pool_insert = multiprocessing.Pool(4)
hashs = None
for ns,nd in zip(listNS,listND):
        
    for quality in list_quality:
        os.system("./Finger_data/source/generate_vector_python ~/Finger_data/file_min_index.txt  ~/Finger_data/Vector_Index/ {} {} {}".format(ns,nd,quality))
        os.system("./Finger_data/source/generate_vector_python ~/Finger_data/file_min_search.txt ~/Finger_data/Vector_Search/ {} {} {}".format(ns,nd,quality))
        print("    Vetores Gerados")

        for bits in list_bits:

            print("bits:", bits,"n_bits:",ns*ns*(nd+1),"hashs:",ns*ns*(nd+1)//bits, "quality:",quality )




            ###Indexação




            #hashs = None
            hashs = Hash(ns*ns*(nd+1)//bits,bits)
            print("    Created hash")


            for x,i,files in os.walk("Finger_data/Vector_Index/"):
                for ind,file in enumerate(files):
                    if(True):#ind >=2000):


                       # print(ind+1,file)
                        file_op = open(x+file)
                        vector_data = file_op.readlines() 
                        file_op.close()
                        list_positions = pool_index.map(generate_list_int,zip(vector_data,list(np.repeat(bits,len(vector_data)))))
                        id = int(file.split('.')[0])
                        len_list = len(list_positions)
                        #print(hashs)
                        #print(len(hashs.text_in_hashs))
                        list(map(generate_word, zip(list_positions, np.repeat(id,len_list), list(range(len_list)))))                          
                        
            print("    Indexed")
            
            
            
            ##Analise Index
            count_index_voids = []
            for i in range(ns*ns*(nd+1)//bits):
                #Range nas Hashs

                count_void_elements = 0


                for j in range(2**bits):
                    #Range nas posições da Hash


                    if(hashs.text_in_hashs[i][j] == ''):
                        count_void_elements += 1


                count_index_voids.append([count_void_elements,i])

            df = pd.DataFrame(count_index_voids,columns=['voids','index'])
            df.to_csv('Resultado_Preliminar_Indice/result{}_{}_{}.csv'.format(ns*ns*(nd+1),quality,bits),index=False)


            
            lista_vazia = df[df['voids'] ==2**bits ]['index'].values
            lista_com_1 = df[df['voids'] == 2**bits-1 ]['index'].values
            index_not_found = list(chain.from_iterable([lista_vazia,lista_com_1]))
            
            
            ###Search

            pool = multiprocessing.Pool(4)


            pos_rank =[]
            for x,i,files in os.walk("Finger_data/Vector_Search/"):
                count = 0
                for file in files:

                    if(True):#count >= 618):

                        file_op = open(x+file)
                        vector_data = file_op.readlines() 
                        file_op.close()
                        time1 = datetime.datetime.now()


                        a = pool.map(realize_rank,zip(vector_data,np.repeat(bits,len(vector_data)),np.repeat([index_not_found],len(vector_data),axis=0)))
                        
                        #a = map(realize_rank,zip(vector_data,np.repeat(bits,len(vector_data))))

                        persons = list(chain.from_iterable(a))

                        ranking = func_rank_elements(persons)
                        time2 = datetime.datetime.now()
                        id = file.split(".")[0]
                        pos_  = -1
                        for p in range(len(ranking)):
                            if(id  == ranking[p]):
                                pos_ = p + 1
                                break
                        print("count:",count,"filename:",file,"pos_rank:",pos_,"time:",time2-time1)
                        pos_rank.append([pos_,file,time2-time1])
                        
                    count+=1
                    
                    
            pool.close()
            

            df = pd.DataFrame(pos_rank,columns=['pos_rank','file','time'])
            df.to_csv('Resultado_Preliminar/result_query{}_{}_{}.csv'.format(ns*ns*(nd+1),quality,bits),index=False)





# In[ ]:




