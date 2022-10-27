from enum import Flag
from re import L
import pandas as pd
import numpy as np

df = pd.read_csv('./ml-latest-small/ratings.csv')

#intializing
minimum_support = 0.1
minumum_confidence = 0.1



#Preprocessing
df = df[['userId','movieId','rating']]
df['movieId'] = df['movieId'].apply(lambda x: str(x))

df.drop(df.loc[df['rating']<3].index, inplace=True)
count=df['userId'].value_counts()
df_1 = df[df['userId'].isin(count[count > 10].index)]
df_1 = df_1.reset_index()
df_1 = df_1[['userId','movieId','rating']]

#Training, Test Split
training=np.array([])
test = np.array([])

for i in df_1['userId'].unique():
    sample = df_1[df_1['userId']==i].index
    s1 = sample.to_numpy()
    k = len(s1)
    l = int(0.8*k)
    np.random.shuffle(s1)
    training_1, test_1 = s1[:l], s1[l:]
    training = np.concatenate([training,training_1])
    test = np.concatenate([test,test_1])

training_list_indices = training.astype(int).tolist()
test_list_indices=test.astype(int).tolist()

training_df = df_1.iloc[training_list_indices]
test_df=df_1.iloc[test_list_indices]

Unique_List= (training_df['movieId'].append(training_df['movieId'])).unique() # USed this later.
u_list = Unique_List.tolist()

#used this later


# converting the training Df to list of lists to be used for algorithm
train_list = training_df.groupby(by = ["userId"])["movieId"].apply(list).reset_index()
train_list = train_list[['movieId']]
data = train_list["movieId"]
d1=data.to_list()
d1

# converting the test Df to list of lists to be used for algorithm
test_list = test_df.groupby(by = ["userId"])["movieId"].apply(list).reset_index()
test_list = test_list[['movieId']]
data1 = test_list["movieId"]
d2=data1.to_list()
d2

total = len(d1)
total1 = len(d2)

# to count the occurances of the set
def count_values(Items, moviedata):
    count = 0
    for i in range(len(moviedata)):
        if set(Items).issubset(set(moviedata[i])):
            count+=1
    return count

# To get the support of the set
def support(items, moviedata):
    count = 0
    for i in range(len(moviedata)):
        if set(items).issubset(set(moviedata[i])):
            count+=1
    numertaor = count
    denominator = len(moviedata)
    support = numertaor/denominator
    return support

#to get the confidence of the set
def confidence(item1,item2,moviedata):
    count1 = 0
    for i in range(len(moviedata)):
        if set(item2).issubset(set(moviedata[i])):
            count1+=1
    count2 = 0
    for i in range(len(moviedata)):
        if set(item1).issubset(set(moviedata[i])):
            count2+=1
    
    num = count1
    den = count2
    conf = num/den
    return conf

# Join two item sets condition checker
def join_two(item1,item2):
    item1.sort(key = lambda x: u_list.index(x))
    item2.sort(key = lambda x: u_list.index(x))
    for i in range(len(item1)-1):
        if item1[i] != item2[i]:
            return []
    o1 = u_list.index(item1[-1])
    o2 = u_list.index(item2[-1])
    if o1<o2:
        k =[item2[-1]]    
        return item1+k
    return []

#Join two itemsets main function
def join_set(items):
    c1 = []
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            item1 = items[i]
            item2 = items[j]
            joined_itemset = join_two(item1, item2)
            if len(joined_itemset)>0:
                c1.append(joined_itemset)
    
    return c1

#To get frequent itemsets
def Frequent_itemsets(items, moviedata):
    Frequent_sets = []
    support_value = []
    l1 = len(items)
    for i in range(l1):
        sup = support(items[i],moviedata)
        if sup >=minimum_support:
            Frequent_sets.append(items[i])
            support_value.append(sup)
    return Frequent_sets, support_value

#Print any support table at any needed iteration.
def print_table(T_table,support_value):
    with open('demo.txt', 'w') as f:
        print("Movieset -- support",file = f)
        l1 = len(T_table)
        for i in range(l1):
            print("{} : {}".format(T_table[i], support_value[i]),file = f)
        print("\n\n", file=f)
        f.close()



#start of algorithm and initialization
c_table ={}
L_table ={}
support_count_table = {}
i_size = 1

c_table.update({i_size: [[f]for f in Unique_List]})
#1 frquent itemsets
L_list,sup_value = Frequent_itemsets(c_table[i_size], d1)
L_table.update({i_size : L_list})
support_count_table.update({i_size : sup_value})

#Loop for findind the most frequent itemsets of highest order.
k = i_size+1

def Final_Frequentitemsets(d1,k):
    Flag = False
    while not Flag:
        c_table.update({k : join_set(L_table[k-1])})
        print("Table c_table{}: \n".format(k))
        #print_table(c_table[k], [count_values(i, d1) for i in c_table[k]])
        L_list,sup = Frequent_itemsets(c_table[k],d1)
        L_table.update({k : L_list})
        support_count_table.update({k: sup})
        if len(L_table[k])==0:
            Flag = True
        else:
            print("Table L_table{}: \n".format(k))
           # print_table(L_table[k], support_count_table[k])
        # print(L_table[k])
        k+=1

Final_Frequentitemsets(d1,k)


length = len(L_table)-1
## Final_Rules Extraction.
def apriori(L,length,data_input):
    temp = L[length]
    output = {}
   # df_ouput_1 = pd.DataFrame(columns=['X','Y','Confidence','support'])
    X = []
    Y = []
    confidence_V = []
    support_V = []
    l1 = len(L[length])
    for i in range(0,l1):
        current = temp[i]
        l2 = len(current) 
        for j in range(0,l2):
            mov1 = [current[j]]
            conf = confidence(mov1,current,data_input)
            sup = support(current,data_input)
            if conf>minumum_confidence:
                mov2 =[x for x in current if x not in mov1]
                X.append(mov1)
                Y.append(mov2)
                confidence_V.append(conf)
                support_V.append(sup)
                #df_ouput.insert(loc=i+j,column='X',value=conf)
    len1 = len(X)
    df_ouput = pd.DataFrame(list(zip(X,Y,confidence_V,support_V)),columns=['X','Y','Confidence','support'])
    with open('Team37_Associationrules.txt', 'w') as f:
        print("X-Y-Conf-Supp",file=f)
        for i in range(0,len1):
            # print("{} : {}".format(T_table[i], support_value[i])
            print("{}->{}".format(X[i],Y[i],),file=f)
            print(":{}:{}".format(confidence_V[i],support_V[i]),file =f)
            print("\n")
    f.close()                    
    return df_ouput

df_ouput=apriori(L_table,length,d1)
#print(df_ouput)


# recommender
#if len(df_ouput)>=100:
df1 = df_ouput.sort_values(by=['Confidence']).head(100)
df2 = df_ouput.sort_values(by=['support']).head(100)
df1 = df1[['X','Y']]
df2 = df2[['X','Y']]
#df1['X'] = df1['X'].apply(lambda x : [x])
#df2['X'] = df2['X'].apply(lambda x : [x])
with open('Team37_top100rulesBySupport.txt', 'w') as f:
    print(df2,file=f)
    f.close()
with open('Team37_Top100RulesByConf.txt','w') as f:
    print(df1,file=f)
    f.close()
df3 = df1.merge(df2, how='inner',indicator=False).sort_values(by=['Confidence'])







    


