import numpy as np
import cPickle as pickle

def loadDPchallengefromPickle():
    full_list = pickle.load(open('preprocessing/DPChallenge.p','rb'))
    print 'DPchallenge 16K sample size',len(full_list)
    return full_list


def convert_type_DPChallenge(x):
    return x[0],int(x[1]),float(x[2]),np.asarray(x[3],dtype=np.float32)

def train_val_test(n_train,n_val,n_test):
    full_list = loadDPchallengefromPickle()
    if n_train + n_val + n_test != len(full_list):
        print 'Not correct partition'
    else:
        arr = np.arange(len(full_list))
        np.random.seed(5)
        #print arr
        np.random.shuffle(arr)
        train_list = [full_list[x] for x in arr[0:n_train]]
        val_list = [full_list[x] for x in arr[n_train:n_train+n_val]]
        test_list = [full_list[x] for x in arr[n_train+n_val:]]

    #print train_list[0]
    train_list = map(convert_type_DPChallenge,train_list)
    val_list = map(convert_type_DPChallenge,val_list)
    test_list = map(convert_type_DPChallenge,test_list)

    print 'train samples',len(train_list)
    print 'validation samples',len(val_list)
    print 'testing samples',len(test_list)
    return train_list,val_list,test_list

def train_val_test_top_bottom(n_train,n_val,n_test):
    from random import shuffle

    full_list = loadDPchallengefromPickle()
    n_negative = (n_train + n_val + n_test)/2
    n_positive = (n_train + n_val + n_test)/2
    if True:
        print "Percentage of samples taken:",(n_train + n_val + n_test)/(len(full_list)+0.0)
        np.random.seed(5)

        mean_list = np.asarray([float(x[2]) for x in full_list])
        sorted_ascent = np.argsort(mean_list)
        #print mean_list[sorted_ascent[-1]]
        negative_samples = [(full_list[i][0],[0,1]) for i in sorted_ascent[0:n_negative]]
        positive_samples = [(full_list[i][0],[1,0]) for i in sorted_ascent[-n_positive:]]
        #print len(negative_samples)
        #print len(positive_samples)
        arr = np.arange(n_negative)
        np.random.shuffle(arr)
        negative_train_list = [negative_samples[x] for x in arr[0:n_train/2]]
        negative_val_list = [negative_samples[x] for x in arr[n_train/2:n_train/2+n_val/2]]
        negative_test_list = [negative_samples[x] for x in arr[n_train/2+n_val/2:]]
        arr = np.arange(n_positive)
        np.random.shuffle(arr)
        positive_train_list = [positive_samples[x] for x in arr[0:n_train/2]]
        positive_val_list = [positive_samples[x] for x in arr[n_train/2:n_train/2+n_val/2]]
        positive_test_list = [positive_samples[x] for x in arr[n_train/2+n_val/2:]]
        train_list = negative_train_list + positive_train_list
        val_list = negative_val_list + positive_val_list
        test_list = negative_test_list + positive_test_list

        shuffle(train_list)
        shuffle(val_list)
        shuffle(test_list)
        return train_list,val_list,test_list

def split(anno,portion=[7,1,2]):
    pr_train = portion[0]/(sum(portion)+0.0)
    pr_val = portion[1]/(sum(portion)+0.0)
    #pr_test = portion[2]/(sum(portion)+0.0)
    name_list = [x for x in anno]
    name_list_i = range(len(anno))

    train_idx = [0,int(len(anno)*pr_train)]
    val_idx = [int(len(anno)*pr_train),int(len(anno)*(pr_train+pr_val))]
    test_idx = [int(len(anno)*(pr_train+pr_val)),len(anno)]

    np.random.seed(5)
    np.random.shuffle(name_list_i)
    train = [(name_list[i],anno[name_list[i]]) for i in name_list_i[train_idx[0]:train_idx[1]]]
    val = [(name_list[i],anno[name_list[i]]) for i in name_list_i[val_idx[0]:val_idx[1]]]
    test = [(name_list[i],anno[name_list[i]]) for i in name_list_i[test_idx[0]:test_idx[1]]]
    return train,val,test


if __name__ == "__main__":
    #train_val_test(10000,3000,3284)
    train_val_test_top_bottom(1600,400,1600)
    #readAVA.loadDPchallengefromPickle()
