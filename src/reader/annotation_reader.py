import pandas as pd
import numpy as np

def cal_anno(data,n_class,output_binary=False):
    n_sample = (data.shape[0]+0.0)
    if n_class == 3:
        c1 = np.logical_or(data == 1 , data == 2)
        c2 = data == 3
        c3 = np.logical_or(data == 4 , data == 5)
        c1 = np.sum(c1,axis=0)/n_sample
        c2 = np.sum(c2,axis=0)/n_sample
        c3 = np.sum(c3,axis=0)/n_sample
        a = np.asarray([c1,c2,c3])
        #print a
        if output_binary:
            a[a>=(1/n_sample)] = 1
            a[a<(1/n_sample)] = 0
        #print a
        return a
    elif n_class == 5:
        c1 = data == 1
        c2 = data == 2
        c3 = data == 3
        c4 = data == 4
        c5 = data == 5

        c1 = np.sum(c1,axis=0)/n_sample
        c2 = np.sum(c2,axis=0)/n_sample
        c3 = np.sum(c3,axis=0)/n_sample
        c4 = np.sum(c4,axis=0)/n_sample
        c5 = np.sum(c5,axis=0)/n_sample
        a = np.asarray([c1,c2,c3,c4,c5])
        #print a
        if output_binary:
            a[a>=(1/n_sample)] = 1
            a[a<(1/n_sample)] = 0
        #print a
        return a

class reader:
    def __init__(self,path,name_list):
        self.path = path
        self.name_list = name_list
    def joint_read(self):
        all_df = []
        for name in self.name_list:
            df = self.__read_csv(self.path+name)
            all_df.append(df)
            print df.shape
        self.all_df = pd.concat(all_df).drop_duplicates().reset_index(drop=True)
        #return all_df
        #print all_df.groupby(['Image'])['Image is interesting'].count()
    def __read_csv(self,x):
        return pd.read_csv(x)

    def query(self):
        #['Respondent ID','Image is interesting','Image content is emotionally positive','Image is aesthetically pleasing']
        columns = ['Respondent ID','Image is interesting','Image content is emotionally positive',\
                   'Image is aesthetically pleasing','Is it professional and high-quality?',\
                   'Does it make you feel positive?','Include it in a Smart Album']
        df_groups = self.all_df[columns].groupby(self.all_df['Image'])
        anno_dict = {}
        user_dict = {}
        for k in df_groups.groups.keys():
            #print len(df_groups.get_group(k))
            user_id = list(df_groups[columns[0]].get_group(k))
            anno = []
            for i in range(1,7,1):
                anno.append( list(df_groups[columns[i]].get_group(k)) )
            anno = np.asarray(anno,dtype=np.uint8)
            anno = cal_anno(anno.T,3,output_binary=False)
            anno_dict[k] = anno
            user_dict[k] = user_id
            #print user_id
            #print anno.T,'lllll'
        return anno_dict,user_dict
    def query_byID(self,n_class):
        #['Respondent ID','Image is interesting','Image content is emotionally positive','Image is aesthetically pleasing']
        columns = ['Image','Image is interesting','Image content is emotionally positive',\
                   'Image is aesthetically pleasing','Is it professional and high-quality?',\
                   'Does it make you feel positive?','Include it in a Smart Album']
        df_groups = self.all_df[columns].groupby(self.all_df['Respondent ID'])
        anno_dict = {}
        user_ids = []
        for k in df_groups.groups.keys():
            #print len(df_groups.get_group(k))
            img_anno = {}
            img_ids = list(df_groups[columns[0]].get_group(k))
            user_ids.append([k,len(img_ids)])
            anno = []
            for i in range(1,7,1):
                anno.append( list(df_groups[columns[i]].get_group(k)) )
            anno = np.asarray(anno,dtype=np.uint8)
            for i,img_name in enumerate(img_ids):
                tmp_s = []
                for x in anno[:,i]:
                    if x==1 or x==2:
                        tmp_s.append(np.asarray([1,0,0]))
                    elif x==3:
                        tmp_s.append(np.asarray([0,1,0]))
                    elif x==4 or x==5:
                        tmp_s.append(np.asarray([0,0,1]))
                tmp_s = np.asarray(tmp_s)
                img_anno[img_name] = tmp_s.T

            #print img_anno
            anno_dict[k] = img_anno
            #user_dict[k] = user_id
            #print user_id
            #print anno.T,'lllll'
        return anno_dict,user_ids

    def query_all(self):
        #['Respondent ID','Image is interesting','Image content is emotionally positive','Image is aesthetically pleasing']
        columns = ['Image','Image is interesting','Image content is emotionally positive',\
                   'Image is aesthetically pleasing','Is it professional and high-quality?',\
                   'Does it make you feel positive?','Include it in a Smart Album']
        df_groups = self.all_df[columns].groupby(self.all_df['Respondent ID'])
        anno_all = []
        user_ids = []
        for k in df_groups.groups.keys():
            #print len(df_groups.get_group(k))
            user_ids.append(k)
            img_ids = list(df_groups[columns[0]].get_group(k))
            anno = []
            for i in range(1,7,1):
                anno.append( list(df_groups[columns[i]].get_group(k)) )
            anno = np.asarray(anno,dtype=np.uint8)
            for i,img_id in enumerate(img_ids):
                tmp_s = []
                for x in anno[:,i]:
                    if x==1 or x==2:
                        tmp_s.append(np.asarray([1,0,0]))
                    elif x==3:
                        tmp_s.append(np.asarray([0,1,0]))
                    elif x==4 or x==5:
                        tmp_s.append(np.asarray([0,0,1]))
                tmp_s = np.asarray(tmp_s)
                anno_all.append((k,img_id,tmp_s.T))
            #print anno.shape
        user_dict = {}
        for i,id in enumerate(user_ids):
            tmp = np.zeros((len(user_ids)))
            tmp[i] = 1
            user_dict[id] = tmp
        #print user_dict
        return anno_all, user_dict


if __name__ == "__main__":
    path = '/media/demcare/1.4T_Linux/SYNC/annotation/'
    r = reader(path,['DCU Insight Survey Results Sheet 1.csv','DCU Insight Survey Results Sheet 2.csv','surveys.csv'])
    r.joint_read()
    #anno,user = r.query()
    #anno,user = r.query_byID(3)
    #print user
    r.query_all()
