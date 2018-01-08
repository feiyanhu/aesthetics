import model.aesthetic as aesth
import reader.annotation_reader as anno
import pickle
import preprocessing.split as split
import numpy as np
from sklearn.metrics import confusion_matrix


def iterator(name_list,batch_size=12):
    #all_name_list, all_data = read_all_images()
    for i in range(0,len(name_list),batch_size):
        tmp_data = []
        tmp_label = []
        tmp_name_list = name_list[i:i+batch_size]
        for name,label in tmp_name_list:
            img_path = path + 'insight/' + name
            img = aesth.image_cv_reader(img_path,(224,224))
            tmp_data.append(img)
            tmp_label.append(label)
        tmp_data = np.asarray(tmp_data)
        tmp_label = np.asarray(tmp_label)
        #print tmp_label.shape,tmp_data.shape
        #labels = name_list[1]
        yield tmp_data,tmp_label


def train(train_data,val_data,ii,new_name):
    epochs = 5
    print len(train_data),len(val_data)
    model = aesth.model()
    '''for layer in model.model.layers:
        layer.trainable = True'''
    model.modify_last_layers(2048,3)
    model.compile(lr=1e-3)
    all_val_loss = []
    for epoch in range(epochs):
        print 'epoch',epoch
        epoch_loss = []
        batch_count = 0
        for tr_x,tr_y in iterator(train_data):
            #print tr_x.shape
            batch_loss = model.batch_train(tr_x, tr_y)
            #print batch_loss
            epoch_loss.append(batch_loss)
            batch_count = batch_count + 1
            if batch_count%5 == 0:
                print 'batch',batch_count,'batch loss', np.mean(epoch_loss)
            if batch_count%20 == 0:
                val_loss = []
                for x_val,y_val in iterator(val_data):
                    val_batch_loss = model.batch_test(x_val, y_val)
                    val_loss.append(val_batch_loss)
                    #print val_batch_loss
                print 'validation loss is',np.mean(val_loss)
                all_val_loss.append(np.mean(val_loss))
                if np.mean(val_loss) == np.min(all_val_loss):
                    print 'save model with minimum validation loss'
                    model.model.save('../weights/'+new_name+str(ii))

def test(test_data,ii):
    model = aesth.model()
    model.modify_last_layers(2048,3)
    model.model.load_weights('../weights/sync_anno_'+str(ii))
    p_all = []
    p_hat_all = []
    for te_x,te_y in iterator(test_data,80):
        a = model.batch_predict_raw(te_x)
        #print a.shape
        p_hat = np.argmax(a,axis=1)
        #print p_hat
        #print te_y
        p = np.argmax(te_y,axis=1)
        p_all.extend(p.tolist())
        p_hat_all.extend(p_hat.tolist())
    p_all = np.asarray(p_all)
    p_hat_all = np.asarray(p_hat_all)
    #print p_all
    a = confusion_matrix(p_all,p_hat_all)
    print a

def process_labels(anno,n):
    print len(anno)
    tmp_dict = {}
    #all_d = []
    for k in anno:
        data = anno[k]
        data = data[:,n]
        if np.max(data)>0.5:
            tmp = np.zeros((3))
            tmp_i = np.argmax(data)
            tmp[tmp_i] = 1
            tmp_dict[k] = tmp
    return tmp_dict

if __name__ == "__main__":
    path = '/media/demcare/1.4T_Linux/SYNC/'
    r = anno.reader(path+'annotation/',['DCU Insight Survey Results Sheet 1.csv','DCU Insight Survey Results Sheet 2.csv','surveys.csv'])
    r.joint_read()
    anno,user = r.query()

    for i in range(6):
        #print len(anno_tmp)
        anno_tmp = process_labels(anno,i)
        train_data,val_data,test_data = split.split(anno_tmp)
        print len(train_data),len(val_data)
        #exit()
        #train(train_data,val_data,i)
        test(test_data,i)
