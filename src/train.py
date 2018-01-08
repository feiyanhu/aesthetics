import pickle
import numpy as np
from model import aesthetic
from preprocessing import split


def image_iterator(name_list,batch_size):
    for x in range(0,len(name_list),batch_size):
        image_batch = []
        label_batch = []
        for name in name_list[x:x+batch_size]:
            im_path = dp_path+name[0]+'.jpg'
            im = aesthetic.image_cv_reader(im_path,(224,224))
            image_batch.append(im)
            label_batch.append(name[1])
        #print np.asarray(image_batch).shape,np.asarray(label_batch).shape
        yield np.asarray(image_batch),np.asarray(label_batch)



def train(epochs,batch_size,tr_list,val_list,new_model_name):
    model = aesthetic.model()
    model.compile(lr=1e-4)

    all_val_loss = []
    for epoch in range(epochs):
        print 'epoch',epoch
        epoch_loss = []
        batch_count = 0
        for x_tr,y_tr in image_iterator(tr_list,batch_size):
            batch_loss = model.batch_train(x_tr, y_tr)
            epoch_loss.append(batch_loss)
            batch_count = batch_count + 1
            if batch_count%5 == 0:
                print 'batch',batch_count,'batch loss', np.mean(epoch_loss)
            if batch_count%20 == 0:
                val_loss = []
                for x_val,y_val in image_iterator(val_list,batch_size):
                    val_batch_loss = model.batch_test(x_val, y_val)
                    val_loss.append(val_batch_loss)
                    #print val_batch_loss
                print 'validation loss is',np.mean(val_loss)
                all_val_loss.append(np.mean(val_loss))
                if np.mean(val_loss) == np.min(all_val_loss):
                    print 'save model with minimum validation loss'
                    model.model.save('../weights/'+new_model_name)
                if len(all_val_loss)>1:
                    if all_val_loss[len(all_val_loss)-1] == all_val_loss[len(all_val_loss)-2]:
                        c = input("End of training, Quit?")
                        if c == 'Y' or c == 'y':
                            exit()


if __name__ == "__main__":
    #dp_path = ''
    tr_list, val_list, test_list = split.train_val_test_top_bottom(1600,800,1600)
    train(10,10,tr_list,val_list,'new_model')
