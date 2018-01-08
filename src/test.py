from model import aesthetic
import numpy as np

if __name__ == "__main__":
    img_id = 'test.jpg'
    img = aesthetic.image_cv_reader(img_id,(224,224))
    img = np.asarray([img])
    model = aesthetic.model()
    #model.modify_last_layers(4,1024)
    print model.batch_predict(img)
