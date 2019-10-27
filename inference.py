import keras
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from keras import optimizers
from sklearn.metrics import precision_recall_fscore_support

def inference(csv_file):
    json_file = open('model_isolation.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model_wieghts_isolation.h5")
    df = pd.read_csv(csv_file, header=None, names=["id", "label"], dtype=str)
    df = df.replace({'true':'isolation','false':'noaction', 'True':'isolation', 'False':'noaction'})
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_generator=test_datagen.flow_from_dataframe(
        dataframe=df,
        directory="./",
        x_col="id",
        y_col="label",
        class_mode="categorical",
        batch_size=1,
        target_size=(320,180))
    loaded_model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])
    result1 = loaded_model.predict_generator(valid_generator, steps=valid_generator.samples)
    result = []
    result1 = result1.argmax(axis=1)
    for i in result1:
        if i == 0:
            result.append('isolation')
        else:
            result.append('noaction')
    accuracy = (df['label'].values ==result).mean()
    ans = precision_recall_fscore_support(df['label'].values, result, average='macro')
    return {'accuracy':accuracy, 'recall':ans[1], 'precision':ans[0]}
