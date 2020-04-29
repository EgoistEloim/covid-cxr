import numpy as np
import shap
import pandas as pd
import yaml
import os
import datetime
import dill
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from visualization.visualize import visualize_explanation
from predict import predict_instance, predict_and_explain
from data.preprocess import remove_text
from IPython import embed

def setup_shap():
    cfg = yaml.full_load(open('/home/paperspace/covid-cxr/config.yml', 'r'))
    shap_dict = {}
    shap_dict['TRAIN_SET'] = pd.read_csv(cfg['PATHS']['TRAIN_SET'])
    shap_dict['TEST_SET'] = pd.read_csv(cfg['PATHS']['TEST_SET'])
    shap_dict['MODEL'] = load_model(cfg['PATHS']['MODEL_TO_LOAD'], compile=False)

    test_img_gen = ImageDataGenerator(preprocessing_function=remove_text,
                                      samplewise_std_normalization=True, samplewise_center=True)
    test_generator = test_img_gen.flow_from_dataframe(dataframe=shap_dict['TEST_SET'],
                                                      directory=cfg['PATHS']['RAW_DATA'],
                                                      x_col="filename", y_col='label_str',
                                                      target_size=tuple(cfg['DATA']['IMG_DIM']), batch_size=1,
                                                      class_mode='categorical', validate_filenames=False, shuffle=False)
    train_img_gen = ImageDataGenerator(preprocessing_function=remove_text,
                                      samplewise_std_normalization=True, samplewise_center=True)
    train_generator = test_img_gen.flow_from_dataframe(dataframe=shap_dict['TRAIN_SET'],
                                                      directory=cfg['PATHS']['RAW_DATA'],
                                                      x_col="filename", y_col='label_str',
                                                      target_size=tuple(cfg['DATA']['IMG_DIM']), batch_size=1,
                                                      class_mode='categorical', validate_filenames=False, shuffle=False)
    shap_dict['TEST_GENERATOR'] = test_generator
    shap_dict['TRAIN_GENERATOR'] = train_generator
    return shap_dict

def shap_explain(shap_dict):

    model = shap_dict['MODEL']

    def map2layer(x, layer, model):
        feed_dict = dict(zip([model.layers[0].input.experimental_ref()], [x.copy()]))

        graph = tf.compat.v1.get_default_graph()
        print(graph.get_operations())
        with tf.compat.v1.Session() as sess:
            ret = sess.run(model.layers[layer], feed_dict)
        return ret

    preprocess_input = []
    for i in range(100):
        img, label = shap_dict['TEST_GENERATOR'].next()
        img = np.squeeze(img, axis=0)
        preprocess_input.append(img)
    inference = []
    for i in range(100):
        img, label = shap_dict['TEST_GENERATOR'].next()
        img = np.squeeze(img, axis=0)
        inference.append(img)
    preprocess_input = np.array(preprocess_input)
    inference = np.array(inference)
    e = shap.GradientExplainer((model.layers[0].input, model.layers[-1].output),
                               map2layer(preprocess_input.copy(), 0, model))
    shap_values, indexes = e.shap_values(map2layer(inference, 0, model), ranked_outputs=2)
    return shap_values, indexes

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    shap_dict = setup_shap()
    shap_values, indexes = shap_explain(shap_dict)
    shap.image_plot(shap_values, to_explain)
