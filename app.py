import flask
from flask import Flask
import sys, os
import numpy as np
from keras.models import Model
from keras.applications import VGG19
from keras.preprocessing import image
from PIL import Image
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from sklearn.neighbors import NearestNeighbors
import io
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from matplotlib import colors as mcolors
from flask import send_file
import json

app = flask.Flask(__name__)
global graph, y_pred, sess,classes,knn, n_neighbours
global tfidf, id_to_category, my_model, similar_model

def load_tf_model():
    print("Loading TF model")
    model_dir = "./mymodel.meta"
    global sess,graph,y_pred, classes
    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_dir)
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    graph = tf.get_default_graph()

    y_pred = graph.get_tensor_by_name("y_pred:0")

    classes = ['bags>>backpack', 'hats>>baseball cap',  'women>>top>>blouse_shirt', 'top>>coat',
               'hat>>fedora','women>>dress>>floor_length_dress', 'bags>>handbag', 'top>>hoodie', 'bottom>>jeans_women',
           'accessories>>jewellery', 'women>>dress>>knee_length_dress', 'top>>leather_jacket',
           'women>>bottom>>skirt', 'shoes>>sneakers', 'socks', 'man>>suit_jacket', 'accessories>>sunglasses',
           'top>>tshirt', 'hat>>beanie', 'shoes>>heels']

def predict_image_class(path):
    global sess, graph, y_pred, classes
    images = []
    filename = path
    image = Image.open(filename).convert('RGB')
    image = image.resize((128, 128), Image.BILINEAR)
    image = np.multiply(image, 1.0 / 255.0)

    images.append(image)
    images = np.array(images)
    images = np.float32(images)

    x_batch = images.reshape(1, 128, 128, 3)

    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, 20))
    y_test_images = np.float32(y_test_images)

    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)

    return result



def load_text_model():
    print("Loading linear regression")
    df = pd.read_csv('index.csv')
    col = ['Category', 'Title']
    df = df[col]
    df = df[pd.notnull(df['Title'])]
    df.columns = ['Category', 'Title']
    df['category_id'] = df['Category'].factorize()[0]
    category_id_df = df[['Category', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    global tfidf, id_to_category
    id_to_category = dict(category_id_df[['category_id', 'Category']].values)
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(df.Title).toarray()
    labels = df.category_id
    pkl_filename = "text_model.pkl"
    with open(pkl_filename, 'rb') as file:
        global my_model
        my_model = pickle.load(file)
    global colors
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    colors = list(colors.keys())
    colors = colors[8:]


def get_class_color(txt):
    preds = []
    text = []
    text.append(txt)
    global tfidf, my_model, id_to_category
    text_features = tfidf.transform(text)
    predictions = my_model.predict(text_features)
    for predicted in predictions:
        print("Predicted as: '{}'".format(id_to_category[predicted]))
        preds.append(id_to_category[predicted])
    keys = []
    text = text[0].lower()
    for col in colors:
        if col in text:
            keys.append(col)
    if len(keys)>0:
         return preds, keys
    else:
        return preds

def load_features_knn():
    print("Loading features..")
    features = np.loadtxt("./short.txt", delimiter=",")
    global n_neighbours, knn
    n_neighbours = 6
    print("KNN")
    knn = NearestNeighbors(n_neighbors=n_neighbours, algorithm="brute", metric="cosine")
    knn.fit(features)
    print(knn)
    print(features[:5])

def load_client_model():
    print("Loading model.. ")
    base_model = VGG19(weights='imagenet')
    global similar_model
    similar_model = Model(input=base_model.input, output=base_model.get_layer('block4_pool').output)
    print(similar_model.summary())


def prepare_image(path):
    print("Preparing image..")

    image = Image.open(path)
    image = image.resize((224,224))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

     # return the processed image
    return image

def find_topk_unique(indices, distances, k):

    # Sort by ascending distance
    i_sort_1 = np.argsort(distances)
    distances_sorted = distances[i_sort_1]
    indices_sorted = indices[i_sort_1]

    window = np.array(indices_sorted[:k], dtype=int)  # collect first k elements for window intialization
    window_unique, j_window_unique = np.unique(window, return_index=True)  # find unique window values and indices
    j = k  # track add index when there are not enough unique values in the window
    # Run while loop until window_unique has k elements
    while len(window_unique) != k:
        # Append new index and value to the window
        j_window_unique = np.append(j_window_unique, [j])  # append new index
        window = np.append(window_unique, [indices_sorted[j]])  # append new value
        # Update the new unique window
        window_unique, j_window_unique_temp = np.unique(window, return_index=True)
        j_window_unique = j_window_unique[j_window_unique_temp]
        # Update add index
        j += 1

    # Sort the j_window_unique (not sorted) by distances and get corresponding
    # top-k unique indices and distances (based on smallest distances)
    distances_sorted_window = distances_sorted[j_window_unique]
    indices_sorted_window = indices_sorted[j_window_unique]
    u_sort = np.argsort(distances_sorted_window)  # sort

    distances_top_k_unique = distances_sorted_window[u_sort].reshape((1, -1))
    indices_top_k_unique = indices_sorted_window[u_sort].reshape((1, -1))

    return indices_top_k_unique, distances_top_k_unique

def find_similar(path):
    imgs_test = []

    img_orig = image.load_img(path, target_size=(224, 224))  # load
    imgs_test.append(np.array(img_orig))

    global similar_model, knn, n_neighbours
    print(similar_model.summary())
    print(knn)
    img_test_ar = image.img_to_array(img_orig)
    img_test = np.expand_dims(img_test_ar, axis=0)
    img_test = imagenet_utils.preprocess_input(img_test)
    print(img_test)
    img_test_features = similar_model.predict(img_test).flatten()

    distances, indices = knn.kneighbors(np.array([img_test_features]), return_distance=True)
    distances = distances.flatten()
    indices = indices.flatten()
    indices, distances = find_topk_unique(indices, distances, n_neighbours)
    return indices, distances

@app.route("/similar", methods=["POST"])
def predict_similar():
    # initialize the data dictionary that will be returned from the
    # view
    data = {}
    data["success"] = False

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        data = {"post": True}
        if 'image' in flask.request.files:
            data['image_parsing'] = True
            temp_dir = './uploads'

            # file = flask.request.files['image']
            # data['file'] = True
            # upload_path = os.path.join(temp_dir, file.filename)
            # file.save(upload_path)

            f = flask.request.files['image']

            indices, distances = find_similar(f)

            images_dir = os.listdir('./short')
            images = []
            for index, file in enumerate(images_dir):
                if (index in indices[0]):
                    img = image.load_img(os.path.join(f, file))
                    images.append(np.array(img))
                    r = {"name": file, "index": index}
                    data["predictions"].append(r)


    return flask.jsonify(data)


@app.route("/predict", methods=["POST"])
def predict_image():
    data ={}
    data["success"] =  False

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        data['post'] = True
        if 'image' in flask.request.files:
            data['image_parsing'] = True


            f = flask.request.files['image']
            result = predict_image_class(f)


            top_k = result[0].argsort()[-len(result[0]):][::-1]
            data["predictions"] = []
            for node_id in top_k:
                human_string = classes[node_id]
                score = 1*result[0][node_id]
                print('%s (score = %.5f)' % (human_string, score))
                data["predictions"].append((str(human_string), str(score)))

            data["success1"] = True
        txt = str(flask.request.values['title'])
        print(txt)
        data["success2"] = True
        if txt == "Title":
            data["success2"] = False

        if  data["success2"] and txt:
            # data['text_parsing'] = flask.request.data['title']
            txt1 = flask.request.values["title"]

            print("im here", txt1)
            cat, col = get_class_color(txt1)
            data["category_title"] = cat
            data["color"] = col

            data["success2"] = True


    # return the data dictionary as a JSON response
    return json.dumps(data)


if __name__ == '__main__':
    print(("Please wait until server has fully started"))
    # load_client_model()
    load_tf_model()
    load_text_model()
    # load_features_knn()
    app.run()