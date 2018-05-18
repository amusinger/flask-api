import flask
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from matplotlib import colors as mcolors
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
    print("Loading text")
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
        if 'title' in flask.request.values:
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