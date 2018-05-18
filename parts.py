# im = Image.open(f)
#
# imgs_test = []
#
# # Read image file
# img_orig = image.load_img(f, target_size=(224, 224))  # load
# imgs_test.append(np.array(img_orig))
#
# # Pre-process for model input
# img_test_ar = image.img_to_array(img_orig)  # convert to array
# img_test = np.expand_dims(img_test_ar, axis=0)
# img_test = imagenet_utils.preprocess_input(img_test)
# print(similar_model)
# # img_test_features = similar_model.predict(img_test).flatten()  # features
# img_test_features = similar_model.predict(img_test)
# # Find top-k closest image feature vectors to each vector
# distances, indices = knn.kneighbors(np.array([img_test_features]), return_distance=True)
# distances = distances.flatten()
# indices = indices.flatten()
# indices, distances = find_topk_unique(indices, distances, n_neighbours)
# print(indices)
# images_path = "./short/"
# images_dir = os.listdir(images_path)
# images = []
# data["predictions"] = []
# for index, file in enumerate(images_dir):
#     if (index in indices[0]):
#         img = image.load_img(os.path.join(images_path, file))
#         images.append(np.array(img))
#         r = {"name": file, "index": index}
#         data["predictions"].append(r)
#
# # indicate that the request was a success
# data["success"] = True
# return send_file(images, mimetype='image/jpg')
# return the data dictionary as a JSON response
# in_memory_file = io.BytesIO()
# file.save(in_memory_file)
# data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
# color_image_flag = 1
# img = cv2.imdecode(data, color_image_flag)
# img_received = file.read()
# img_opened = Image.open(upload_path)
# bin_file = Image.open(io.BytesIO(img_opened.read()))
# img_proc = prepare_image(upload_path)
# print(img_proc)

# image = Image.open(upload_path)
# # resize the input image and preprocess it
# image = image.resize((224, 224))
# # if the image mode is not RGB, convert it
# if image.mode != "RGB":
#     image = image.convert("RGB")
# image = img_to_array(image)
# image = np.expand_dims(image, axis=0)
# image = imagenet_utils.preprocess_input(image)

