# Add the following two functions above the linear_model function to implement a wide and deep neural network model:
def parse_hidden_units(s):
    return [int(item) for item in s.split(',')]

def wide_and_deep_model(output_dir,nbuckets=5,
                        hidden_units='64,32', learning_rate=0.01):
    real, sparse = get_features()
    # lat/lon cols can be discretized to "air traffic corridors"
    latbuckets = np.linspace(20.0, 50.0, nbuckets).tolist()
    lonbuckets = np.linspace(-120.0, -70.0, nbuckets).tolist()
    disc = {}
    disc.update({
       'd_{}'.format(key) : \
           tflayers.bucketized_column(real[key], latbuckets) \
           for key in ['dep_lat', 'arr_lat']
    })
    disc.update({
       'd_{}'.format(key) : \
           tflayers.bucketized_column(real[key], lonbuckets) \
           for key in ['dep_lon', 'arr_lon']
    })
    # cross columns that make sense in combination
    sparse['dep_loc'] = tflayers.crossed_column( \
           [disc['d_dep_lat'], disc['d_dep_lon']],\
           nbuckets*nbuckets)
    sparse['arr_loc'] = tflayers.crossed_column( \
           [disc['d_arr_lat'], disc['d_arr_lon']],\
           nbuckets*nbuckets)
    sparse['dep_arr'] = tflayers.crossed_column( \
           [sparse['dep_loc'], sparse['arr_loc']],\
           nbuckets ** 4)
    sparse['ori_dest'] = tflayers.crossed_column( \
           [sparse['origin'], sparse['dest']], \
           hash_bucket_size=1000)

    # create embeddings of all the sparse columns
    embed = {
       colname : create_embed(col) \
          for colname, col in sparse.items()
    }
    real.update(embed)

    #lin_opt=tf.train.FtrlOptimizer(learning_rate=learning_rate)
    #l_rate=learning_rate*0.25
    #dnn_opt=tf.train.AdagradOptimizer(learning_rate=l_rate)
    estimator = tflearn.DNNLinearCombinedClassifier(
         model_dir=output_dir,
         linear_feature_columns=sparse.values(),
         dnn_feature_columns=real.values(),
         dnn_hidden_units=parse_hidden_units(hidden_units))
         #linear_optimizer=lin_opt,
         #dnn_optimizer=dnn_opt)
    estimator = tf.contrib.estimator.add_metrics(estimator, my_rmse)
    return estimator

# Page down to the run_experiment function at the end of the file. You need to reconfigure the experiment to call the deep neural network estimator function instead of the linear classifier function.
# Replace the line estimator = dnn_model(output_dir) with:
  #estimator = linear_model(output_dir)
# estimator = dnn_model(output_dir)
estimator =  wide_and_deep_model(output_dir, 5, '64,32', 0.01)