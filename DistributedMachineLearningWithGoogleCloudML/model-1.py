
# Insert the following code below the linear_model function, above the definition for the serving_input_fn:
def create_embed(sparse_col):
    dim = 10 # default
    if hasattr(sparse_col, 'bucket_size'):
       nbins = sparse_col.bucket_size
       if nbins is not None:
          dim = 1 + int(round(np.log2(nbins)))
    return tflayers.embedding_column(sparse_col, dimension=dim)

# Add the following below the create_embed function you just added:
def dnn_model(output_dir):
    real, sparse = get_features()
    all = {}
    all.update(real)

    # create embeddings of the sparse columns
    embed = {
       colname : create_embed(col) \
          for colname, col in sparse.items()
    }
    all.update(embed)

    estimator = tflearn.DNNClassifier(
         model_dir=output_dir,
         feature_columns=all.values(),
         hidden_units=[64, 16, 4])
    estimator = tf.contrib.estimator.add_metrics(estimator, my_rmse)
    return estimator

# Page down to the run_experiment function at the end of the file. Replace the line estimator = linear_model(output_dir)with:
#estimator = linear_model(output_dir)
estimator = dnn_model(output_dir)