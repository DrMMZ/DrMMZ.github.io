def conv(x, num_filter, filter_size, stride, padding, layer):
  """
  A convolution layer with batch normalization and ReLU.

  Inputs:
  -x: tensor of shape (N,H,W,C)
  -num_filter: integer, number of filters
  -filter_size: integer, filter size
  -stride: integer, number of strides
  -padding: string, "valid" or "same"
  -layer: string, layer's name

  Output:
  -x: tensor of shape (N, 1+((H+2p-filter_size)//stride), 1+((W+2p-filter_size)//stride), num_filter)
  where p is the number of padding on one side 
  """
  x = tf.keras.layers.Conv2D(num_filter, filter_size, stride, padding, name=layer+'_conv')(x)
  x = tf.keras.layers.BatchNormalization(name=layer+'_bn')(x)
  x = tf.keras.layers.Activation('relu')(x)
  return x


def conv_1x1(x, num_filter, layer):
  """
  A 1 by 1 convolution layer.

  Inputs:
  -x: tensor of shape (N,H,W,C)
  -num_filter: integer, number of filters
  -layer: string, layer's name

  Output:
  -x: tensor of shape (N,H,W,num_filter)
  """
  x = conv(x, num_filter, filter_size=1, stride=1, padding='same', layer=layer+'_1x1')
  return x


def reduce_conv(x, num_filter_1, num_filter_2, filter_size, stride, layer):
  """
  A 1 by 1 (bottleneck) and 3 by 3 or 5 by 5 convolution layers.

  Inputs:
  -x tensor of shape (N,H,W,C)
  -num_filter_1: number of filters in bottleneck
  -num_filter_2: number of filters in convolution layer
  -filter_size: integer in {3,5}, filter size for convolution layer
  -stride: integer in {1,2}, number of strides for convolution layer
  -layer: string, layer's name

  Output:
  -x: tensor of shape (N,H,W,num_filter_2)
  """
  x = conv(x, num_filter_1, 1, stride=1, padding='same', layer=layer+'_bottle')

  if stride == 1:
    x = conv(x, num_filter_2, filter_size, stride, padding='same', layer=layer)

  elif stride == 2:
    if filter_size == 3:
      x = tf.keras.layers.ZeroPadding2D(1)(x)
      x = conv(x, num_filter_2, filter_size, stride, padding='valid', layer=layer)
    elif filter_size == 5:
      x = tf.keras.layers.ZeroPadding2D(2)(x)
      x = conv(x, num_filter_2, filter_size, stride, padding='valid', layer=layer)

  return x


def pool_project(x, num_filter, method, layer):
  """
  A max or average pooling layer and a 1 by 1 convolution layer.

  Inputs:
  -x: tensor of shape (N,H,W,C)
  -num_filter: integer, number of filters in convolution layer
  -method: string, "max" or "avg" for pooling method
  -layer, string, layer's name

  Output:
  -x: tensor of shape (N,H,W,num_filter)
  """
  if method == 'max':
    x = tf.keras.layers.MaxPool2D(3, strides=1, padding='same')(x)
  elif method == 'avg':
    x = tf.keras.layers.AveragePooling2D(3, strides=1, padding='same')(x)

  x = conv(x, num_filter, 1, stride=1, padding='same', layer=layer+'_pool')

  return x
  
  
def inception_module(x, num_filters, stride, method, layer):
  """
  Implement the inception module: 

  conv_1x1(), 3x3 reduce_conv(), 5x5 reduce_conv() and pool_project()

  where conv_1x1(), 5x5 reduce_conv() and pool_project() are optional.

  Inputs:
  -x: tensor of shape (N,H,W,C)
  -num_filters: dictionary of number of filters with keys '1x1', 'bottle_3x3', 
  '3x3', 'bottle_5x5', '5x5' and 'pool'; if the value is None, then the inception 
  module won't proceed that operation; however, if the value of 'pool' is None, 
  it will proceed maxpooling with a stride of 2
  -stride: integer in {1,2}, number of strides in 3 by 3 or 5 by 5
  -method: string, "max" or "avg" for pooling method
  -layer, string, layer's name

  Output:
  -x: tensor of shape 
  (N, H//stride, W//stride, num_filters['1x1']+num_filters['3x3']+num_filters['5x5']+num_filters['pool'])
  """
  if num_filters['1x1'] != None:
    x_1x1 = conv_1x1(x, num_filters['1x1'], layer)

  x_3x3 = reduce_conv(x, num_filters['bottle_3x3'], num_filters['3x3'], 3, stride, layer=layer+'_3x3')

  if num_filters['5x5'] != None:
    x_5x5 = reduce_conv(x, num_filters['bottle_5x5'], num_filters['5x5'], 5, stride, layer=layer+'_5x5')

  if num_filters['pool'] != None:
    x_pool = pool_project(x, num_filters['pool'], method, layer)
  elif num_filters['pool'] == None:
    x_pool = tf.keras.layers.ZeroPadding2D(1)(x)
    x_pool = tf.keras.layers.MaxPool2D(3, strides=2, padding='valid')(x_pool)

  if stride == 1 and num_filters['5x5'] != None:
    x = tf.keras.layers.concatenate([x_1x1, x_3x3, x_5x5, x_pool], axis=3)
  elif stride == 1 and num_filters['5x5'] == None:
    x = tf.keras.layers.concatenate([x_1x1, x_3x3, x_pool], axis=3)
  elif stride == 2:
    x = tf.keras.layers.concatenate([x_3x3, x_5x5, x_pool], axis=3)

  return x
  
  
def FaceNet(input_shape=(96,96,3)):
  """
  Implement FaceNet, which encodes an image to a 128-dimensional vector followed
  by L2 normalization.

  Input:
  -input_shape: input shape of the image (96,96,3), channel last

  Output:
  -model: TensorFlow keras model
  """
  X_input = tf.keras.Input(input_shape)

  # block 1, conv, (48,48,64)
  X = tf.keras.layers.ZeroPadding2D(3)(X_input)
  X = conv(X, 64, 7, 2, 'valid', '1')

  # block 1, maxpool, (24,24,64)
  X = tf.keras.layers.ZeroPadding2D(1)(X)
  X = tf.keras.layers.MaxPool2D(3, 2)(X)

  # block 2, inception, (24,24,192)
  X = conv(X, 64, 1, 1, 'same', '2_3x3_bottle')
  X = conv(X, 192, 3, 1, 'same', '2_3x3')

  # block 2, maxpool, (12,12,192)
  X = tf.keras.layers.ZeroPadding2D(1)(X)
  X = tf.keras.layers.MaxPool2D(3, 2)(X)

  # block 3a, inception, (12,12,256)
  X = inception_module(
      X, num_filters={'1x1':64, 'bottle_3x3':96, '3x3':128, 'bottle_5x5':16, '5x5':32, 'pool':32},
      stride=1, method='max', layer='3a')
  # block 3b, inception, (12,12,320)
  X = inception_module(
      X, num_filters={'1x1':64, 'bottle_3x3':96, '3x3':128, 'bottle_5x5':32, '5x5':64, 'pool':64},
      stride=1, method='avg', layer='3b')
  # block 3c, inception, (6,6,640)
  X = inception_module(
      X, num_filters={'1x1':None, 'bottle_3x3':128, '3x3':256, 'bottle_5x5':32, '5x5':64, 'pool':None},
      stride=2, method='max', layer='3c')
  
  # block 4a, inception, (6,6,640)
  X = inception_module(
      X, num_filters={'1x1':256, 'bottle_3x3':96, '3x3':192, 'bottle_5x5':32, '5x5':64, 'pool':128},
      stride=1, method='avg', layer='4a')
  # block 4b, inception, (3,3,1024)
  X = inception_module(
      X, num_filters={'1x1':None, 'bottle_3x3':160, '3x3':256, 'bottle_5x5':64, '5x5':128, 'pool':None},
      stride=2, method='avg', layer='4b')
  
  # block 5a, inception, (3,3,736)
  X = inception_module(
      X, num_filters={'1x1':256, 'bottle_3x3':96, '3x3':384, 'bottle_5x5':None, '5x5':None, 'pool':96},
      stride=1, method='avg', layer='5a')
  # block 5b, inception, (3,3,736)
  X = inception_module(
      X, num_filters={'1x1':256, 'bottle_3x3':96, '3x3':384, 'bottle_5x5':None, '5x5':None, 'pool':96},
      stride=1, method='max', layer='5b')
  
  # avg pool, (1,1,736)
  X = tf.keras.layers.AveragePooling2D(3, 1)(X)

  # fully-connected, (128,)
  X = tf.keras.layers.Flatten()(X)
  X = tf.keras.layers.Dense(128, name='dense')(X)

  # L2 normalization, (128,)
  X = tf.math.l2_normalize(X, axis=1)

  model = tf.keras.Model(X_input, X, name='FaceNet')

  return model
  
  
def triplet_loss(model, anchor, positive, negative, alpha=0.2, reduce=True):
  """
  Implement the triplet loss.

  Inputs:
  -model: FaceNet
  -anchor: the anchor images, numpy of shape (N,96,96,3)
  -positive: the positive images, other image of the same anchor, numpy of shape (N,96,96,3)
  -negative: the negative images, other person than anchor, numpy of shape (N,96,96,3)
  -alpha: scalar, a margin between positive and negative
  -reduce: boolean, if sum losses over the number of examples

  Output:
  -loss: scalar if reduce=True, or array if reduce=False, the triplet loss
  """
  a_encoded, p_encoded, n_encoded = model(anchor), model(positive), model(negative)

  d_pos = tf.norm(a_encoded - p_encoded, axis=1)
  d_neg = tf.norm(a_encoded - n_encoded, axis=1)

  if reduce:
    loss = tf.reduce_sum(tf.maximum(tf.square(d_pos) - tf.square(d_neg) + alpha, 0))
  else:
    loss = tf.maximum(tf.square(d_pos) - tf.square(d_neg) + alpha, 0)

  return loss
  
  
def plot(X1, X2, X3, title1, title2, title3):
  plt.figure(figsize=(12, 10))
  plt.subplot(1, 3, 1)
  plt.imshow(X1)
  plt.title(title1)
  plt.axis('off')
  plt.subplot(1, 3, 2)
  plt.imshow(X2)
  plt.title(title2)
  plt.axis('off')
  plt.subplot(1, 3, 3)
  plt.imshow(X3)
  plt.title(title3)
  plt.axis('off')
  plt.show()
  
  
def D2(X1, X2, model):
  """
  Calculate the squared distance matrix between X1 and X2 using model.

  Inputs:
  -X1: numpy of shape (N1,96,96,3)
  -X2: numpy of shape (N2,96,96,3)
  -model: FaceNet

  Output:
  -D: the squared distance matrix, numpy of shape (N1,N2)
  """
  N1 = X1.shape[0]
  N2 = X2.shape[0]
  D = np.zeros((N1,N2))

  fX1 = model(X1).numpy() # (N1,128)
  fX2 = model(X2).numpy() # (N2,128)
  fX1_SumSquare = np.sum(np.square(fX1), axis=1) #(N1,)
  fX2_SumSquare = np.sum(np.square(fX2), axis=1) #(N2,)
  mul = np.dot(fX1, fX2.T) # (N1,N2)
  D += fX1_SumSquare[:,None] - 2*mul  + fX2_SumSquare

  return D
  
  
def select_triplet(X, y, model, index_class, d=1.0, vis=False, print_every=50):
  """
  Select triplets based on the FaceNet paper.

  Input:
  -X: numpy of shape (N,96,96,3)
  -y: label, of shape (N,)
  -model: FaceNet
  -d: scalar, the threshold for determining if it is the same identity
  -index_class: dictionary from indices to classes (identities)
  -vis: boolean, visualize the anchor, positive and negative
  -print_every: integer, visualize every print_every

  Outputs:
  -anchors, positives, negatives, numpy of shape (N,96,96,3)
  -E_true, E_pred: based on the truth label y and the threshold d, respectively, 
  elementary matrix (N,N) representing the identity, 1 is same, 0 otherwise
  """
  N = X.shape[0]

  # calculate the squared distance matrix
  D = D2(X, X, model)

  # list all person names
  names = []
  for label in y:
    names.append(index_class[label])

  positives, negatives = np.zeros_like(X), np.zeros_like(X)
  E_true, E_pred = np.zeros((N,N)), np.zeros((N,N))
    
  # the same identity (elementary) matrix using d
  E_pred[D <= d] = 1

  for i in range(N):
    pos_name = index_class[y[i]]

    # the indices of potential positives/negatives in names 
    pos_indices = []
    neg_indices = []
    for t, name in enumerate(names):
      if name == pos_name:
        pos_indices.append(t)
      else:
        neg_indices.append(t)
        
    #print(i, pos_name, len(pos_indices))
        
    # the same identity (elementary) matrix using ground truth labels    
    E_true[i, pos_indices] = 1
    
    # select positive[i] where encoded positive[i] is the most far away from fX[i]
    pos_dists = D[i, pos_indices]
    t_pos = pos_indices[np.argmax(pos_dists)]
    positives[i] = X[t_pos] #tf.gather(X, indices=t, axis=0)

    # select negative[i] where encoded negative[i] is the least far away from fX[i], 
    # and the square distance between encoded negative[i] and fX[i] is greater than
    # the square distance between encoded positive[i] and fX[i]
    neg_dists = D[i, neg_indices]
    mask = np.zeros_like(neg_dists)
    mask[neg_dists > D[i,t_pos]] = 1
    idxes = np.where(mask==1)[0]

    # potential problematic images, i.e., no semi-hard negative
    problems = []
    if len(idxes) > 0:
      neg_indices_filtered = []
      for idx in idxes:
        neg_indices_filtered.append(neg_indices[idx])
    elif len(idxes) == 0:
      problems.append(i)
      neg_indices_filtered = neg_indices
      #print(i, pos_name, len(pos_indices))
    neg_dists_filtered = D[i, neg_indices_filtered]
    t_neg = neg_indices_filtered[np.argmin(neg_dists_filtered)]
    negatives[i] = X[t_neg] #tf.gather(X, indices=t, axis=0)

    if vis:
      if i % print_every == 0 or i in problems:
        print(i, D[i,t_pos], D[i,t_neg])
        plot(X[i], positives[i], negatives[i], pos_name, names[t_pos], names[t_neg])

  return X, positives, negatives, E_true, E_pred
  
  
def train(train_generator, val_generator, model, index_class, d=1.0, alpha=0.2, lr=1e-3, 
          num_epochs=1, print_every=50, device='gpu:0'):
  """
  Train FaceNet with the triplet loss.

  Inputs:
  -train_generator: iterable batches of numpy images and labels
  -val_generator: iterable numpy images and labels
  -model: FaceNet
  -d: scalar, the threshold for determining if it is the same identity
  -index_class: dictionary from indices to classes for selecting triplets
  -alpha: scalar, a margin in triplet loss
  -lr: learning rate in Adam optimizer
  -num_epochs: number of epochs
  -print_every: print training progress every print_every
  -device: GPU '/gpu:0' or CPU '/cpu:0'

  Output:
  -prints progress during training
  """
  with tf.device(device):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    #train_acc = tf.keras.metrics.Accuracy(name='train_acc')
    train_auc = tf.keras.metrics.AUC(name='train_auc')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_auc = tf.keras.metrics.AUC(name='val_auc')

    t = 0
    for epoch in range(num_epochs):
      train_loss.reset_states()
      #train_acc.reset_states()
      train_auc.reset_states()

      for i, (X_batch, y_batch) in enumerate(train_generator):
        t1 = time.time()
        if i >= len(train_generator):
            break

        with tf.GradientTape() as tape:
          anchor, positive, negative, E_true, E_pred = select_triplet(X_batch, y_batch, model, index_class, d)
        
          loss = triplet_loss(model, anchor, positive, negative, alpha)
          gradients = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(zip(gradients, model.trainable_variables))

          train_loss.update_state(loss)

          y_true, y_pred = E_true.reshape(-1), E_pred.reshape(-1)
          #train_acc.update_state(y_true, y_pred)
          train_auc.update_state(y_true, y_pred)

          if t % print_every == 0:
            val_loss.reset_states()
            val_auc.reset_states()
            for j, (X_batch_val, y_batch_val) in enumerate(val_generator):
              if j >= len(val_generator):
                break
                
              a_val, p_val, n_val, E_true_val, E_pred_val = select_triplet(X_batch_val, y_batch_val, model, index_class, d)
              loss_val = triplet_loss(model, a_val, p_val, n_val, alpha)
                
              val_loss.update_state(loss_val)

              y_true_val, y_pred_val = E_true_val.reshape(-1), E_pred_val.reshape(-1)
              val_auc.update_state(y_true_val, y_pred_val)

            t2 = time.time()

            template = 'Iteration {}, Time/iter {}, Epoch {}, Loss: {}, AUC: {}, Val Loss: {}, Val AUC: {}'
            print(template.format(t, t2-t1, epoch+1, train_loss.result(), train_auc.result(), val_loss.result(), val_auc.result()))
          
          t += 1
