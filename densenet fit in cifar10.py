####################
# 避免占满
####################

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

####################
# DenseNet201
####################

clf = tf.keras.applications.DenseNet201(
    input_shape = (32, 32, 3),
    weights = 'imagenet',
    include_top = False,

    classes = 10,
)
    
clf.trainable = True

model = tf.keras.Sequential([
    clf,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense( 10, activation='softmax' )
])
        
model.compile(
    optimizer = tf.keras.optimizers.Adam( learning_rate=1e-4 ),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.summary()

####################
# cifar10
####################

# 归一化数据
(x_train, y_train), (x_test, y_test) = \
tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 独热编码
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

####################
# train
####################

# 没加earlystopping，就是测试一下，去别的代码里面复制

model.fit( x_train, y_train,
           validation_data = (x_test, y_test),

           epochs = 10000, batch_size = 32,

           verbose = 1, # 2 一次训练就显示一行

           shuffle = True, # 再次打乱

           # max_queue_size = 10000,
           workers = 4,
           use_multiprocessing = True,
           )
