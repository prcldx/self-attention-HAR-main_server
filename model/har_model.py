import tensorflow as tf

from .attentive_pooling import AttentionWithContext
from .self_attention.encoder import EncoderLayer
from .self_attention.positional_encoding import PositionalEncoding
from .sensor_attention import SensorAttention

# 函数式地创建模型 在TF官网上“自定义创建类”有详细的介绍，这种创建模型的方式对复杂组合式模型结构很友好，其实我也很容易改成model类型的方式
def create_model(n_timesteps, n_features, n_outputs, _dff=512, d_model=128, nh=4, dropout_rate=0.2, use_pe=True):
    inputs = tf.keras.layers.Input(shape=(n_timesteps, n_features,))

    si, _ = SensorAttention(n_filters=128, kernel_size=3, dilation_rate=2)(inputs)#可能会出现numpy版本与tensorflow版本等级不太兼容的问题，解决方案：降低numpy的版本

    x = tf.keras.layers.Conv1D(d_model, 1, activation='relu')(si)
    #是否要用位置编码
    if use_pe:
        x *= tf.math.sqrt(tf.cast(d_model, tf.float32))#x*sqrt(d)为什么要进行这一步缩放
        x = PositionalEncoding(n_timesteps, d_model)(x)
        x = tf.keras.layers.Dropout(rate=dropout_rate)(x)

    x = EncoderLayer(d_model=d_model, num_heads=nh, dff=_dff, rate=dropout_rate)(x)
    x = EncoderLayer(d_model=d_model, num_heads=nh, dff=_dff, rate=dropout_rate)(x)
    # x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = AttentionWithContext()(x)
    # x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(n_outputs * 4, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    # x = tf.keras.layers.Dense(128, activation='relu') (x)

    predictions = tf.keras.layers.Dense(n_outputs, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)

    return model
