

import params
import argparse
import utility_functions

import tensorflow as tf
from tensorflow.keras import models 
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

def get_model():
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_outputs = [vgg.get_layer(name).output for name in params.style_layers]
    content_outputs = [vgg.get_layer(name).output for name in params.content_layers]
    model_outputs = style_outputs + content_outputs
    return models.Model(vgg.input, model_outputs)

def get_content_loss(content, target):
    return tf.reduce_mean(tf.square(content - target))

def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def hist_match(source, template):
    shape = tf.shape(source)
    source = tf.layers.flatten(source)
    template = tf.layers.flatten(template)
    hist_bins = 255
    max_value = tf.reduce_max([tf.reduce_max(source), tf.reduce_max(template)])
    min_value = tf.reduce_min([tf.reduce_min(source), tf.reduce_min(template)])
    hist_delta = (max_value - min_value)/hist_bins
    hist_range = tf.range(min_value, max_value, hist_delta)
    hist_range = tf.add(hist_range, tf.divide(hist_delta, 2))
    s_hist = tf.histogram_fixed_width(source, 
                                        [min_value, max_value],
                                         nbins = hist_bins, 
                                        dtype = tf.int64
                                        )
    t_hist = tf.histogram_fixed_width(template, 
                                         [min_value, max_value],
                                         nbins = hist_bins, 
                                        dtype = tf.int64
                                        )
    s_quantiles = tf.cumsum(s_hist)
    s_last_element = tf.subtract(tf.size(s_quantiles), tf.constant(1))
    s_quantiles = tf.divide(s_quantiles, tf.gather(s_quantiles, s_last_element))

    t_quantiles = tf.cumsum(t_hist)
    t_last_element = tf.subtract(tf.size(t_quantiles), tf.constant(1))
    t_quantiles = tf.divide(t_quantiles, tf.gather(t_quantiles, t_last_element))


    nearest_indices = tf.map_fn(lambda x: tf.argmin(tf.abs(tf.subtract(t_quantiles, x))), 
                                  s_quantiles, dtype = tf.int64)

    s_bin_index = tf.to_int64(tf.divide(source, hist_delta))

    s_bin_index = tf.clip_by_value(s_bin_index, 0, 254)
    matched_to_t = tf.gather(hist_range, tf.gather(nearest_indices, s_bin_index))
    return tf.reshape(matched_to_t, shape)

def hist_loss(source, target):
    histogram = hist_match(source, target)
    loss = get_content_loss(source, target)
    return loss

def compute_loss(init_image, style_img_content_features, style_img_gram_features, style_img_histogram_features):
    style_score = 0
    content_score = 0
    hist_score = 0

    model_outputs = model(init_image)
    init_image_gram_features = [gram_matrix(style_layer[0]) for style_layer in model_outputs[:params.num_style_layers]]
    init_img_content_features = [content_layer[0] for content_layer in model_outputs[params.num_style_layers:]]
    init_img_histogram_features = [hist_layer[0] for hist_layer in model_outputs[:params.num_style_layers]]

    weight_per_style_layer = 1.0 / float(params.num_style_layers)
    for content_img_gram_layer, style_img_gram_layer in zip(init_image_gram_features, style_img_gram_features):
        style_score += weight_per_style_layer * get_content_loss(content_img_gram_layer, style_img_gram_layer)

    weight_per_content_layer = 1.0 / float(params.num_content_layers)
    for content_img_content_layer, style_img_content_layer in zip(init_img_content_features, style_img_content_features):
        content_score += weight_per_content_layer * get_content_loss(content_img_content_layer, style_img_content_layer)

    weight_per_hist_layer = 1.0 / float(params.num_style_layers)
    for content_img_histogram_layer, style_img_histogram_layer in zip(init_img_histogram_features, style_img_histogram_features):
        hist_score += weight_per_hist_layer * hist_loss(content_img_histogram_layer, style_img_histogram_layer)

    loss = (style_score * params.style_weight) + (content_score * params.content_weight) + (hist_score / params.hist_ratio)
    return loss

def compute_grads(init_image, style_img_content_features, style_img_gram_features, style_img_histogram_features):
    with tf.GradientTape() as tape: 
        all_loss = compute_loss(init_image, style_img_content_features, style_img_gram_features, style_img_histogram_features)
    total_loss = all_loss
    return tape.gradient(total_loss, init_image), all_loss


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--content_img", help="Enter Content Image Input path here")
    parser.add_argument("--style_img", help="Enter Style Image Input path here")

    args = parser.parse_args()
    content_image_path = args.content_img
    style_image_path = args.style_img

    content_img = utility_functions.read_image(content_image_path)
    content_img = tf.keras.applications.vgg19.preprocess_input(content_img)

    style_img = utility_functions.read_image(style_image_path)
    style_img = tf.keras.applications.vgg19.preprocess_input(style_img)

    model = get_model() 
    for layer in model.layers:
        layer.trainable = False


    style_img_outputs = model(style_img)
    style_img_gram_features = [gram_matrix(style_layer[0]) for style_layer in style_img_outputs[:params.num_style_layers]]
    style_img_histogram_features = [hist_layer[0] for hist_layer in style_img_outputs[:params.num_style_layers]]
    style_img_content_features = [content_layer[0] for content_layer in style_img_outputs[params.num_style_layers:]]

    init_image = tfe.Variable(content_img.copy(), dtype=tf.float32)
    opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)


    norm_means = utility_functions.np.array([103.939, 116.779, 123.68])
    min_vals = - norm_means
    max_vals = 255 - norm_means

    for i in range(params.num_iterations):
        grads, all_loss = compute_grads(init_image, style_img_content_features, style_img_gram_features, style_img_histogram_features)
        loss = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        if i % 100 == 0:
            utility_functions.io.imsave('out.jpg', utility_functions.convert_to_real_image(init_image.numpy()))
        print(loss)