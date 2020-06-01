from tensorflow.python.ops.custom_gradient import custom_gradient
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.eager import backprop

def recompute_grad(f):
    """An eager-compatible version of recompute_grad.
    For f(*args, **kwargs), this supports gradients with respect to args or
    kwargs, but kwargs are currently only supported in eager-mode.
    Note that for keras layer and model objects, this is handled automatically.
    Warning: If `f` was originally a tf.keras Model or Layer object, `g` will not
    be able to access the member variables of that object, because `g` returns
    through the wrapper function `inner`.  When recomputing gradients through
    objects that inherit from keras, we suggest keeping a reference to the
    underlying object around for the purpose of accessing these variables.
    Args:
      f: function `f(*x)` that returns a `Tensor` or sequence of `Tensor` outputs.
    Returns:
     A function `g` that wraps `f`, but which recomputes `f` on the backwards
     pass of a gradient call.
    """

    # TODO(cdfreeman) Add is_recomputing functionality from graph mode version

    @custom_gradient
    def inner(*args, **kwargs):
        """Inner function closure for calculating gradients."""
        current_var_scope = variable_scope.get_variable_scope()

        result = f(*args, **kwargs)

        def grad(*dresult, **grad_kwargs):
            """Gradient function calculation for inner function."""
            variables = grad_kwargs.get("variables")
            with backprop.GradientTape() as t:
                id_args = [gen_array_ops.identity(x) for x in args]
                t.watch(id_args)
                if variables is not None:
                    t.watch(variables)
                with ops.control_dependencies(dresult):
                    with variable_scope.variable_scope(current_var_scope):
                        result = f(*id_args, **kwargs)
            kw_vars = []
            if variables is not None:
                kw_vars = list(variables)
            grads = t.gradient(result, list(id_args) + kw_vars, output_gradients=dresult)
            return grads[:len(id_args)], grads[len(id_args):]

        return result, grad

    return inner


import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

l = 0

# 不加这句修饰只能跑batchsize=16，否则可跑batchsize=64以上
@recompute_grad
def transformer(x, dim=1024):
    with tf.variable_scope(name_or_scope='layer_{}'.format(l),
                           use_resource=True, reuse=tf.AUTO_REUSE):
        q = tf.layers.dense(x, dim)  # [bs, len, dim]
        k = tf.layers.dense(x, dim)
        v = tf.layers.dense(x, dim)

        qk = tf.matmul(q, k, transpose_b=True)  # [bs, len, len]
        att = tf.nn.softmax(qk, axis=2)

        out = tf.matmul(att, v)
        out = tf.layers.dense(out, dim * 4)
        out = tf.nn.relu(out)
        out = tf.layers.dense(out, dim)
        return out + x


from tqdm import tqdm

input = tf.random_uniform(shape=[64, 512], minval=0, maxval=29999, dtype=tf.int32)
labels = tf.random_uniform(shape=[64, 512], minval=0, maxval=29999, dtype=tf.int32)
embedding_table = tf.get_variable(name='embedding', shape=[30000, 1024])
emb = tf.nn.embedding_lookup(embedding_table, input)
x = emb
for i in range(24):
    x = transformer(x)
    l += 1
logits = tf.layers.dense(x, 30000)  # [bs, len, 30000]
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
print(loss)
adam = tf.train.AdamOptimizer(1e-5)
train_op = adam.minimize(loss)

sess_config = tf.ConfigProto()
sess_config.allow_soft_placement = True
sess_config.gpu_options.allow_growth = True
with tf.Session(config=sess_config) as sess:
    sess.run(tf.global_variables_initializer())
    for step in tqdm(range(10000)):
        loss_value, _ = sess.run([loss, train_op])
        print('Step:{} Loss:{}'.format(step, loss_value))
