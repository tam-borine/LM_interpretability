import tensorflow as tf
from model import shape_list, attn, mlp, positions_for

# Forked from openai's gpt2 implementation with a couple of extra named `tf.identity`.

def norm(x, scope, *, axis=-1, epsilon=1e-5, reuse=tf.AUTO_REUSE):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope(scope, reuse=reuse):
        n_state = x.shape[-1].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        x = tf.identity(x, name='pre_norm')
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = tf.identity(x, name='pre_affine')
        x = x*g + b
        x = tf.identity(x, name='post_norm')
        return x

def block(x, scope, *, past, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        x = tf.identity(x + a, name='post_attn')
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
        x = tf.identity(x + m, name='out')
        return x, present

def instrumented_model(
    hparams, X, past=None, scope='model', 
    return_present=False,
    reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch_dim, sequence_dim = shape_list(X)

        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        past_length = 0 if past is None else tf.shape(past)[-2]
        tok_e = tf.gather(wte, X, name='tok_e')
        pos_e = tf.gather(wpe, positions_for(X, past_length), name='pos_e')
        h = tf.add(tok_e, pos_e, name='h_in')

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, f'h{layer}', past=past, hparams=hparams)
            presents.append(present)
        if return_present:
          results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f')

        # Decode intermediate activation as if they were logits
        for layer in range(hparams.n_layer):
          h_pre_norm = tf.get_default_graph().get_tensor_by_name(f'model/h{layer}/ln_1/pre_norm:0')
          h_post_norm = tf.get_default_graph().get_tensor_by_name(f'model/h{layer}/ln_1/post_norm:0')
          # In this scope, `norm` computes the final layer norm pre-wte.
          h_pre_norm = norm(h_pre_norm, 'ln_f', reuse=True)
          for k, v in dict(pre_norm=h_pre_norm, post_norm=h_post_norm).items():
            v_flat = tf.reshape(v, [batch_dim*sequence_dim, hparams.n_embd])
            logits = tf.matmul(v_flat, wte, transpose_b=True)
            results[f'logits_{k}_{layer}'] = tf.reshape(logits, [batch_dim, sequence_dim, hparams.n_vocab])

        # Decode final activation as logits
        h_flat = tf.reshape(h, [batch_dim*sequence_dim, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        results['logits_out'] = tf.reshape(logits, [batch_dim, sequence_dim, hparams.n_vocab])
        return results
