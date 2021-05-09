import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import tensorflow_datasets as tfds
import tqdm

# got this code from nsheppard
# vs. tf.Saver, this puts vars directly into the GPU so we don't fill up RAM

def split_by_params(vs, n=200e6, f=None):
  if f is None:
    f = lambda x: np.prod(x.shape.as_list())
  i = 0
  xs = []
  for variable in vs:
    xs.append(variable)
    count = f(variable)
    i += count
    if i >= n:
      yield xs
      xs = []
      i = 0
  yield xs

def grab_values(variables, reader, reshape=False):
  for variable in variables:
    name = variable.name.split(':')[0]
    value = reader.get_tensor(name)
    yield variable, value

def assign_values(variables, values, session=None):
  session = session or tf.get_default_session()
  ops = [x.initializer for x in variables]
  vals = dict([(x.initializer.inputs[1], value) for x, value in zip(variables, values)])
  session.run(ops, vals)

def load_snapshot(ckpt, session=None, var_list=None, reshape=False):
  session = session or tf.get_default_session()
  reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
  vs = var_list or tf.trainable_variables()
  for variables in tqdm.tqdm(list(split_by_params(vs))):
    values = [value for variable, value in grab_values(variables, reader, reshape=reshape)]
    assign_values(variables, values, session=session)

def tokenized_wikipedia_dataset(min_tokens_per_text, max_tokens_per_text, batch_size=1):
  ds = tfds.load('wikipedia/20200301.en', split='train', try_gcs=True)

  def text_to_encoded(text):
    encoded = np.array(e.encode(str(text))[:MAX_NUM_TOKENS_PER_TEXT], np.int32)
    return [encoded] if len(encoded) >= MIN_NUM_TOKENS_PER_TEXT else []

  num_tokens, *_ = *({MAX_NUM_TOKENS_PER_TEXT} & {MIN_NUM_TOKENS_PER_TEXT}), None

  ds = ds.flat_map(lambda data: tf.data.Dataset.from_generator(
      text_to_encoded, tf.int32, tf.TensorShape((num_tokens,)), (data['text'],)))

  ds = ds.batch(1, True).prefetch(tf.data.experimental.AUTOTUNE)