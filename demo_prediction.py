import os
import time
import itertools
import sys
import numpy as np
import tensorflow as tf
import udc_model
import udc_hparams
import udc_metrics
import udc_inputs
from models.dual_encoder import dual_encoder_model
from models.helpers import load_vocab
import requests
import config

tf.flags.DEFINE_string("raw_query", None, "Question that must be answered")
tf.flags.DEFINE_string("model_dir", None, "Directory to load model checkpoints from")
tf.flags.DEFINE_string("vocab_processor_file", config.project_path + "data/vocab_processor.bin", "Saved vocabulary processor file")
FLAGS = tf.flags.FLAGS

if not FLAGS.model_dir:
  print("You must specify a model directory")
  sys.exit(1)

if not FLAGS.raw_query:
  print("You must ask a question!")
  sys.exit(1)

solr_server = config.solr_server
col_name = config.solr_core

raw_query = FLAGS.raw_query
query = raw_query.replace(" ", "%20")
number_results = config.n_candidates

url_query = solr_server + col_name + 'select?defType=edismax&indent=on&bq=responseTo:[*%20TO%20*]^5&q.alt=' + query + '&qf=responseTo&rows=' + str(number_results) + '&wt=json'

r = requests.get(url_query).json()

candidates_objects = r['response']['docs']

# Canditates to be answers to the given question
candidates = [ c['content'] for c in candidates_objects ]
print(candidates)

def tokenizer_fn(iterator):
  return (x.split(" ") for x in iterator)

# Load vocabulary
vp = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
  FLAGS.vocab_processor_file)

# Load your own data here
INPUT_CONTEXT = raw_query
POTENTIAL_RESPONSES = candidates

def get_features(context, utterance):
  context_matrix = np.array(list(vp.transform([context])))
  utterance_matrix = np.array(list(vp.transform([utterance])))
  context_len = len(context.split(" "))
  utterance_len = len(utterance.split(" "))
  features = {
    "context": tf.convert_to_tensor(context_matrix, dtype=tf.int64),
    "context_len": tf.constant(context_len, shape=[1,1], dtype=tf.int64),
    "utterance": tf.convert_to_tensor(utterance_matrix, dtype=tf.int64),
    "utterance_len": tf.constant(utterance_len, shape=[1,1], dtype=tf.int64),
  }
  return features, None

if __name__ == '__main__':
  hparams = udc_hparams.create_hparams()
  model_fn = udc_model.create_model_fn(hparams, model_impl=dual_encoder_model)
  estimator = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir)

  # Ugly hack, seems to be a bug in Tensorflow
  # estimator.predict doesn't work without this line
  estimator._targets_info = tf.contrib.learn.estimators.tensor_signature.TensorSignature(tf.constant(0, shape=[1,1]))

  # Question from the user
  print("Context: {}".format(INPUT_CONTEXT))
  scores = []

  final_answer = 'I do not understand. Can you please ask another question?'
  position = 0

  # We iterate over all the possible answers and score them with the Dual Encoder
  for r in POTENTIAL_RESPONSES:
    prob = estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT, r))
    results = next(prob)
    scores.append(results)
    print(r, results)

  # We sort them by score and write the results in a .txt file that will be read later.
  # If you are only interested in the best answer from the Dual Encoder, just take nn_candidates[0]
  nn_candidates = sorted(range(len(scores)), key=lambda i: scores[i])[-3:][::-1]
  nn_candidates = POTENTIAL_RESPONSES[nn_candidates[0]] + "___***___" + POTENTIAL_RESPONSES[nn_candidates[1]]+ "___***___" + POTENTIAL_RESPONSES[nn_candidates[2]] + "___|||___"
  to_write = nn_candidates +  "___***___".join(candidates[:3])
  predict_dir = config.project_path + 'answers.txt'  
  with open(predict_dir, "w+") as text_file:
    text_file.write(to_write)
    text_file.close()
  print("Benchmark result:", candidates[0])
  print("Dual encoder result:", nn_candidates)
  #return candidates[0], final_answer
