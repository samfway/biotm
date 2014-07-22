#!/usr/bin/env python

from biotm.topic_models.plsa.em import plsa_em
from numpy import zeros

num_docs, num_words, num_topics = 10, 100, 2
X = zeros((num_docs, num_words))
p_w_z = zeros((num_words, num_topics))
p_z_d = zeros((num_topics, num_docs))
p_d = zeros(num_docs)
folding = False

plsa_em.plsa_em(X, p_w_z, p_z_d, p_d, folding)
