import math, collections

class LaplaceBigramLanguageModel:

    def __init__(self, corpus):
      self.key_map = collections.defaultdict(lambda: 0)
      self.bi_key_map = collections.defaultdict(lambda: 0)
      self.train(corpus)

    def train(self, corpus):
      """Takes a HolbrookCorpus corpus, does whatever training is needed."""
      for sentence in corpus.corpus:
        previous_token = ''  
        for datum in sentence.data:  
          token = datum.word
          self.key_map[token] = self.key_map[token] + 1
          if (previous_token):
              bigram = previous_token + '|' + token
              self.bi_key_map[bigram] = self.bi_key_map[bigram] + 1
          previous_token = token

    def score(self, sentence):
      """Takes a list of strings, returns a score of that sentence."""
      score = 0.0
      previous_token = '' 
      for token in sentence:
        if previous_token:
            bigram = previous_token + '|' + token  
            score += math.log(self.bi_key_map[bigram] + 1)
            score -= math.log(self.key_map[previous_token]+len(self.key_map))
        previous_token = token    
      return score
