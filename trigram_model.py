import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Spring 2020
Homework 1 - Programming Component: Trigram Language Models
Yassine Benajiba
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n <= len(sequence).
    """
    
    # Constraint.
    if n >= 1 and n <= len(sequence):
        # Deep copy sequence to squence_c.
        sequence_c = []
        for seq in sequence:
            sequence_c.append(seq)
    
        # Add one 'STOP' to the end of the sequence.
        sequence_c.append('STOP')
    
        # According to n, add one or more'START' to the beginning of the sequence.
        if n == 1:
            sequence_c.insert(0, 'START')
        else: 
            for i in range(n - 1):
                sequence_c.insert(0, 'START')
        
        # Extract n-grams.
        padded_ngrams = []
        for i in range(len(sequence_c)):
            if i + 1 - n >= 0:
                if n == 1:
                    padded_ngrams.append(tuple(sequence_c[i+1-n: i+1],))
                else:
                    padded_ngrams.append(tuple(sequence_c[i+1-n: i+1]))
        
        return padded_ngrams


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
        
        # Compute and store the total number of words in words_count
        self.words_count = sum(self.unigramcounts.values())


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int) 
        self.trigramcounts = defaultdict(int)
        
        ##Your code here
        
        for sentence in corpus:
            if len(sentence) >= 1: # Check whether or not n <= len(sentence)
                for unigram in get_ngrams(sentence, 1):
                    self.unigramcounts[unigram] += 1
                if len(sentence) >= 2: # Check whether or not n <= len(sentence)
                    for bigram in get_ngrams(sentence, 2):
                        self.bigramcounts[bigram] += 1
                    if len(sentence) >= 3: # Check whether or not n <= len(sentence)
                        for trigram in get_ngrams(sentence, 3):    
                            self.trigramcounts[trigram] += 1
        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        
        if trigram[0:2] == ('START', 'START'): # If the trigram is ('START', 'START', '...'), then we can use self.unigramcounts[('START',)] as the denominator. 
            trigram_p = self.trigramcounts[trigram] / self.unigramcounts[tuple([trigram[0]],)]
        elif self.bigramcounts[trigram[0:2]] == 0: # If we haven't seen ('word1','word2') in the training set, then the probability of trigram ('word1','word2','word3') should be 0.
            trigram_p = 0
        else: # In other case, we can compute the raw trigram probability as normal.
            trigram_p = self.trigramcounts[trigram] / self.bigramcounts[trigram[0:2]]
        
        return trigram_p

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        
        bigram_p = self.bigramcounts[bigram] / self.unigramcounts[tuple([bigram[0]],)]
        
        return bigram_p
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        # hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        
        unigram_p = self.unigramcounts[unigram] / self.words_count # self.words_count is the total number of words in the corpus.
        
        return unigram_p

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        import numpy as np
        
        start = tuple(['START', 'START'])
        result = []
        
        for i in range(20):
            count = 0 # Used to compute the number of trigrams, the first two words of which is what start represents.
            next_word = [] # Used to store all the possible words that may be generated next.
            word_count = [] # Used to store the count of each word corresponding to next_word.
            word_probability = [] # Used to store the probability of each word appearing after the first two words corresponding to next_word.
            
            # Find out the trigrams, the first two words of which is what start represents.
            for trigram in self.trigramcounts.keys():
                if trigram[0:2] == start:
                    count += self.trigramcounts[trigram] # Sum the count.
                    next_word.append(trigram[2]) # Append the possible word to next_word.
                    word_count.append(self.trigramcounts[trigram]) # Append the count of the possible word to word_count.
            
            # Compute the probability of each word appearing after the first two words corresponding to next_word.
            for i in range(len(next_word)):
                word_probability.append(word_count[i]/count)
            
            # Generating the next word.
            pick_result = np.random.multinomial(1, word_probability) # Draw a random word once.
            word = next_word[np.where(pick_result == 1)[0][0]]
            
            result.append(word) # Append it to the result.
            
            # If the generated word is 'STOP', then jump out of the loop. If not, then continue.
            if word == 'STOP':
                break
            else:
                start = tuple([start[1], word])
            
            # If it not generating 'STOP' but having repeated t times, then it will finish the loop.
            
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        
        smoothed_trigram_p = (lambda1 * self.raw_trigram_probability(trigram) 
                              + lambda2 * self.raw_bigram_probability(trigram[1:3]) 
                              + lambda3 * self.raw_unigram_probability(tuple([trigram[2],])))
        
        return smoothed_trigram_p
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        import math
        
        log_prob = 0
        if len(sentence) >= 3: # check whether or not n <= len(sentence)
            trigrams = get_ngrams(sentence, 3)

            for trigram in trigrams:
                log_prob += math.log2(self.smoothed_trigram_probability(trigram))
                             
        return log_prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        sum_logprob = 0
        M = 0 # Used to calculate the total number of words in the corpus.
        for sentence in corpus: 
            sum_logprob += self.sentence_logprob(sentence)
            M += (len(sentence) + 2) # Except for the number of words in the sentence, we also need to add one 'START' and one 'STOP'.
            
        l = sum_logprob / M
        perplexity = 2**(-l)
            
        return perplexity


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)  # model1: Language model for "high" essays
        model2 = TrigramModel(training_file2)  # model2: Language model for "low" essays

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            total += 1
            pp_h_1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon)) # Use model1 to compute the probability of the essay.
            pp_h_2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon)) # Use model2 to compute the probability of the essay.
            
            if pp_h_2 < pp_h_1:
                label = 'low'
            else:
                label = 'high'
            
            if label == 'high':
                correct += 1
                
    
        for f in os.listdir(testdir2):
            total += 1
            pp_l_1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon)) # Use model1 to compute the probability of the essay.
            pp_l_2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon)) # Use model2 to compute the probability of the essay.
            
            if pp_l_2 < pp_l_1:
                label = 'low'
            else:
                label = 'high'
            
            if label == 'low':
                correct += 1
        
        accuracy = correct / total
        
        return accuracy

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)





