import pandas as pd
import nltk
import re
import numpy as np
nltk.download('punkt')
from nltk import word_tokenize, FreqDist, ngrams
from sys import argv
import os

class Summary():
    
    def __init__(self, text) -> None:
        self.text = text.lower()
        self.sentences = self.sen_tokenize(self.text)
        self.tokens = []
        self.one_word_pivot = pd.DataFrame(columns=["lead","follow","freq"])
        self.two_word_pivot = pd.DataFrame(columns=["lead","follow","freq"])
        
        self.word_process()
        self.tokenized_text()
        self.make_one_word_pivot()
        self.make_two_word_pivot()
        
        
    def word_process(self):
        new_text = ""
        new_sentences = []
        
        for line in self.sentences:
            line = re.sub(r'[^\w\s]', '', line)
            new_line = " stx "+line+" enx "
            new_text = new_text + new_line
            new_sentences.append(new_line)
            
        self.text = new_text
        self.sentences = new_sentences
        
    
    def tokenized_text(self):
        self.tokens = word_tokenize(self.text)
    
    
    def get_score(self, lead, follow):
        return self.one_word_pivot.loc[self.one_word_pivot.index == lead][follow].fillna(0).values[0]
    
    
    def sen_tokenize(self, text):
        return nltk.sent_tokenize(text)
    
    
    def make_one_word_pivot(self):
        follow = self.tokens[1:]
        follow.append("enx")


        self.one_word_pivot["lead"] = self.tokens
        self.one_word_pivot["follow"] = follow
        
        self.one_word_pivot["freq"] = self.one_word_pivot.groupby(["lead","follow"])["lead","follow"].transform("count").copy()["lead"]
        self.one_word_pivot = self.one_word_pivot.drop_duplicates()
        
        self.one_word_pivot = self.one_word_pivot.pivot(index="lead",columns="follow", values="freq")
        sum_words = self.one_word_pivot.sum(axis=1)
        self.one_word_pivot = self.one_word_pivot.apply(lambda x: x/sum_words)
        
    
    
    def make_two_word_pivot(self):
        tri_grams = list(ngrams(self.tokens, 3))
        
        follow = [ pair[-1] for pair in tri_grams ]
        # follow.append("enx")

        self.two_word_pivot["lead"] = [ pair[:-1] for pair in tri_grams ]
        self.two_word_pivot["follow"] = follow
        
        self.two_word_pivot["freq"] = self.two_word_pivot.groupby(["lead","follow"])["lead","follow"].transform("count").copy()["lead"]
        self.two_word_pivot = self.two_word_pivot.drop_duplicates()
        
        self.two_word_pivot = self.two_word_pivot.pivot(index="lead",columns="follow", values="freq")
        sum_words = self.two_word_pivot.sum(axis=1)
        self.two_word_pivot = self.two_word_pivot.apply(lambda x: x/sum_words)
        
    
    
    def get_next(self, lead):
        ans = ""
        if isinstance(lead, tuple):
            ans = np.random.choice(a=self.two_word_pivot.columns,size=1,p=self.two_word_pivot.loc[self.two_word_pivot.index == lead].fillna(0).values[0])[0]
        else:
            ans = np.random.choice(a=self.one_word_pivot.columns,size=1,p=self.one_word_pivot.loc[self.one_word_pivot.index == lead].fillna(0).values[0])[0]
        return ans
    

    def summary(self):
        result = ""
        
        NO_OF_SENTENCES = 10
        MIN_WORDS_EACH_SENTENCE = 8
        
        for _ in range(NO_OF_SENTENCES):
            
            curr_sent = []
            count = 0
            curr_sent.append("stx")
            prev = ""
            while count < MIN_WORDS_EACH_SENTENCE or curr_sent[-1] != "enx":
                if len(curr_sent) == 1:
                    next = self.get_next("stx")
                else:
                    lead = tuple(curr_sent[-2:])
                    next = self.get_next(lead)
                curr_sent.append(next)
                count += 1
                
            result = result + " ".join(curr_sent[1:-1]) + ".\n"
            os.system('cls')
            os.system('clear')
        return result
        
    
    
if __name__ == '__main__':
    input_path, output_relative_path = argv[1],argv[2]
    with open(input_path,'r') as f:
        text = " ".join(f.readlines())
        sum_obj = Summary(text=text)
        with open(output_relative_path,'w') as f2:
            f2.write(sum_obj.summary())
    
    