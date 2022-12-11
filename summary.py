import pandas as pd
import nltk
import re
import numpy as np
nltk.download('punkt')
from nltk import word_tokenize, FreqDist
from sys import argv


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
    
    
    def get_sent_parts(self, parts_length = 5):
        i = 0
        while True:
            if i+parts_length > len(self.sentences):
                return self.sentences[i:]
            else:
                yield self.sentences[i:i+parts_length]
                i += parts_length
    
    
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
        
    
    
    def make_two_word_pivot():
        pass
    
    
    def get_next(self,words_list,lead):
        sen_score = -1
        ans = "---"
        for follow_word in words_list:
            curr_score = self.get_score(lead,follow_word)
            if curr_score > sen_score and curr_score > 0.00000000:
                sen_score = curr_score
                ans = follow_word
        if ans == "---":
            ans = np.random.choice(a=self.one_word_pivot.columns,size=1,p=self.one_word_pivot.loc[self.one_word_pivot.index == lead].fillna(0).values[0])[0]
        return ans
    

    def summary(self):
        result = ""
        
        for some_sen in self.get_sent_parts(5):
            
            next_sen = ""
            words_tokens = []
            
            for sent  in some_sen:
                words_tokens.extend(word_tokenize(sent))
            
            words_freq = FreqDist(words_tokens)
            prev = "stx"
            count = 0
            common = [k for k,v in words_freq.items() if v > 1 ]
            
            for word in ["stx","and","but","what","why"]:
                try:
                    common.remove(word)
                except:
                    pass
                
                
            N = len(common)
            while count < N:
                next = self.get_next(common,prev)
                try:
                    common.remove(next)
                except:
                    pass
                if next == "enx":
                    next, prev = ".", "stx"
                    next_sen = next_sen +"."
                else:
                    next_sen = next_sen +" "+ next
                    prev = next
                count += 1
            result = result+next_sen+"\n"
            
        return result
        
    
    
if __name__ == '__main__':
    input_path, output_relative_path = argv[1],argv[2]
    with open(input_path,'r') as f:
        text = " ".join(f.readlines())
        sum_obj = Summary(text=text)
        with open(output_relative_path,'w') as f2:
            f2.write(sum_obj.summary())
    
    