
from textblob import TextBlob, Word
import re
from urlextract import URLExtract


def clean_text(file):
    data = file
    cleaned = []

    extractor = URLExtract()
    counter=0

    for item in data:
        item = item.lower()
        for i in extractor.find_urls(item):
            item = item.replace(i, 'thisisaurl')

        item = item.replace('@handle', 'thisisahandle')
        #item = re.sub('^[@handle] ', 'thisisahandle ', string=item)
        #item = item.replace(' @handle ', ' thisisahandle ')
        #item = re.sub(' [@handle]$', ' thisisahandle', string=item)
        #item = item.replace(' rt ', ' thisisaretweet ')
        #item = re.sub('^[rt] ', 'thisisaretweet ', string=item)    
        #item = re.sub(' [rt]$', ' thisisaretweet', string=item)
        item = re.sub('[ ]([\S]{1,}).com ', ' domaincom ', string=item)     
        item = re.sub('([\S]{1,})[.]com[ ]', 'domaincom ', string=item)
        item = re.sub(' ([\S]{1,})[.][com]$', ' domaincom', string=item)   
        item = item.replace(':)', 'happyemoji')
        item = item.replace(':D', 'happyemoji')
        item = item.replace('(;', 'happyemoji') 
        item = item.replace('(:', 'happyemoji') 
        item = item.replace('-)', 'happyemoji')    
        item = item.replace(';)', 'happyemoji')   
        item = item.replace('o)', 'happyemoji')
        

        item = item.replace(';(', 'sademoji')
        item = item.replace(':(', 'sademoji')
        item = item.replace('o(', 'sademoji')
        item = item.replace('-(', 'sademoji')
        item = item.replace(');', 'sademoji') 
        item = item.replace('):', 'sademoji') 
        
        item = item.replace(r'\W', '')
        puncuation = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
        item = re.sub(pattern=puncuation, repl=' ', string=item)
        item = re.sub(u"(\ud83d[\ude00-\ude4f])|" #emoticons
                      u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
                      u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)    
                      u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
                      u"(\ud83c[\udde0-\uddff])"
                      "+", flags=re.UNICODE, repl="emoji", string=item)  
        #words = item.split()
        #words = [Word(word).correct() for word in words]
        #item = " ".join(words)
        #words = [stemming.stem(word=word) for word in words]
        #words = [lemmatization.lemmatize(word=word) for word in words]
        cleaned.append(item)
        counter = counter+1
        print(counter)
    return cleaned

if __name__ == '__main__':
    cleaneddata = []
    with open('train_tweets.txt', 'r', encoding='utf-8') as file:
        cleaneddata = clean_text(file)

    with open('cleaned_data.txt', 'w', encoding='utf-8') as cleanfile:
        for i in cleaneddata:
            cleanfile.write(i) 

    cleanedunlabeldata = []
    with open('test_tweets_unlabeled.txt', 'r', encoding='utf-8') as file1:
       cleanedunlabeldata = clean_text(file1)

    with open('cleaned_unlabel_data.txt', 'w', encoding='utf-8') as cleanfile1:
        for j in cleanedunlabeldata:
            cleanfile1.write(j)       




