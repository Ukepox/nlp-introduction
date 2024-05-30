import PyPDF2
import re
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
from gensim.models.tfidfmodel import TfidfModel
from gensim.corpora.dictionary import Dictionary
from typing import Union


def extract_text_from_pdf(pdf_path) :
    """
    Extracts text from a pdf_path. May not read some characters if they are design heavy
    """
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

def tokenize(pdf_file, *,
                by = 'words no stop',
                doc_splitter = None,
                language = 'English',
                first_doc = 1) :
    """
    Tokenizes a pdf document. Can be tokenized by words with or without stop words or by sentences.
    Can also create a section splitter using the doc_splitter argument, creating a list of word tokenizers for each section.
    For example, if the document is clearly distinguised by chapters, specify doc_splitter = "Chapters".
    """


    valid_by_values = ['words no stop', 'words with stop', 'sentences']
    pdf = extract_text_from_pdf(pdf_file)
    if doc_splitter :
        docs = re.split(doc_splitter, pdf)[first_doc:]
        if by == 'words no stop' :
            docs_words = [word_tokenize(doc) for doc in docs]
            docs_words = [[word.lower() for word in words if word.isalpha()] for words in docs_words]
            docs_good_words = [[word for word in  words if word not in stopwords.words(language)] for words in docs_words]
            return docs_good_words
        elif by == 'words with stop' :
            docs_words = [word_tokenize(doc) for doc in docs]
            docs_words = [[word.lower() for word in words if word.isalpha()] for words in docs_words]
            return docs_words
        elif by == 'sentences' :
            docs_sents = [sent_tokenize(doc) for doc in docs]
            docs_sents = [[sent.replace('\n', ' ') for sent in doc] for doc in docs_sents ]
            return docs_sents
        else :
            raise ValueError(f"{by} is not a valid tokenizer, valid tokenizer include {valid_by_values}")
    else : 
        doc = pdf
        if by == 'words no stop' :
            doc_words = word_tokenize(doc)
            doc_words = [word.lower() for word in doc_words if word.isalpha()]
            doc_good_words = [word for word in  doc_words if word not in stopwords.words(language)]
            return doc_good_words
        elif by == 'words with stop' :
            doc_words = word_tokenize(doc)
            doc_words = [word.lower() for word in doc_words if word.isalpha()]
            return doc_words
        elif by == 'sentences' :
            doc_sents = sent_tokenize(doc)
            doc_sents = [sent.replace('\n',' ') for sent in doc_sents]
            return doc_sents
        else :
            raise ValueError(f"{by} is not a valid tokenizer, valid tokenizer include {valid_by_values}")

def get_top_words(
    document: Union[str,list], *,
    language = 'English',
    num_top_words = 10
) :
    """
    Returns the top num_top_words of a document. Stop words are not considered.
    """
    if isinstance(document, str) :
        doc_words = tokenize(
            document,
            language = language
        )
        return Counter(doc_words).most_common(num_top_words)
    elif isinstance(document, list) and isinstance(document[0], str) :
        if ' ' in document[0] :
            raise TypeError("document is not a valid word tokenizer")
        return Counter(document).most_common(num_top_words)
    else : raise TypeError("document is not a valid type. It should be a file path to a pdf or a word tokenizer")


def get_topic_words(document: Union[str,list], *,
                    doc_splitter = None,
                    language = 'English',
                    first_doc = 1,
                    num_topic_words = 10) :
    """
    Returns a Pandas Dataframe with topic words for each section in a document using tf-idf model:
    weight of word i in section j = Frequency of word i in section j * log (number of documents / number of documents with word i)
    document paramter can be either a word tokenizer or a pdf file.
    """
    

    #Checking if document is either a word tokenizer or a file path
    if isinstance(document, str) and not doc_splitter :
        raise ValueError("doc_splitter must be specified in order to get topic words")
    if isinstance(document, list) and isinstance(document[0],str) :
        raise TypeError("document parameter is not a valid word tokenizer")
    if isinstance(document, str) :

        #------------------------
        #DOCUMENT IS A FILE PATH
        #------------------------

        #Creating a word tokenizer if it is a file path
        docs_words = tokenize(document,
                            doc_splitter = doc_splitter,
                            language = language,
                            first_doc = first_doc,
                            )
        
        #Getting the topic words using the tf-idf model
        dictionary = Dictionary(docs_words)
        corpus = [dictionary.doc2bow(doc) for doc in docs_words]
        tfidf = TfidfModel(corpus)
        doc_topics = []
        for doc_corpus in corpus :
            tfidf_weights = tfidf[doc_corpus]
            #Sorting by weight, x[0] is the word id and x[1] is the associated
            tfidf_weights.sort(key = lambda x : x[1], reverse = True)
            top_words = tfidf_weights[:num_topic_words]
            topic_words = [dictionary.get(word[0]) for word in top_words]
            doc_topics.append(topic_words)
        
        #Creating a dataframe from the model
        doc_topics_df = pd.DataFrame(doc_topics)
        doc_topics_df = doc_topics_df.rename(columns={i : f'Word {i + 1}' for i in range(doc_topics_df.shape[1])})
        doc_topics_df['Doc number'] = [f'Doc {i+1}' for i in range(doc_topics_df.shape[0])]
        doc_topics_df = doc_topics_df.set_index('Doc number')
        return doc_topics_df
    elif isinstance(document, list) :
        
        if ' ' in document[0][0] :
            #Docment is not a valid word tokenizer (lazy check)
            raise TypeError("document parameter is not a valid word tokenizer.")

        #---------------------------------------------------------------
        # DOCUMENT IS A WORD TOKENIZER (MAY OR MAY NOT HAVE STOP WORDS)
        #----------------------------------------------------------------
        
        #Same code from file path, only that the tokenize() function is not called
        docs_words = document
        dictionary = Dictionary(docs_words)
        corpus = [dictionary.doc2bow(doc) for doc in docs_words]
        tfidf = TfidfModel(corpus)
        doc_topics = []
        for doc_corpus in corpus :
            tfidf_weights = tfidf[doc_corpus]
            tfidf_weights.sort(key = lambda x : x[1], reverse = True)
            top_words = tfidf_weights[:num_topic_words]
            topic_words = [dictionary.get(word[0]) for word in top_words]
            doc_topics.append(topic_words)
        doc_topics_df = pd.DataFrame(doc_topics)
        doc_topics_df = doc_topics_df.rename(columns={i : f'Word {i + 1}' for i in range(doc_topics_df.shape[1])})
        doc_topics_df['Doc number'] = [f'Doc {i+1}' for i in range(doc_topics_df.shape[0])]
        doc_topics_df = doc_topics_df.set_index('Doc number')
        return doc_topics_df
    
    #Document is neither a word tokenizer or a file path :
    else : raise TypeError("document parameter is not a valid type, should be a word tokenizer (a list of of lists) or a file path")


