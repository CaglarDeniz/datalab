import nltk 
import pdb
import copy
import numpy as np 
from tqdm import tqdm
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

# print ("Successfully imported dependencies")


""" A function used to load a document into a managable format, which 
    in order:  
        - splits the string into sentences using [.!] as delimiters
        - stems the sentences using nltk's PorterStemmer
        - transforms all the words to lowercase
        - removes stop words

    Input : A document in string format 
    Output: None
    Return Value : Document data ready for processing, and only sentence tokenized version 
"""
def loadDocument(document:str) -> list : 

    
    porter_stemmer = PorterStemmer()
    
    tokenizer = RegexpTokenizer(r'\w+')
    
    tokenized_sent = sent_tokenize(document)

    tokenized_word = [tokenizer.tokenize(sentence) for sentence in tokenized_sent]

    list_of_sentences = tokenized_word

    # removing stop words, and transforming to lowercase

    for i in range(len(list_of_sentences)):
        list_of_sentences[i] = [ porter_stemmer.stem(word).lower() for word in list_of_sentences[i] if not word in stopwords.words()]

    return list_of_sentences,tokenized_sent

""" A method that returns a summary for the supplied text document in a formatted manner. 
    This function uses the LSA, cross method mentioned in Ozsoy et Al. to algebraically decide on a sentences importance
    to be able to include it in the summary. The input matrix is filled using the 
    frequency counts.

    Input: Formatted list of document data
    Output: None
    Return Value: The summary text as a string, and the summarization percentage
"""
def lsa_cross(document:str, num_concepts:int=-1, ratio_summarize:int=0.3) -> str : 

    sentences,unformat_sent = loadDocument(document)


    # Creating the input matrix 
    # Counting the number of unique words in entire document 

    unique_word_dict = {} 
    unique_word_count = 0 


    for sentence in sentences : 

        for word in sentence : 

            if word in unique_word_dict: 
                continue 
            else : 
                unique_word_count+= 1 
                unique_word_dict[word] = True 

    sentence_count = len(sentences) 

    # creating input matrix 

    input_matrix = np.zeros(shape=(unique_word_count,sentence_count),dtype=np.int64)

    # filling up the matrix

    for j in range(len(sentences)): 
        for i in range(len(input_matrix)): 

            input_matrix[i][j] = sentences[j].count(list(unique_word_dict.keys())[i])

    # taking the SVD of the matrix 

    U, S, Vh = np.linalg.svd(input_matrix)



    # preprocessing of SVD matrix based on average row values 

    for i in range(len(Vh)): 
        average = np.mean(Vh[i])
        Vh[i] =  [ 0 if cell <= average else cell for cell in Vh[i] ]

    # calculate the length of each sentence given the number of concepts to be used

    # if no number has been given, take into account all concepts
    if num_concepts == -1 : 
        num_concepts = Vh.shape[0]
    elif num_concepts < -1 :
        sys.exit("the number of concepts must be a non-negative number")

    Vh = Vh[:,:num_concepts]
    # calculate the length of the sentence vectors 

    vector_lengths = [] 

    for i in range(Vh.shape[1]) : 

        sentence_vector = Vh[:,i]
        vector_length = np.linalg.norm(sentence_vector)
        vector_lengths.append([vector_length,i])

    vector_lengths.sort(key=lambda x : x[0])

    return_string = ""

    # pick the sentences that have made up the vectors with the greatest length

    for i in range(int(ratio_summarize*len(unformat_sent))): 

        return_string+= unformat_sent[len(unformat_sent)-1-i] 

    return return_string, ratio_summarize

doc = "America has changed dramatically during recent years. Not only has the number of graduates in traditional engineering disciplines such as mechanical, civil, electrical, chemical, and aeronautical engineering declined, but in most of the premier American universities engineering curricula now concentrate on and encourage largely the study of engineering science.  As a result, there are declining offerings in engineering subjects dealing with infrastructure, the environment, and related issues, and greater concentration on high technology subjects, largely supporting increasingly complex scientific developments. While the latter is important, it should not be at the expense of more traditional engineering.Rapidly developing economies such as China and India, as well as other industrial countries in Europe and Asia, continue to encourage and advance the teaching of engineering. Both China and India, respectively, graduate six and eight times as many traditional engineers as does the United States. Other industrial countries at minimum maintain their output, while America suffers an increasingly serious decline in the number of engineering graduates and a lack of well-educated engineers. "

ret,ratio_summarize = lsa_cross(doc,-1,0.3)

print("This is the original document: \n\n",doc,"\n\n")

print("This is the summary, which has been shrunk by {0} percent: \n\n". format(ratio_summarize),ret)
