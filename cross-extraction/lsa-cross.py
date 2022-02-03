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

doc = "Introduction. Pancreatic cancer is difficult to target because its fibrotic microenvironment not only acts as a barrier for delivery of tumor cell targeting drugs, but it also generates an anti-inflammatory environment and prevents immunotherapy (Balachandran et al., 2019). One of the current paradigms for treatment of PDA focuses on combining chemotherapy with immune modulators that reprogram tumor-promoting macrophages toward a pro-inflammatory phenotype (Bastea et al., 2019; Mitchem et al., 2013; Pandey and Storz, 2019). A deeper understanding of the mechanisms that play a role in macrophage polarization can provide insights to develop such new interventions. Genetic mouse models have shown that pancreatic ductal adenocarcinoma (PDA) most likely originates from precancerous pancreatic intraepithelial neoplasm (PanIN) lesions (reviewed in Storz, 2017). The development and progression of these early lesions is dependent on crosstalk between a multitude of host cells in their microenvironment. Of these, inflammatory (M1-polarized) and alternatively activated (M2-polarized) macrophages are the most consequential cell types. The initial influx of macrophages, which induces local inflammation, occurs in response to an aberrant release of chemokines from pancreatic cells undergoing transformation (Liou et al., 2015). However, local inflammation alone is not an efficient driver of oncogenic progression and requires additional inflammatory signaling, genetic alterations, and downregulation of factors that maintain acinar cell identity (Carrière et al., 2011; Cobo et al., 2018; Guerra et al., 2011; Guerra et al., 2007). Inflammatory macrophages contribute to pre-neoplastic lesion formation via secretion of inflammatory mediators, which regulate reorganization of the acinar microenvironment and initiate acinar-to-ductal metaplasia (ADM) (Liou et al., 2015; Liou et al., 2013; Sawey et al., 2007). While proinflammatory M1 macrophages are important for the initiation of precancerous lesions, this population dwindles and M2 macrophages become more predominant (Liou et al., 2017). These M2 macrophages are chitinase-like protein 3 (Ym1/Chil3), arginase-1 (Arg1), resistin-like alpha (Fizz1/Retnla), and interleukin-1 receptor antagonist protein (IL-1ra) positive and promote lesion growth, drive fibrogenesis, and block T-cell infiltration (Bastea et al., 2019; Liou et al., 2017). Later, at the tumor stage, alternatively activated macrophages represent approximately 85% of tumor-associated macrophages (TAMs) in the microenvironment (Partecke et al., 2013). For full-blown pancreatic cancer, tissue-resident macrophages have been suggested to shape fibrotic responses (Zhu et al., 2017), while infiltrating monocytes generate an immunosuppressive environment (Zhang et al., 2017; Zhu et al., 2017). C-X-C motif chemokine 10 (CXCL10), also known as IFNg-induced protein 10 (IP-10), acts through its cognate receptor C-X-C motif chemokine receptor 3 (CXCR3) (Groom and Luster, 2011) and regulates the chemotaxis of CXCR3+ immune cells such as macrophages, T cells, and natural killer (NK) cells (Luster and Ravetch, 1987; Tomita et al., 2016; Zhou et al., 2010). With respect to cancer aggressiveness and patient prognosis, the presence of CXCL10 and CXCR3 has shown conflicting results depending on the type and stage of the disease (Fulton, 2009; Jacquelot et al., 2018; Li et al., 2015). In pancreatic cancer, both CXCL10 and CXCR3 are expressed in tumor tissue (Delitto et al., 2015), and their presence has been correlated with poor prognosis (Liu et al., 2011; Lunardi et al., 2014). However, the role of CXCL10/CXCR3 signaling during early development of the disease has not been addressed. In our present study, we show that CXCL10, produced by precancerous lesions cells, is involved in the onset of inflammation by chemoattracting macrophages. We further show that CXCL10 signaling to CXCR3 is a key event for inflammatory macrophage identity and that inhibition of CXCL10/ CXCR3 signaling leads to a polarization shift to an alternatively activated phenotype. In vivo, we demonstrate the importance of CXCL10/CXCR3 signaling in the maintenance of an inflammatory microenvironment, and that its blockage drives tumor progression.To identify factors that are released by precancerous lesion cells, we performed a cytokine/chemokine assay. Therefore, we used SM3 cells, which have been isolated from the precancerous epithelium of a KC mouse and form lesions with PanIN features when cultivated on extracellular matrix (Agbunag et al., 2006; Liou et al., 2017). In this screen limited to these in vitro lesion cells (Figure 1—figure supplement 1A), besides known factors such as C-C motif chemokine 5 (CCL5) and metalloproteinase inhibitor 1 (TIMP-1), we found strong expression of CXCL10, which has previously been identified as a chemoattractant for macrophages (Tomita et al., 2016; Zhou et al., 2010). We then used fluorescent in situ hybridization (FISH) to determine whether Cxcl10 is produced in pancreatic precancerous lesion areas of Ptf1a/p48cre;LSL-KrasG12D (KC) mice. While Cxcl10 was undetectable in normal adjacent acini (Figure 1—figure supplement 1B), we found significant expression of Cxcl10 in ADM and PanIN1 lesions (Figure 1A). Quantification analyses of samples stained for Cxcl10 mRNA by ISH indicated approximately fivefold higher expression in ADM than in PanIN (Figure 1B, Figure 1—figure supplement 1C). Next, we isolated primary pancreatic acinar cells from LSL-KrasG12D mice and adenovirally infected them with either GFP (control) or Cre-GFP, to test whether Cxcl10 expression is upregulated during the KRasG12D-driven ADM process (Figure 1—figure supplement 1D). However, expression of KRasG12D was unable to increase Cxcl10 expression, indicating an external stimulus as a driver. CXCL10 (also IP-10, interferon gamma-inducible protein 10) expression has previously been shown to be induced by interferon gamma (IFNg) via activation of signal transducer and activator of transcription 1 (STAT1) (Han et al., 2010; Luster and Ravetch, 1987). Therefore, we tested if this pathway is active in PanIN cells. Treatment of SM3 cells with IFNg induced an over 60-fold increase in Cxcl10 mRNA (Figure 1C), as well as increased CXCL10 protein production (Figure 1D) and secretion (Figure 1E). To test whether CXCL10 expression is indeed mediated through STAT1 signaling, we combined IFNg stimulation with the pan-JAK inhibitor NVP-BSK805. We found that IFNg stimulation led to phosphorylation of STAT1 at Y701 (activating phosphorylation), increased expression of CXCL10, and that pre-treatment with NVP-BSK805 inhibited IFNg-induced pY701-STAT1 and CXCL10 expression (Figure 1F). T cells and NK cells are known IFNg producers in the pancreatic microenvironment (Brauner et al., 2010; Chapoval et al., 2001; Loos et al., 2009). To determine whether these cells could be an in vivo source for IFNg in our mouse model, we performed an ISH for Ifng combined with IHC for T-cell surface glycoprotein CD4 (CD4), T-cell surface glycoprotein CD8 (CD8), or NKG2-D type II integral membrane protein (NKG2D) markers. As expected from published data, we found both T cells and NK cells as a potential source for IFNg (Figure 1G)With respect to early events leading to development of PDA, the influx of macrophages into the pancreas has been demonstrated following injury and during development and progression of pancreatic lesions (Gea-Sorlı´ and Closa, 2009; Liou et al., 2015). Moreover, CXCL10 has been demonstrated as a chemoattractant for macrophages along with other immune cells (Liu et al., 2011). This prompted us to test whether macrophages are responsive to CXCL10. We found that non-polarized peritoneal macrophages express high levels of the CXCL10 receptor Cxcr3, while M1-polarized (inflammatory) macrophages express moderate levels, and M2-polarized (alternatively activated) macrophages do not express this receptor (Figure 2A). Transwell invasion assays using both peritoneal and bone marrow-derived macrophages suggest that CXCL10 can act as a chemoattractant for M1-polarized macrophages (Figure 2B, Figure 2—figure supplement 1A). However, since tissueresident macrophages have been attributed important roles in established pancreatic cancer (Zhu et al., 2017), we also determined if this population can be the recipients for CXCL10. Approximately 80% of tissue resident macrophages in normal mouse pancreas express CXCR3 (Figure 2— figure supplement 1A), but when isolated, these cells do not proliferate in response to CXCL10 (Figure 2—figure supplement 1B). In sum, our in vitro data suggests that CXCL10 may drive the chemoattraction of inflammatory macrophages to the pancreas.  ext, we determined if pancreatic macrophages or T cells express CXCR3 in KC mice. Therefore, we sorted for pancreatic CD3+ or F4/80+ cells and then for the presence of CXCR3. We found that approximately 40% of pancreatic macrophages in KC mice express CXCR3 (Figure 2C). Moreover, an in situ IF-IHC analysis of pancreata of KC mice indicated that inflammatory (F4/80+;pY701- STAT1+) macrophages express CXCR3, while alternatively activated (F4/80+;Ym1+) macrophages do not express this receptor (Figure 2D), which also confirmed above in vitro data. An overlay between an ISH for Cxcr3 and IF-IHC for inflammatory macrophages (CD68+;iNOS+) in human patient tumors showed that ~70% of CXCR3+ cells are M1 macrophages and confirmed this population as a potential recipient for CXCL10 (Figure 2E,F)"
ret,ratio_summarize = lsa_cross(doc,-1,0.3)

print("This is the original document: \n\n",doc,"\n\n")

print("This is the summary, which has been shrunk by {0} percent: \n\n". format(ratio_summarize),ret)
