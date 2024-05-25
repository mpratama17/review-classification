import numpy as np

class TFIDF:
    def __init__(self, data):
        '''
        data: list of string (sentence), format must be in array of string,
        not in dataframe or list of list.
        '''
        self.data = data
        self.word_list = self.create_word_list()
        self.word_count_list = self.create_word_count_list()
        self.tf_matrix = self.create_tf_matrix()
        self.idf_vector = self.create_idf_vector()
        self.tfidf_matrix = self.create_tfidf_matrix()


    def create_word_list(self):
        word_list = []  # 
        for sentence in self.data:
            for word in sentence.split():
                if word not in word_list:
                    word_list.append(word)
        return word_list
    
    def create_word_count_list(self):  # DF / Menhgitung jumlah kata yang muncul
        word_count = {}
        for w in self.word_list:
            word_count[w] = 0
            for sentence in self.data:
                for word in sentence.split():
                    if word == w:
                        word_count[w] += 1
        return word_count
    
    def count_tf(self, sentence):
        tf_vector = [0] * len(self.word_list) # output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for word in sentence.split():
            for i, w in enumerate(self.word_list): # i = index, w = word
                if w == word:
                    tf_vector[i] += 1
        # divide each element in tf_vector by the length of the sentence
        tf_vector = np.array(tf_vector) / len(sentence.split())
        return tf_vector
    
    def create_tf_matrix(self):
        tf_matrix = []
        for sentence in self.data:
            tf_matrix.append(self.count_tf(sentence))
        return tf_matrix
    
    def create_idf_vector(self): #
        idf_vector = []
        length_data = len(self.data)
        for w in self.word_list:
            count  = 0
            for sentence in self.data:
                if w in sentence.split():
                    count += 1
            idf_vector.append(np.log(length_data / count))
        return idf_vector
    
    def create_tfidf_matrix(self):
        tfidf_matrix = []
        for tf_vector in self.tf_matrix:
            tfidf_matrix.append(np.multiply(tf_vector, self.idf_vector))
        return tfidf_matrix

    def transform(self, sentence):
        tf_vector = self.count_tf(sentence)
        tfidf_vector = np.multiply(tf_vector, self.idf_vector)
        return tfidf_vector
    
    def transform_batch(self, batch):
        result = []
        for sentence in batch:
            tf_vector = self.transform(sentence)
            result.append(tf_vector)
        return result
    
    def getTopWord(self, transform_tfidf_vector, n = 5):   
        # return index of top 5 cord
        topWords = np.argsort(transform_tfidf_vector)[::-1][:n]
        result = []
        for i in topWords:
            if transform_tfidf_vector[i] != 0:
                result.append(
                    {
                        "Word": self.word_list[i],
                        "Score": transform_tfidf_vector[i]
                    }
                )
        return result 
    
    # create funtion to get top word from all dataset
    def getTopWordFromAll(self, n = 10):
        top_words = []
        for i, word in enumerate(self.word_list):
            tfidf_values = [vector[i] for vector in self.tfidf_matrix]
            avg_tfidf = np.mean(tfidf_values)
            top_words.append({"Kata": word, "Score": avg_tfidf})
        top_words = sorted(top_words, key=lambda x: x["Score"], reverse=True)
        return top_words[:n]
    
    # buat fungsi untuk mendapatkan kata yang memiliki nilai tfidf = 0 pada dataset training
    def getZeroTfidfWord(self):
        zero_tfidf_words = []
        for i, word in enumerate(self.word_list):
            tfidf_values = [vector[i] for vector in self.tfidf_matrix]
            if np.mean(tfidf_values) == 0:
                zero_tfidf_words.append(word)
        return zero_tfidf_words
    

    
    


# if __name__ == "__main__":
#     dataset = ["hello hello down there", 
#             "hello up there", 
#             "hello down there asd apa iya ahha", 
#             "hello up there",
#             "apa kabar",
#             "apa kabar",
#             "apa kabar",
#             "apa kabar",
#             ]
#     tfidf = TFIDF(dataset)
    # get topWord from all from dataset


    # topWord = tfidf.getTopWord(tfidf.transform_batch(dataset[0]))
    # print(topWord)

    # top_words = tfidf.getTopWordFromAll()
    # print("\nTop 5 words based on average TF-IDF score in the dataset:")
    # for i, word_info in enumerate(top_words):
    #     print(f"{i+1}. Word: {word_info['Kata']}, Average Score: {word_info['Score']}")
    # # print("Ini word list", tfidf.word_list)
    # print("Ini word count list", tfidf.word_count_list)

    # zero_tfidf_words = tfidf.getZeroTfidfWord()
    # print("\nWords with 0 TF-IDF score:")
    # print(zero_tfidf_words)
    # print("Panjang dari word list", len(tfidf.word_list))
    # # tr = tfidf.transform("hello up there")
    # res = tfidf.getTopWord(tfidf.transform("hello up there"))
    # print(res)

    # print(tfidf.tf_matrix) 

