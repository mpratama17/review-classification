import numpy as np
class MultinomialNB:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.class_counts = {}
        self.class_feature_counts = {} # hasilnya per kelas, di dalam kelas terdiri dari list vocab besar dan val nya
        self.total_feature_counts = {} 
        self.prior = {}
        self.likelihood = {}
        self.posteriors = {}

    def fit(self, X, y):
        classes = np.unique(y)
        total_sentences = len(y)
        num_features = len(X[0]) # jumlah fitur

        # Menghitung prior
        for label in classes:
            self.class_counts[label] = 0
            for sentence_label in y:
                if sentence_label == label:
                    self.class_counts[label] += 1
            self.prior[label] = self.class_counts[label] / total_sentences
        
        # Menghitung frekuensi masing-masing feature di setiap kelas 
        for label in classes:
            feature_counts = []
            for tfidf_vector, tfidf_class in zip(X, y):
                if tfidf_class == label:
                    feature_counts.append(tfidf_vector)
            self.class_feature_counts[label] = np.sum(feature_counts, axis=0) # axis = 0, itu vertical
            
        # Menghitung total kemunculan fitur di setiap kelas
        for label in classes:
            for tfidf_vector, tfidf_class in zip(X, y):
                if tfidf_class == label:
                    self.total_feature_counts[label] = np.sum(tfidf_vector)
                
        # Menghitung likelihood
        for label in classes:
            # likelihood = (jumlah kata x_i di kelas c + alpha) / (jumlah kata di kelas c + alpha * jumlah kata unik)
            class_feature_counts = [result + self.alpha for result in self.class_feature_counts[label]] # hasilnya array
            # 
            total_feature_counts = self.total_feature_counts[label]
            self.likelihood[label] = class_feature_counts / (total_feature_counts + self.alpha * num_features)
                
    def predict(self, X):
        predictions = []
        for sentence in X:
            posteriors = {}
            for label, class_probs in self.prior.items():
                posterior = class_probs
                for i, word_count in enumerate(sentence):
                    if word_count != 0.0:
                        likelihood_word_given_label = self.likelihood[label][i] 
                        posterior *= likelihood_word_given_label ** word_count
                        # posterior *= likelihood_word_given_label
                posteriors[label] = posterior
                self.posteriors[label] = posteriors[label]
                # evidence
                # evidence = sum(posteriors.values())
                # for label in posteriors:
                #     posteriors[label] = round(posteriors[label] / evidence, 3) 
            predicted_label = max(posteriors, key=posteriors.get)
            predictions.append(predicted_label)
        return predictions

# import numpy as np
# if __name__ == '__main__':
#     X = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
#     y = np.array([0, 1, 0, 1])
#     mnb = MultinomialNB()
#     mnb.fit(X, y)
#     print("hasil", mnb.predict(np.array([[1, 0, 0], [0, 1, 1]])))