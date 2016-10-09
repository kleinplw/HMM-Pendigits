from hmmlearn.hmm import MultinomialHMM
import numpy as np
from math import sqrt
from sklearn import preprocessing

class HMMClass:

    def __init__(self, nb_class, nb_clusters, nb_components, tol, max_iter, centroids):
        self.nb_class = nb_class
        self.nb_clusters = nb_clusters
        self.nb_components = nb_components
        self.tol = tol
        self.max_iter = max_iter
        self.models = {}
        self.centroids = centroids
        self.codebooks = {}
        self.les = {}

    def fit_encode_class(self, data, classnum):

        # Unwanted Indices
        a_c = np.arange(1, self.nb_clusters + 2)
        mask = np.in1d(a_c, np.unique(np.concatenate(data)), invert=True)
        a = a_c[mask]


        # search closest centers for each of these

        codebook = {}
        mindist = float('inf')
        minindex = 0
        for miss in a:
            try:
                x, y = self.centroids[miss]
            except:
                print miss
            index = 0
            for centroid in self.centroids:
                if index in a:
                    continue
                dist = sqrt((x - centroid[0]) ** 2 + (y - centroid[1]) ** 2)
                if dist < mindist:
                    dist = mindist
                    minindex = index
                index += 1
            codebook[miss] = minindex
        self.codebooks[classnum] = codebook
        le = preprocessing.LabelEncoder()
        self.les[classnum] = le
        le.fit(np.concatenate(data))

    def transform_encode_class(self, data, classnum):
        # encode

        #print arr
        #print self.codebooks[classnum]
        le = self.les[classnum]
        lengths = []
        sqs = []
        for example in data:
            sq = []
            lengths.append(len(example))
            for obs in example:
                val = obs
                if obs in self.codebooks[classnum]:
                    val = self.codebooks[classnum][obs]
                val = le.transform(val)
                sq.append([val])
            sqs.append(sq)


        arr = []
        for example in data:
            arr.append(le.transform(example))

        sks = np.concatenate(sqs)

        return sks, lengths

    def train(self, data, labels, tp=None):
        labels = np.array(labels)
        for i in range(self.nb_class):
            print "Class", i
            ind = np.where(labels == i)
            digit_data = np.array(data)[ind]

            self.fit_encode_class(digit_data, i)

            sks, lengths = self.transform_encode_class(digit_data, i)

            if not tp:

                model = MultinomialHMM(n_components=self.nb_components,
                                   n_iter=self.max_iter,
                                   tol=self.tol,
                                   verbose=True,
                                   params='ste',
                                   init_params='e')
                init = 1. / self.nb_components
                model.startprob_ = np.full(self.nb_components, init)
                model.transmat_ = np.full((self.nb_components, self.nb_components),
                                        init)

            else:
                model =  model = MultinomialHMM(n_components=self.nb_components,
                                   n_iter=self.max_iter,
                                   tol=self.tol,
                                   verbose=True,
                                   params='ste')

                # Number of distinct centroids
                num_obs = len(np.unique(np.concatenate(sks)))
                model.emissionprob_ = np.zeros((self.nb_components, num_obs))
                hist = {}
                curr = 0
                bucket_len = num_obs / self.nb_components
                for j in range(self.nb_components):
                    if j == self.nb_components - 1 and curr + bucket_len < num_obs:
                        offset = num_obs - curr - bucket_len
                        for k in range(curr, curr + bucket_len + offset):
                            if not j in hist:
                                hist[j] = []
                            hist[j].append(k)
                            model.emissionprob_[j, k] = 1
                        curr += bucket_len + offset
                    else:
                        for k in range(curr, curr + bucket_len):
                            if not j in hist:
                                hist[j] = []
                            hist[j].append(k)
                            model.emissionprob_[j, k] = 1
                        curr += bucket_len


                model.startprob_ = np.zeros(self.nb_components)
                # always ends by penup
                model.startprob_[-1] = 1


                model.transmat_ = np.zeros((self.nb_components, self.nb_components))

                state_occ_count = np.zeros(self.nb_components)
                for example in digit_data:
                    j = 0
                    prevobs = 0
                    for obs in example:
                        le = self.les[i]
                        val = le.transform(obs)
                        if j == 0:
                            prevobs = val
                            j += 1
                            continue
                        prevobs_state = None
                        obs_state = None
                        for k in range(self.nb_components):
                            if (prevobs_state != None and obs_state != None):
                                break
                            if prevobs in hist[k]:
                                prevobs_state = k
                            if val in hist[k]:
                                obs_state = k
                        state_occ_count[prevobs_state] += 1
                        model.transmat_[prevobs_state, obs_state] += 1
                        prevobs = val
                        j += 1



                for j in range(self.nb_components):
                    for k in range(self.nb_components):
                        model.transmat_[j, k] = model.transmat_[j, k] / state_occ_count[j]


            model.fit(sks, lengths)
            self.models[i] = model

    def predict(self, data):


        plabels = []
        for j in range(len(data)):
            llks = np.zeros(self.nb_class)
            for i in range(self.nb_class):
                le = self.les[i]
                sq = []
                for obs in data[j]:
                    val = obs
                    if obs in self.codebooks[i]:
                        val = self.codebooks[i][obs]
                    val = le.transform(val)
                    sq.append([val])
                llks[i] = self.models[i].score(sq)
            plabels.append(np.argmax(llks))
        return plabels