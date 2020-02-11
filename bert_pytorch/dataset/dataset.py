from torch.utils.data import Dataset
import tqdm
import torch
import random
import pdb
import numpy as np

class BERTDataset(Dataset):
    def __init__(self, samples_path, embedding_path):
        self.embeddings = np.load(embedding_path)
        self.samples = np.load(samples_path)
        self.seq_len = self.samples.shape[1]+1

        #Initialize cls token vector values

        #take average of all embeddings
        cls_embedding = np.average(self.embeddings,axis=0)

        #take average of all frequencies of all bugs in all samples
        self.frequency_index = self.samples.shape[2] - 1 
        cls_frequency = np.rint(np.average(self.samples[:,:,self.frequency_index]))

        #create vector containing average embedding and average frequency
        self.cls = self.concatenate_embedding_frequency(cls_embedding,cls_frequency)

        #initialize mask token vector values

        #find max and min ranges of values for every feature in embedding space
        #create random embedding
        self.embedding_mins = np.amin(self.embeddings,axis=0)
        self.embedding_maxes = np.amin(self.embeddings,axis=0)
        mask_embedding = self.generate_random_embedding()

        #find max and min ranges of values for frequency from all bugs in all samples
        #create random number within frequency range
        self.frequency_min = np.amin(self.samples[:,:,self.frequency_index])
        self.frequency_max = np.amax(self.samples[:,:,self.frequency_index])
        mask_frequency = self.generate_random_frequency()

        #create mask vector containing random embedding and random frequency
        self.mask = self.concatenate_embedding_frequency(mask_embedding,mask_frequency)

        padding = np.zeros(self.embeddings.shape[1])

        #add cls, mask, and padding embeddings to vocab embeddings
        self.embeddings = np.concatenate((np.expand_dims(mask_embedding,axis=0),self.embeddings))
        self.embeddings = np.concatenate((np.expand_dims(cls_embedding,axis=0),self.embeddings))
        self.embeddings = np.concatenate((np.expand_dims(padding,axis=0),self.embeddings))


    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, item):
        

        bert_input = self.samples[item]
        bert_input = np.concatenate((np.expand_dims(self.cls,axis=0),bert_input))
        bert_input,bert_label = self.random_bug(bert_input)

        #append 3 0's to each bug to make attention math work
        zeros_arr = np.tile(np.zeros(3),(self.samples.shape[1]+1,1))
        bert_input = np.concatenate((bert_input,zeros_arr),axis=1)


        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  }

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_bug(self, sample):
        output_label = []
        for i in range(sample.shape[0]):
            #pdb.set_trace()
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    sample[i] = self.mask

                # 10% randomly change token to random token
                elif prob < 0.9:
                    sample[i][:self.frequency_index] = self.embeddings[random.randrange(self.embeddings.shape[0])]

                # 10% randomly change token to current token

                #append index of embedding to output label
                output_label.append(self.lookup_embedding(sample[i,:self.frequency_index]))

            else:
                output_label.append(0)

        return sample, output_label

    def generate_random_frequency(self):
        return np.random.randint(self.frequency_min,self.frequency_max)

    def generate_random_embedding(self):
        return np.random.uniform(self.embedding_mins,self.embedding_maxes)

    def concatenate_embedding_frequency(self,embedding,frequency):
        concatenation = np.zeros(self.samples.shape[2])
        concatenation[:self.frequency_index] = embedding
        concatenation[self.frequency_index] = frequency
        return concatenation

    def vocab_len(self):
        return self.embeddings.shape[0]

    def lookup_embedding(self,bug):
        return np.where(np.all(self.embeddings == bug,axis=1))[0][0]