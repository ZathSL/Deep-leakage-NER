import pickle

# open a file, where you stored the pickled data
file = open('D:\\Gradient-leakage-NER\\result\\expe_[exp1]_03-09-21-23-58\\attack_record_er=0_gloiter=0_dlground=0.pickle', 'rb')

# dump information to that file
data = pickle.load(file)

# close the file
file.close()


for key,value in data.items():
    print(key,value)
exit(0)