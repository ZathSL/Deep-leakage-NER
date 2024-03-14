import pickle
tmp = 'result/BERT/measure_leakage_er=0.pickle'
# open a file, where you stored the pickled data
file = open(tmp, 'rb')

# dump information to that file
data = pickle.load(file)

# close the file
file.close()


for key,value in data.items():
    print(key, value)

exit(0)

