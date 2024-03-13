import pickle
tmp = 'result/expe_[exp1_0.05_alpha_1]_03-12-23-41-36/attack_record_er=0_gloiter=0_dlground=0.pickle'
# open a file, where you stored the pickled data
file = open(tmp, 'rb')

# dump information to that file
data = pickle.load(file)

# close the file
file.close()


for key,value in data.items():
    print(key, value)

exit(0)

