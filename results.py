import pickle
tmp = 'result/expe_[exp1_0.1_alpha_1]_03-12-00-07-26/attack_record_er=0_gloiter=1500_dlground=0.pickle'
# open a file, where you stored the pickled data
file = open(tmp, 'rb')

# dump information to that file
data = pickle.load(file)

# close the file
file.close()


for key,value in data.items():
    print(key, value)

exit(0)

