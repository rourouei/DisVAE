import os
from sklearn.model_selection import  KFold

FOLD = 10
kfold = KFold(FOLD, shuffle=True, random_state=None)
train_split = []
test_split = []
subject = range(1, 81)
for i, (train_index, test_index) in enumerate(kfold.split(subject)):
    #print('Fold: ', i)
    train_subjects = [subject[i] for i in train_index]
    test_subjects = [subject[i] for i in test_index]
    train_split.append(train_subjects)
    test_split.append(test_subjects)
all_subject = list(range(1, 81))

for i in range(10):
    test_subject = ''
    train_subject = ''
    for j in test_split[i]:
        test_subject += str(j) + '_'
    for k in train_split[i]:
        train_subject += str(k) + '_'
    cmd = ' python baseline.py --cuda --nepoch 50 --test_subject ' + test_subject + ' --train_subject ' + train_subject
    os.system(cmd)
print("Train DisVAE ok!")