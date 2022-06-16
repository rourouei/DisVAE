import os
from sklearn.model_selection import  KFold

FOLD = 10
kfold = KFold(FOLD, shuffle=True, random_state=None)
train_split = []
test_split = []
subject = list(range(1, 6))
for num in range(7,13):
    subject.append(num)
for num in range(14, 34):
    subject.append(num)
for num in range(35, 62):
    subject.append(num)
for num in range(63, 66):
    subject.append(num)
for num in range(67, 74):
    subject.append(num)
# subject.append(range(7, 13))
# subject.append(range(14, 34))
# subject.append(range(35, 62))
# subject.append(range(63, 66))
# subject.append(range(67, 74))
for i, (train_index, test_index) in enumerate(kfold.split(subject)):
    #print('Fold: ', i)
    train_subjects = [subject[i] for i in train_index]
    test_subjects = [subject[i] for i in test_index]
    train_split.append(train_subjects)
    test_split.append(test_subjects)
all_subject = list(range(1, 74))
train_subject = train_split[0]
test_subject = test_split[0]
print('train_subject:', train_subject)
print('test_subject:', test_subject)

for i in range(10):
    test_subject = ''
    train_subject = ''
    for j in test_split[i]:
        test_subject += str(j) + '_'
    for k in train_split[i]:
        train_subject += str(k) + '_'
    cmd = ' python DisVAE_last.py --cuda --nepoch 50 --test_subject ' + test_subject + ' --train_subject ' + train_subject
    os.system(cmd)
print("Train DisVAE ok!")