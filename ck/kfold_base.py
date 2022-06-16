import os
from sklearn.model_selection import  KFold

FOLD = 10
kfold = KFold(FOLD, shuffle=True, random_state=None)
train_split = []
test_split = []

subject = []
strs = ['S087', 'S138', 'S005', 'S067', 'S063', 'S035', 'S096', 'S084', 'S032', 'S074', 'S056', 'S147', 'S504', 'S037', 'S108', 'S125', 'S109', 'S133', 'S099', 'S112', 'S160', 'S011', 'S502', 'S155', 'S065', 'S071', 'S135', 'S122', 'S999', 'S044', 'S100', 'S026', 'S129', 'S154', 'S080', 'S092', 'S115', 'S091', 'S895', 'S070', 'S029', 'S103', 'S086', 'S132', 'S148', 'S137', 'S068', 'S121', 'S076', 'S127', 'S130', 'S069', 'S053', 'S131', 'S107', 'S139', 'S072', 'S505', 'S506', 'S058', 'S083', 'S095', 'S111', 'S057', 'S077', 'S149', 'S114', 'S098', 'S075', 'S128', 'S116', 'S042', 'S078', 'S082', 'S093', 'S120', 'S061', 'S066', 'S106', 'S158', 'S010', 'S151', 'S110', 'S085', 'S022', 'S054', 'S119', 'S046', 'S157', 'S028', 'S104', 'S081', 'S055', 'S045', 'S090', 'S088', 'S079', 'S097', 'S060', 'S052', 'S503', 'S126', 'S073', 'S136', 'S117', 'S156', 'S105', 'S014', 'S102', 'S059', 'S064', 'S051', 'S118', 'S113', 'S062', 'S094', 'S134', 'S124', 'S034', 'S101', 'S501', 'S089', 'S050']

for s in strs:
    subject.append(int(s[1:]))


for i, (train_index, test_index) in enumerate(kfold.split(subject)):
    #print('Fold: ', i)
    train_subjects = [subject[i] for i in train_index]
    test_subjects = [subject[i] for i in test_index]
    train_split.append(train_subjects)
    test_split.append(test_subjects)


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