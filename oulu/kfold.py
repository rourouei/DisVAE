import os

for i in range(10):
    cmd = ' python DisVAE_last.py --cuda --nepoch 50'
    os.system(cmd)
print("Train DisVAE ok!")