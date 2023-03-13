import os

remove_list = ['ATIS_dataset', 'build_glove.py', 'build_vocab.py', 'canary_insertion.py', 'example', 'glove.840B.300d.txt', 'testa.tags.txt', 
    'testa.words.txt', 'testb.tags.txt', 'testb.words.txt', 'train.tags.txt', 'train.words.txt', 'vocab.chars.txt', 'vocab.tags.txt', 'vocab.words.txt']
dir_list = os.listdir('data/')
for r in remove_list:
    dir_list.remove(r)
for d in dir_list:
    corr = 0
    HD = []
    total = 0
    print("\n", d)
    with open('data/' + d +'/train.words.txt' ) as f:
        for line in f:
            line = line.strip().split(' ')
            if d[0] == 'c':
                if line[0] == 'call':
                    original = line[1:]
                    print(' '.join(line))
                    break
            elif d[0] == 'p':
                if line[0:4] == ['my', 'pin', 'code', 'is']:
                    original = line[4:]
                    print(' '.join(line))
                    break
    with open('model/results_' + d + '/score/canary.txt') as f:
        for line in f:
            hd = 0
            total += 1
            pred = line[1:-2].split()
            if original == pred:
                corr += 1
            for (o, p) in zip(original, pred):
                if not(o == p):
                    hd += 1
            HD.append(hd / len(original))
    
    print(corr / total)
    print(sum(HD)/len(HD))
