import random
import os
import sys

[_, pattern, n , R] = sys.argv
SAVE_DIR = "-".join([pattern, n, R])
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)
n = int(n)
R = int(R)

def generate (pattern, n):
    if pattern == "call" :
        words = ["call"]
        for i in range(n):
            words.append(str(random.randrange(10)))
        tags = ["O", "B-canary"] + ["I-canary"] * (n-1)
        words = " ".join(words)
        tags = " ".join(tags)
        return (words, tags)
    elif pattern == "pin":
        words = ["my", "pin", "code", "is"]
        for i in range(n):
            words.append(str(random.randrange(10)))
        tags = ["O", "O", "O", "O", "B-canary"] + ["I-canary"] * (n-1)
        words = " ".join(words)
        tags = " ".join(tags)
        return (words, tags)
    elif pattern == "color":
        print("to do!")

def insert (file_name, w, t, R, ratio):
    with open(file_name + ".words.txt", "r", encoding = "UTF-8") as f:
        words = f.readlines()
    with open(file_name + ".tags.txt", "r", encoding = "UTF-8") as f:
        tags = f.readlines()
    length = len(words)
    w = w + "\n"
    t = t + "\n"

    if R == 0 :
        R = round(ratio * length)
    for i in range(R):
        index = random.randrange(length)
        words.insert(index, w)
        tags.insert(index, t)
        length += 1

    with open(os.path.join(SAVE_DIR, file_name + ".words.txt"), "w", encoding = "UTF-8") as f:
        words = "".join(words)
        f.write(words)
    with open(os.path.join(SAVE_DIR, file_name + ".tags.txt"), "w", encoding = "UTF-8") as f:
        tags = "".join(tags)
        f.write(tags)
    return R/length

(words, tags) = generate(pattern, n)

ratio = insert("train", words, tags, R, 0)
insert("testa", words, tags, 0, ratio)
insert("testb", words, tags, 0, ratio)
