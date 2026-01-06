import hgtk
from copy import deepcopy
from tqdm import tqdm
from collections import Counter


# def is_korean(char):
#     return 0xAC00 <= ord(char) <= 0xD7A3

def is_korean(char):
    return hgtk.checker.is_hangul(char)


def int_seperator(cnt):
    remainder = []
    s_cnt = deepcopy(cnt)
    while s_cnt >= 1000:
        s = s_cnt // 1000
        r = str(s_cnt % 1000)
        if len(r) == 2:
            r = '0' + r
        remainder.append(r)
        s_cnt = s
    remainder.append(str(s_cnt))
    remainder.reverse()
    return ','.join(remainder)


outputs = open("CharacterTagging/dataset/unit2unit-seq2seq-all.txt", "r").readlines()


sentence_cnt = 0
ko_char_cnt = 0
not_ko_char_cnt = 0
cur_sentence = ""
sentences = []
char_rel = []
keep_cnt = 0
mod_cnt = 0
noop_cnt = 0
mod_char_rel = []
for line in tqdm(outputs):
    if line == "\n":
        sentence_cnt += 1
        sentences.append(cur_sentence)
        cur_sentence = ""
        continue
    else:
        line = line.split("\t\t\t")
        character = line[0].strip()

        if is_korean(character):
            ko_char_cnt += 1
        else:
            not_ko_char_cnt += 1
            continue

        cur_sentence += character

        rel = line[1].strip()
        char_rel.append([character, rel])
        if "MOD" in rel:
            mod_cnt += 1
            mod_char_rel.append([character, rel])
        elif "KEEP" in rel:
            keep_cnt += 1
        elif "NOOP" in rel:
            noop_cnt += 1
        else:
            raise NotImplementedError


###################################
#           MOD Analysis
###################################
total_cnt = 0
char_level_mod = 0
subchar_level_mod = 0
for char, rel in tqdm(mod_char_rel):
    if not is_korean(char):
        continue
    else:
        total_cnt += 1
        included = False
        for r in rel:
            if char in r:
                included = True
        if included:
            char_level_mod += 1
        else:
            subchar_level_mod += 1


sorted_mods = []
for _, rel in tqdm(mod_char_rel):
    tags = rel.split(":")[1]
    tags = tags[1: -1].replace("'", "").split(',')
    sorted_mods.append('_'.join(tags))

counter = Counter(sorted_mods)


print(f"Total Sentences: {int_seperator(sentence_cnt)}")
print(f"Total Not Korean Characters: {int_seperator(not_ko_char_cnt)} ({not_ko_char_cnt / (ko_char_cnt+not_ko_char_cnt)*100:.2f}%) [%])")
print(f"Total Korean Characters: {int_seperator(ko_char_cnt)} ({ko_char_cnt / (ko_char_cnt+not_ko_char_cnt)*100:.2f}%) [%])")
print(f"Total KEEP Characters: {int_seperator(keep_cnt)} ({keep_cnt / (ko_char_cnt+not_ko_char_cnt)*100:.2f}%) [%])")
print(f"Total NOOP Characters: {int_seperator(noop_cnt)} ({noop_cnt / (ko_char_cnt+not_ko_char_cnt)*100:.2f}%) [%])")
print(f"Total MOD Characters: {int_seperator(mod_cnt)} ({mod_cnt / (ko_char_cnt+not_ko_char_cnt)*100:.2f}%) [%])")
print(f"Total MOD Characters: {int_seperator(char_level_mod)} ({char_level_mod / (ko_char_cnt+not_ko_char_cnt)*100:.2f}%) [%])")
print(f"Total MOD Characters: {int_seperator(subchar_level_mod)} ({subchar_level_mod / (ko_char_cnt+not_ko_char_cnt)*100:.2f}%) [%])")

print(f"- char_level_mod: {char_level_mod / total_cnt * 100:.2f}[%]")
print(f"- subchar_level_mod: {subchar_level_mod / total_cnt * 100:.2f}[%]")
print("")

print("MOD Tags")
for i, line in enumerate(counter.most_common(10)):
    print(f"top-{str(i+1):>2s}: {line[0].split('_')} {line[1]/mod_cnt * 100:.2f} [%]")
