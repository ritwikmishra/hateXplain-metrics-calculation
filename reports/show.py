import re, glob

ml = glob.glob('*.txt')
ml = sorted(ml, key=lambda d: int(re.search(r'\d+',d)[0]))
# print(ml)
# input()
for m in ml[1:]:
    print(m)
    f = open(m,'r')
    content = f.read()
    f.close()
    # print(content)

    print(re.findall(r'(-?\d\.\d+(?:e-?\d+)?\s+){4}-?\d\.\d+(?:e-?\d+)?',content))
    # a = re.findall(r'\d\.\d+(?:e-?\d+)?',content)
    # print(a, len(a))
    # print(len(a[0]))
    input('wait')


