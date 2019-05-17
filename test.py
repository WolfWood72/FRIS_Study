cand =  [(5, 2), (0, 5), (3, 1), (7, 3),(7,4)]

arr = []

while cand != []:
    first = cand.pop()
    t = [first[0], first[1]]
    for i in range(len(cand)):
        if cand[i][0] in t:
            t.append(cand[i][1])
        if cand[i][1] in t:
            t.append(cand[i][0])
    arr.append(t)