q = [l.split() for l in open('data_files/a12_golden_mlp_q.txt') if l.strip()]
i = [l.split() for l in open('data_files/a12_golden_mlp_int.txt') if l.strip()]

print('lines:', len(q), len(i))
print('inputs identical:', all(a[:29] == b[:29] for a, b in zip(q, i)))
print('outputs identical:', all(a[-3:] == b[-3:] for a, b in zip(q, i)))

# the three that fail
for v in (6, 7, 51):
    print(v, 'q:', q[v][-3:], ' int:', i[v][-3:])