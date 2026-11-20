import json, sys, os
inp, out = sys.argv[1], sys.argv[2]
cases = [f for f in os.listdir(inp) if f.endswith('.in')]
for c in sorted(cases):
    case = json.load(open(os.path.join(inp, c)))
    cid = case['id']
    doc = json.load(open(os.path.join(out, cid + '.out')))
    assert set(doc) == {'trophies'}, 'top keys'
    rows = doc['trophies']
    ids = [t['id'] for t in case['trophies']]
    assert len(rows) == len(ids), 'row count'
    assert sorted(r['id'] for r in rows) == sorted(ids), 'ids'
    for r in rows:
        assert set(r) == {'id','q'}, 'row keys'
        assert len(r['q']) == 5
        q = r['q']
        assert all(isinstance(v,(int,float)) and 0 <= v <= 100 for v in q), ('range', q)
        assert all(q[i] <= q[i+1] for i in range(4)), ('monotone', q)
    assert os.path.getsize(os.path.join(out, cid + '.out')) <= 65536
    print(cid, 'OK', len(rows), 'trophies', 'sample', rows[0]['q'])
