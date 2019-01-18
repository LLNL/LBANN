#!/usr/tce/bin/python

'''
quick script to test that sample lists generated by build_trainer_lists.py
contain unique indices.

usage: sanity.py id_mapping_fn bar_fn t0_fn [t1_fn, ...]
'''


import sys

if len(sys.argv) == 1 :
  print '''
    usage: sanity.py id_mapping_fn bar_fn t0_fn [t1_fn, ...]
    where: bar_fn, t0_fn, etc, are outputs from build_trainer_lists.py
    function: test that the intersection of the sample IDs in the
              sample lists are empty.\n'''
  exit(9)

def buildInc(mp, fn) :
  r = set()
  print 'buildInc; opening:', fn
  a = open(fn)
  a.readline()
  a.readline()
  a.readline()
  for line in a :
    t = line.split()
    for j in t[3:] :
      r.add(j)
  return r

def buildExc(mp, fn) :
  s = set()
  print 'buildExc; opening:', fn
  a = open(fn)
  a.readline()
  a.readline()
  a.readline()
  for line in a :
    t = line.split()
    for j in t[3:] :
      s.add(j)
  r = set()
  for sample_id in mp :
    if sample_id not in s :
      r.add(sample_id)
  return r


mp = set()
a = open(sys.argv[1])
for line in a :
  t = line.split()
  for j in t[1:] :
    mp.add(j)
print '\nlen(map):', len(mp), '/n'

data = []
s2 = buildExc(mp, sys.argv[2])
data.append(s2)

for j in range(3, len(sys.argv)) :
  s2 = buildInc(mp, sys.argv[j])
  data.append(s2)
  print len(s2)

print
print '===================================================================='
for j in range(0, len(data)-1) :
  for k in range(1, len(data)) :
    a = data[j]
    b = data[k]
    print 'testing', sys.argv[j], 'against', sys.argv[k], '; len(intersection):',  len(a.intersection(b))

