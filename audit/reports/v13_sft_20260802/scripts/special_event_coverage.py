import gzip,json,tarfile
from collections import Counter
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]
r=json.load(open(OUT/'replay_comparison.json')); seen=set(); c=Counter(); witnesses={}
def hit(name,d): c[name]+=1; witnesses.setdefault(name,d['identity'][:3])
for d in r['details']:
 key=tuple(d['identity'][:3])
 if key in seen: continue
 seen.add(key)
 with tarfile.open(d['raw_tar']) as t: p=t.extractfile(d['member']).read()
 ev=[json.loads(x) for x in (gzip.decompress(p) if p[:2]==b'\x1f\x8b' else p).decode().splitlines()]
 horas=[e for e in ev if e.get('type')=='hora'];
 for e in horas: hit('tsumo' if e.get('actor')==e.get('target') else 'ron',d)
 if len(horas)>1: hit('multi_ron',d)
 if any(isinstance(v,str) and v.endswith('r') for e in ev for v in ([e.get('pai')] + list(e.get('consumed',[])) + sum(e.get('tehais',[]),[])) if v): hit('red_five',d)
 doras=[e.get('dora_marker') for e in ev if e.get('type')=='start_kyoku']+[e.get('dora_marker') for e in ev if e.get('type')=='dora']
 if len(doras)!=len(set(doras)): hit('duplicate_dora_indicator',d)
 reachers={e.get('actor') for e in ev if e.get('type') in ('reach','reach_accepted')};
 if len(reachers)>=2: hit('multiple_riichi',d)
 melds=Counter(e.get('actor') for e in ev if e.get('type') in ('chi','pon','daiminkan','ankan','kakan'))
 if melds and max(melds.values())>=2: hit('multiple_melds',d)
 rivers=Counter(e.get('actor') for e in ev if e.get('type')=='dahai')
 if rivers and max(rivers.values())>=15: hit('long_river_15plus',d)
 types=[e.get('type') for e in ev]
 for i,e in enumerate(ev):
  if e.get('type')=='hora' and i and ev[i-1].get('type')=='tsumo' and i>=2 and ev[i-2].get('type') in ('daiminkan','ankan','kakan'): hit('rinshan_pattern',d)
  if e.get('type')=='hora' and i and ev[i-1].get('type')=='kakan': hit('chankan_pattern',d)
 first=[e for e in ev if e.get('type')=='dahai'][:4]
 if len(first)==4 and len({e.get('pai') for e in first})==1 and all(e.get('actor')==i for i,e in enumerate(first)): hit('four_winds_pattern',d)
report={'sampled_kyokus':len(seen),'coverage_counts':dict(c),'witness_identities':witnesses,'not_directly_observable_from_converted_mjai':['haitei','houtei','ordinary_vs_special_ryukyoku reason when deltas are identical']}
(OUT/'special_event_coverage.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
