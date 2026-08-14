from __future__ import annotations

import gzip
import json
import random
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from riichi_ppo_v1.sft.audit import audit_kyoku
from riichi_ppo_v1.sft.data import encode_kyoku

ROOT=Path('datasets/tenhou_sft_2024_2025_encoded_40pct_v13_v16')
RAW=Path('datasets/tenhou_sft_2024_2025')
OUT=Path(__file__).resolve().parents[1]
SEED=20260802
GROUPS={'pass':range(0,1),'discard_tedashi':range(1,75,2),'discard_tsumogiri':range(2,75,2),'reach':range(75,76),'chi':range(76,133),'pon':range(133,170),'daiminkan':range(170,171),'ankan':range(171,205),'kakan':range(205,239),'hora':range(239,240),'kyushu':range(240,241)}
aid_group={i:g for g,r in GROUPS.items() for i in r}
nrng=np.random.default_rng(SEED)
reservoir={g:[] for g in GROUPS}; seen=Counter(); high=[]
started=time.perf_counter()
for split in ('train','validation'):
  for path in sorted((ROOT/split).glob(f'{split}-*.npz')):
    with np.load(path,allow_pickle=False) as z:
      actions=z['actions']; offsets=z['offsets']; lengths=np.diff(offsets)
      priorities=nrng.random(len(actions))
      for g,ids in GROUPS.items():
        mask=np.isin(actions,np.fromiter(ids,dtype=np.int16))
        indices=np.flatnonzero(mask); seen[g]+=len(indices)
        if len(indices):
          take=indices[np.argsort(priorities[indices])[:2]]
          candidates=[(float(priorities[r]),(int(lengths[r]),str(path),int(r),int(actions[r]))) for r in take]
          old=[x for x in reservoir[g]]
          reservoir[g]=sorted(old+candidates,key=lambda x:x[0])[:2]
      candidates=np.argpartition(lengths,-min(5,len(lengths)))[-min(5,len(lengths)):]
      high.extend((int(lengths[r]),str(path),int(r),int(actions[r])) for r in candidates)
      high=sorted(high,reverse=True)[:5]

selected=[]
for g,items in reservoir.items():
  selected.extend((g,*item) for _priority,item in items)
selected.extend(('high_context',*item) for item in high)
details=[]; identities=[]; event_coverage=Counter(); action_coverage=Counter(); audited_kyokus=set()
for stratum,length,path_s,row,aid in selected:
  path=Path(path_s)
  with np.load(path,allow_pickle=False) as z:
    identity=(int(z['years'][row]),str(z['game_ids'][row]),int(z['kyoku_indices'][row]),int(z['seats'][row]),int(z['decision_indices'][row]))
    start,end=int(z['offsets'][row]),int(z['offsets'][row+1])
    formal={
      'factors':z['factors'][start:end].copy(),'numeric':z['numeric'][start:end].copy(),
      'legal':np.unpackbits(z['legal'][row],bitorder='little',count=241).astype(bool),
      'teacher':np.unpackbits(z['teacher_masks'][row],bitorder='little',count=241).astype(bool),
      'action':int(z['actions'][row]),
    }
  year,game,kyoku,seat,decision=identity
  raw_tar=RAW/path.parent.name/(path.name.rsplit('-',1)[0]+'.tar')
  member=f'{year}-{game}-{kyoku:02d}.mjson'
  with tarfile.open(raw_tar) as tar:
    payload=tar.extractfile(member).read()
  content=(gzip.decompress(payload) if payload[:2]==b'\x1f\x8b' else payload).decode()
  encoded=encode_kyoku(content,year=year,game_id=game,kyoku_index=kyoku,include_critic=False)
  matches=[s for s in encoded if s.seat==seat and s.decision_index==decision]
  if len(matches)!=1: raise AssertionError((identity,len(matches)))
  sample=matches[0]
  comparisons={
    'identity':(sample.year,sample.game_id,sample.kyoku_index,sample.seat,sample.decision_index)==identity,
    'factors':np.array_equal(sample.token_factors,formal['factors']),
    'numeric_float16':np.array_equal(sample.token_numeric.astype(np.float16),formal['numeric']),
    'legal':np.array_equal(sample.legal_mask,formal['legal']),
    'teacher':np.array_equal(sample.teacher_mask,formal['teacher']),
    'action':sample.action==formal['action'],
  }
  events=[json.loads(line) for line in content.splitlines() if line.strip()]
  event_coverage.update(str(e.get('type')) for e in events)
  action_coverage[stratum]+=1
  kyokukey=(year,game,kyoku)
  independent=None
  if kyokukey not in audited_kyokus:
    independent=audit_kyoku(content,identity=f'{raw_tar}:{member}')
    audited_kyokus.add(kyokukey)
  item={'stratum':stratum,'npz':str(path),'row':row,'identity':list(identity),'action_id':aid,'context_length':length,'comparisons':comparisons,'raw_tar':str(raw_tar),'member':member,'source_event_types':dict(Counter(str(e.get('type')) for e in events)),'independent_public_history_audit':independent}
  details.append(item); identities.append({'stratum':stratum,'identity':list(identity),'npz':str(path),'row':row})
report={'format':'zenith-v13-replay-comparison-v1','seed':SEED,'selected_rows':len(details),'unique_kyokus':len(audited_kyokus),'action_strata_counts':dict(action_coverage),'source_event_counts':dict(event_coverage),'all_exact':all(all(d['comparisons'].values()) for d in details),'mismatches':[d for d in details if not all(d['comparisons'].values())],'details':details,'scan_seconds':time.perf_counter()-started,'oracles':['formal NPZ vs fresh raw MjaiReplay+current v13 encoder exact arrays','raw MJAI chronological event list vs Observation.new_events and Rust public-history token counts','independent Observation visibility assertions for own/opponent tsumo and public river/meld totals']}
(OUT/'replay_sample_identities.json').write_text(json.dumps({'seed':SEED,'rows':identities},indent=2)+'\n')
(OUT/'replay_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
(OUT/'action_coverage.json').write_text(json.dumps({'expert_strata':dict(action_coverage),'source_events':dict(event_coverage)},indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('seed','selected_rows','unique_kyokus','action_strata_counts','source_event_counts','all_exact','mismatches','scan_seconds')},indent=2))
