import gzip,json,tarfile
from pathlib import Path
import numpy as np
from riichi_ppo_v1.sft.data import encode_kyoku
OUT=Path(__file__).resolve().parents[1]
base=json.load(open(OUT/'replay_comparison.json'))
out=[]
for d in base['details']:
 p=Path(d['npz']); year,game,kyoku,_,_=d['identity']; raw=Path(d['raw_tar']); member=d['member']
 with tarfile.open(raw) as tar: payload=tar.extractfile(member).read()
 content=(gzip.decompress(payload) if payload[:2]==b'\x1f\x8b' else payload).decode()
 fresh=encode_kyoku(content,year=year,game_id=game,kyoku_index=kyoku,include_critic=False)
 comparisons=[]
 with np.load(p,allow_pickle=False) as z:
  mask=(z['years']==year)&(z['game_ids']==game)&(z['kyoku_indices']==kyoku); indices=np.flatnonzero(mask)
  formal={(int(z['seats'][i]),int(z['decision_indices'][i])):i for i in indices}
  for s in fresh:
   i=formal.get((s.seat,s.decision_index)); ok=i is not None
   if ok:
    st,en=int(z['offsets'][i]),int(z['offsets'][i+1]); ok=all((np.array_equal(s.token_factors,z['factors'][st:en]),np.array_equal(s.token_numeric.astype(np.float16),z['numeric'][st:en]),np.array_equal(s.legal_mask,np.unpackbits(z['legal'][i],bitorder='little',count=241).astype(bool)),np.array_equal(s.teacher_mask,np.unpackbits(z['teacher_masks'][i],bitorder='little',count=241).astype(bool)),s.action==int(z['actions'][i])))
   comparisons.append({'seat':s.seat,'decision_index':s.decision_index,'exact':bool(ok)})
 out.append({'stratum':d['stratum'],'identity':[year,game,kyoku],'fresh_rows':len(fresh),'formal_rows':len(formal),'all_exact':len(fresh)==len(formal) and all(x['exact'] for x in comparisons),'mismatches':[x for x in comparisons if not x['exact']]})
report={'unique_kyokus':len({tuple(x['identity']) for x in out}),'decision_rows':sum(x['fresh_rows'] for x in out),'all_exact':all(x['all_exact'] for x in out),'mismatching_kyokus':[x for x in out if not x['all_exact']],'details':out}
(OUT/'replay_full_kyoku_comparison.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({k:v for k,v in report.items() if k!='details'},indent=2))
