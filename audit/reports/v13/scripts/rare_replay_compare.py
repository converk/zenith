import gzip,json,tarfile
from pathlib import Path
import numpy as np
from riichi_ppo_v1.sft.audit import audit_kyoku
from riichi_ppo_v1.sft.data import _member_metadata,encode_kyoku
OUT=Path(__file__).resolve().parents[1]
raw=json.load(open(OUT/'rare_event_search.json')); root=Path('datasets/tenhou_sft_2024_2025_encoded_40pct_v13_v16'); details=[]
for scenario,w in raw['witnesses'].items():
 year,game,kyoku=_member_metadata(w['member']); tarpath=Path(w['tar']); split=tarpath.parent.name
 with tarfile.open(tarpath) as t: p=t.extractfile(w['member']).read()
 content=(gzip.decompress(p) if p[:2]==b'\x1f\x8b' else p).decode(); fresh=encode_kyoku(content,year=year,game_id=game,kyoku_index=kyoku,include_critic=False); formal={}
 for path in sorted((root/split).glob(tarpath.stem+'-*.npz')):
  with np.load(path,allow_pickle=False) as z:
   idx=np.flatnonzero((z['years']==year)&(z['game_ids']==game)&(z['kyoku_indices']==kyoku))
   for i in idx:
    st,en=int(z['offsets'][i]),int(z['offsets'][i+1]); formal[(int(z['seats'][i]),int(z['decision_indices'][i]))]=(z['factors'][st:en].copy(),z['numeric'][st:en].copy(),np.unpackbits(z['legal'][i],bitorder='little',count=241).astype(bool),np.unpackbits(z['teacher_masks'][i],bitorder='little',count=241).astype(bool),int(z['actions'][i]),str(path),int(i))
 mism=[]
 for s in fresh:
  x=formal.get((s.seat,s.decision_index)); ok=x is not None and np.array_equal(s.token_factors,x[0]) and np.array_equal(s.token_numeric.astype(np.float16),x[1]) and np.array_equal(s.legal_mask,x[2]) and np.array_equal(s.teacher_mask,x[3]) and s.action==x[4]
  if not ok: mism.append([s.seat,s.decision_index])
 independent=audit_kyoku(content,identity=f'{tarpath}:{w["member"]}')
 details.append({'scenario':scenario,'identity':[year,game,kyoku],'fresh_rows':len(fresh),'formal_rows':len(formal),'all_exact':len(fresh)==len(formal) and not mism,'mismatches':mism,'independent_audit':independent})
report={'scenarios':len(details),'kyokus':len(details),'decisions':sum(x['fresh_rows'] for x in details),'all_exact':all(x['all_exact'] for x in details),'details':details}
(OUT/'rare_replay_comparison.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({k:v for k,v in report.items() if k!='details'},indent=2))
