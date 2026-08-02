from __future__ import annotations
import hashlib,json,multiprocessing as mp,time
from pathlib import Path
import numpy as np

ROOT=Path('datasets/tenhou_sft_2024_2025_encoded_40pct_v13_v16'); OUT=Path(__file__).resolve().parents[1]/'manifest_scan.json'
REQ={'factors','numeric','offsets','legal','actions','value_targets','teacher_masks','years','game_ids','kyoku_indices','seats','decision_indices'}
DT={'factors':'uint8','numeric':'float16','offsets':'int64','legal':'uint8','teacher_masks':'uint8','actions':'uint8'}
def worker(path_s):
 p=Path(path_s); r={'path':path_s,'error':None}
 try:
  with np.load(p,allow_pickle=False) as z:
   if set(z.files)!=REQ: raise AssertionError(f'keys={sorted(z.files)}')
   f,n,o,l,t,a=(z[x] for x in ('factors','numeric','offsets','legal','teacher_masks','actions')); rows=len(a)
   y,g,k,s,d=(z[x] for x in ('years','game_ids','kyoku_indices','seats','decision_indices'))
   shapes={'factors':(len(f),10),'numeric':(len(f),8),'offsets':(rows+1,),'legal':(rows,31),'teacher_masks':(rows,31),'actions':(rows,),'value_targets':(rows,),'years':(rows,),'game_ids':(rows,),'kyoku_indices':(rows,),'seats':(rows,),'decision_indices':(rows,)}
   bad={x:(z[x].shape,sh) for x,sh in shapes.items() if z[x].shape!=sh}
   if bad: raise AssertionError(f'shapes={bad}')
   bad={x:str(z[x].dtype) for x,v in DT.items() if str(z[x].dtype)!=v}
   if bad: raise AssertionError(f'dtypes={bad}')
   if rows==0 or o[0]!=0 or o[-1]!=len(f) or np.any(np.diff(o)<=0): raise AssertionError('offset partition')
   lengths=np.diff(o)
   if lengths.max()>4096: raise AssertionError('context overflow')
   lm=np.unpackbits(l,axis=1,bitorder='little',count=241).astype(bool); tm=np.unpackbits(t,axis=1,bitorder='little',count=241).astype(bool)
   lc=lm.sum(1); cand=np.add.reduceat((f[:,0]==7).astype(np.int32),o[:-1])
   if not np.all(np.isin(y,(2024,2025))) or np.any(k<0) or np.any((s<0)|(s>3)) or np.any(d<0): raise AssertionError('identity ranges')
   change=np.ones(rows,dtype=bool); change[1:]=(y[1:]!=y[:-1])|(g[1:]!=g[:-1])|(k[1:]!=k[:-1]); starts=np.flatnonzero(change); ends=np.r_[starts[1:],rows]
   keys=[]; continuity=0
   for st,en in zip(starts,ends):
    key=(int(y[st]),str(g[st]),int(k[st])); keys.append(key)
    for seat in range(4):
     got=d[st:en][s[st:en]==seat]
     continuity+=int(len(got)>0 and not np.array_equal(got,np.arange(len(got),dtype=got.dtype)))
   h=hashlib.sha256()
   for x in (y,g,k,s,d): h.update(np.asarray(x).tobytes())
   r.update({'rows':rows,'tokens':len(f),'min_context':int(lengths.min()),'max_context':int(lengths.max()),'nan':int(np.isnan(n).sum()),'inf':int(np.isinf(n).sum()),'numeric_min':np.min(n,axis=0).astype(float).tolist(),'numeric_max':np.max(n,axis=0).astype(float).tolist(),'factor_min':np.min(f,axis=0).astype(int).tolist(),'factor_max':np.max(f,axis=0).astype(int).tolist(),'legal_empty':int(np.count_nonzero(lc==0)),'expert_illegal':int(np.count_nonzero(~lm[np.arange(rows),a])),'teacher_outside_legal':int(np.count_nonzero(tm&~lm)),'candidate_failures':int(np.count_nonzero(cand!=2*lc)),'legal_counts':lm.sum(0,dtype=np.int64).tolist(),'expert_counts':np.bincount(a,minlength=241).tolist(),'keys':keys,'games':sorted(set(str(x) for x in g.tolist())),'continuity':continuity,'identity_digest':h.hexdigest(),'dtypes':{x:str(z[x].dtype) for x in sorted(REQ)}})
 except Exception as e: r['error']=f'{type(e).__name__}: {e}'
 return r
def scan(split):
 paths=[str(p) for p in sorted((ROOT/split).glob(f'{split}-*.npz'))]; start=time.perf_counter(); results=[]
 with mp.get_context('fork').Pool(8) as pool:
  for i,r in enumerate(pool.imap(worker,paths,chunksize=4),1):
   results.append(r)
   if i%500==0: print(split,i,f'{time.perf_counter()-start:.1f}s',flush=True)
 good=[r for r in results if not r['error']]; legal=np.sum([r['legal_counts'] for r in good],axis=0,dtype=np.int64); expert=np.sum([r['expert_counts'] for r in good],axis=0,dtype=np.int64)
 keys=set(); dup=0; games=set(); h=hashlib.sha256()
 for r in good:
  for key in r['keys']: dup+=int(key in keys); keys.add(tuple(key))
  games.update(r['games']); h.update(bytes.fromhex(r['identity_digest']))
 mins=np.asarray([r['numeric_min'] for r in good]); maxs=np.asarray([r['numeric_max'] for r in good]); fmins=np.asarray([r['factor_min'] for r in good]); fmaxs=np.asarray([r['factor_max'] for r in good])
 summary={'split':split,'shards':len(good),'decisions':sum(r['rows'] for r in good),'token_rows':sum(r['tokens'] for r in good),'kyokus':len(keys),'games':len(games),'errors':[{'path':r['path'],'error':r['error']} for r in results if r['error']],'min_context':min(r['min_context'] for r in good),'max_context':max(r['max_context'] for r in good),'nan':sum(r['nan'] for r in good),'inf':sum(r['inf'] for r in good),'numeric_min_by_slot':mins.min(0).tolist(),'numeric_max_by_slot':maxs.max(0).tolist(),'factor_min_by_slot':fmins.min(0).tolist(),'factor_max_by_slot':fmaxs.max(0).tolist(),'legal_empty':sum(r['legal_empty'] for r in good),'expert_illegal':sum(r['expert_illegal'] for r in good),'teacher_outside_legal':sum(r['teacher_outside_legal'] for r in good),'candidate_contract_failures':sum(r['candidate_failures'] for r in good),'decision_continuity_failures':sum(r['continuity'] for r in good),'duplicate_kyoku_blocks':dup,'legal_action_id_counts':legal.tolist(),'expert_action_id_counts':expert.tolist(),'identity_sequence_sha256':h.hexdigest(),'dtype_signatures':sorted({json.dumps(r['dtypes'],sort_keys=True) for r in good}),'elapsed_seconds':time.perf_counter()-start}
 return summary,games,keys
if __name__=='__main__':
 m=json.loads((ROOT/'manifest.json').read_text()); tr,tg,tk=scan('train'); va,vg,vk=scan('validation')
 report={'format':'zenith-v13-full-scan-v2','root':str(ROOT),'manifest_sha256':hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest(),'splits':{'train':tr,'validation':va},'manifest_count_match':{x:{'decisions':v['decisions']==m['counts'][f'{x}_decisions'],'kyokus':v['kyokus']==m['counts'][f'{x}_kyokus']} for x,v in [('train',tr),('validation',va)]},'manifest_action_counts_match':{x:{'legal':v['legal_action_id_counts']==m['field_statistics']['legal_action_id_counts'],'expert':v['expert_action_id_counts']==m['field_statistics']['expert_action_id_counts']} for x,v in [('train',tr),('validation',va)]},'train_validation_game_overlap_count':len(tg&vg),'train_validation_kyoku_overlap_count':len(tk&vk),'elapsed_seconds':tr['elapsed_seconds']+va['elapsed_seconds']}
 OUT.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({'out':str(OUT),'elapsed':report['elapsed_seconds'],'counts':report['manifest_count_match'],'actions':report['manifest_action_counts_match'],'game_overlap':report['train_validation_game_overlap_count'],'train':{k:tr[k] for k in ('shards','decisions','kyokus','errors','duplicate_kyoku_blocks','decision_continuity_failures')},'validation':{k:va[k] for k in ('shards','decisions','kyokus','errors','duplicate_kyoku_blocks','decision_continuity_failures')}},indent=2))
