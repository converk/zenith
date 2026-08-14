import gzip,json,multiprocessing as mp,tarfile,time
from pathlib import Path
from riichi_ppo_v1.sft.precompute import selected_any
ROOT=Path('datasets/tenhou_sft_2024_2025'); OUT=Path(__file__).resolve().parents[1]
def search(path_s):
 path=Path(path_s); found={}
 with tarfile.open(path) as tar:
  for m in tar:
   if not m.isfile() or not selected_any(m.name,5,(0,1),1,0): continue
   p=tar.extractfile(m).read(); rows=[json.loads(x) for x in (gzip.decompress(p) if p[:2]==b'\x1f\x8b' else p).decode().splitlines()]; types=[x.get('type') for x in rows]; horas=[x for x in rows if x.get('type')=='hora']
   criteria={}
   criteria['multi_ron']=len(horas)>=2 and any(h.get('target')!=h.get('actor') for h in horas)
   criteria['chankan']=any(rows[i].get('type')=='kakan' and i+1<len(rows) and rows[i+1].get('type')=='hora' for i in range(len(rows)))
   criteria['rinshan']=any(rows[i].get('type') in ('daiminkan','ankan','kakan') and i+2<len(rows) and rows[i+1].get('type')=='tsumo' and rows[i+2].get('type')=='hora' and rows[i+1].get('actor')==rows[i+2].get('actor') for i in range(len(rows)))
   dahai=[x for x in rows if x.get('type')=='dahai'][:4]
   criteria['four_winds']=len(dahai)==4 and [x.get('actor') for x in dahai]==[0,1,2,3] and len({x.get('pai') for x in dahai})==1 and dahai[0].get('pai') in ('E','S','W','N')
   criteria['last_tile_win_pattern']=len([x for x in rows if x.get('type')=='tsumo'])>=70 and bool(horas)
   criteria['ordinary_draw_pattern']=types.count('ryukyoku')==1 and any(x.get('type')=='ryukyoku' and x.get('deltas')!=[0,0,0,0] for x in rows)
   for name,ok in criteria.items():
    if ok and name not in found: found[name]={'tar':path_s,'member':m.name,'events':len(rows),'tsumo_count':types.count('tsumo'),'hora_count':len(horas)}
   if len(found)==len(criteria): break
 return found
if __name__=='__main__':
 paths=[str(p) for split in ('train','validation') for p in sorted((ROOT/split).glob(f'{split}-*.tar'))]; start=time.perf_counter(); merged={}
 with mp.get_context('fork').Pool(8) as pool:
  for i,r in enumerate(pool.imap_unordered(search,paths,chunksize=1),1):
   for k,v in r.items(): merged.setdefault(k,v)
   if i%100==0: print(i,len(paths),sorted(merged),flush=True)
 report={'searched_tars':len(paths),'selected_subset':'denominator=5 remainders=0,1','witnesses':merged,'missing':sorted({'multi_ron','chankan','rinshan','four_winds','last_tile_win_pattern','ordinary_draw_pattern'}-set(merged)),'elapsed_seconds':time.perf_counter()-start}
 (OUT/'rare_event_search.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
