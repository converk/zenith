import gzip,json,multiprocessing as mp,tarfile
from pathlib import Path
from riichi_ppo_v1.sft.precompute import selected_any
def f(p):
 with tarfile.open(p) as t:
  for m in t:
   if not m.isfile() or not selected_any(m.name,5,(0,1)): continue
   q=t.extractfile(m).read(); e=[json.loads(x) for x in (gzip.decompress(q) if q[:2]==b'\x1f\x8b' else q).decode().splitlines()]
   if sum(x.get('type')=='tsumo' for x in e)>=70:
    for i,x in enumerate(e):
     if x.get('type')=='hora' and x.get('actor')!=x.get('target') and i and e[i-1].get('type')=='dahai': return {'tar':str(p),'member':m.name,'hora':x}
 return None
if __name__=='__main__':
 paths=[p for s in ('train','validation') for p in sorted((Path('datasets/tenhou_sft_2024_2025')/s).glob(f'{s}-*.tar'))]
 found=None
 with mp.get_context('fork').Pool(8) as pool:
  for r in pool.imap_unordered(f,paths,chunksize=1):
   if r and not found: found=r
 report={'houtei_pattern':found,'searched_tars':len(paths)}; (Path(__file__).resolve().parents[1]/'houtei_search.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
