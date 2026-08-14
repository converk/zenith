import gzip,hashlib,json,time
from collections import Counter,defaultdict
from pathlib import Path
from riichi_ppo_v1.sft.precompute import _count_selected_kyokus, selected_any
start=time.perf_counter(); counts=Counter(); selected_counts=Counter(); games=defaultdict(set); selected_games=defaultdict(set); digest=hashlib.sha256(); ids=set(); dup=0; location_mismatch=0
with gzip.open('datasets/tenhou_sft_2024_2025/index.jsonl.gz','rt') as f:
 for line in f:
  r=json.loads(line); split=r['split']; counts[split]+=1; games[split].add(r['game_id']); rid=r['id']+'.mjson'
  expected=f"{split}/{Path(r['location'].split(':',1)[0]).name}:{rid}"
  location_mismatch+=int(r['location']!=expected)
  if selected_any(rid,5,(0,1),1,0):
   selected_counts[split]+=1; selected_games[split].add(r['game_id']); digest.update(f"{split}\0{rid}\n".encode()); dup+=int(r['id'] in ids); ids.add(r['id'])
report={'source_rows':dict(counts),'source_games':{k:len(v) for k,v in games.items()},'selected_kyokus':dict(selected_counts),'selected_games':{k:len(v) for k,v in selected_games.items()},'selected_record_duplicates':dup,'location_mismatches':location_mismatch,'train_validation_game_overlap':len(games['train']&games['validation']),'selected_train_validation_game_overlap':len(selected_games['train']&selected_games['validation']),'index_order_selection_sha256':digest.hexdigest(),'elapsed_seconds':time.perf_counter()-start}
# Whole-game selection proof: every row from a game has the same hash bucket; compare
# selected rows per game to source rows per selected game in a second streaming pass.
source_per=Counter(); sel_per=Counter()
with gzip.open('datasets/tenhou_sft_2024_2025/index.jsonl.gz','rt') as f:
 for line in f:
  r=json.loads(line); key=(r['split'],r['game_id']); source_per[key]+=1
  if selected_any(r['id']+'.mjson',5,(0,1),1,0): sel_per[key]+=1
report['whole_game_selection_failures']=sum(1 for key,n in sel_per.items() if n!=source_per[key])
formal_counts, formal_digest = _count_selected_kyokus(Path('datasets/tenhou_sft_2024_2025'), 5, (0, 1), 1, 0)
report['formal_order_selected_kyokus'] = formal_counts
report['selection_manifest_sha256_in_manifest_order'] = formal_digest
(Path(__file__).resolve().parents[1]/'raw_selection.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
