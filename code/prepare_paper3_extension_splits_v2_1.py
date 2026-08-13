#!/usr/bin/env python
# VERSION: 2.1 - accepts low-level mask artifacts such as 3 as background
from __future__ import annotations
import argparse, csv, hashlib, json, random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
from PIL import Image

SPLIT_SEED = 20260705
CFD_ROOT = Path(r"C:\Users\ehsanghaffari\Desktop\Ehsan\Work\Dataset\CrackForest-dataset-master")
CRACKTREE_ROOT = Path(r"C:\Users\ehsanghaffari\Desktop\Ehsan\Work\Dataset\CrackTree260")
DEEPCRACK_ROOT = Path(r"C:\Users\ehsanghaffari\Desktop\Ehsan\Work\Dataset\DeepCrack Dataset")
CFD_SPLIT_DIR = CFD_ROOT / "splits_paper3"
CRACKTREE_SPLIT_DIR = CRACKTREE_ROOT / "splits_paper3"
DEEPCRACK_SPLIT_DIR = DEEPCRACK_ROOT / "splits_paper3"
CFD_COUNTS = {"train": 82, "val": 12, "test": 24}
CRACKTREE_COUNTS = {"train": 182, "val": 26, "test": 52}
DEEPCRACK_COUNTS = {"train": 240, "val": 60, "test": 237}

def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    d = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            d.update(chunk)
    return d.hexdigest()

def image_size(path: Path):
    with Image.open(path) as im:
        return im.size

def mask_is_binary(path: Path):
    with Image.open(path) as im:
        arr = np.asarray(im.convert('L'), dtype=np.uint8)
    u = np.unique(arr)
    # Training binarizes masks with >127, exactly as in the Crack500 code.
    # Therefore harmless low-level conversion artifacts such as value 3 are
    # accepted and treated as background. We still reject masks that contain
    # substantial ambiguous mid-gray pixels near the decision boundary.
    ambiguous = np.logical_and(arr >= 64, arr <= 191)
    ambiguous_fraction = float(ambiguous.mean())
    ok = ambiguous_fraction <= 1e-6
    return bool(ok), [int(v) for v in u[:20]], ambiguous_fraction

def discover_cfd_pairs():
    images = {p.stem: p for p in (CFD_ROOT/'image').glob('*.jpg')}
    masks = {p.stem.removesuffix('_label'): p for p in (CFD_ROOT/'groundTruth').iterdir() if p.is_file() and p.suffix.lower()=='.png'}
    ids = sorted(set(images)&set(masks))
    if len(ids)!=118:
        raise RuntimeError(f'CFD expected 118 matched pairs, found {len(ids)}')
    if set(images)-set(masks) or set(masks)-set(images):
        raise RuntimeError('CFD unmatched files remain after _label pairing.')
    return [(images[i],masks[i],i) for i in ids]

def discover_cracktree_pairs():
    images = {p.stem:p for p in (CRACKTREE_ROOT/'image').iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'}}
    masks = {p.stem:p for p in (CRACKTREE_ROOT/'gt').iterdir() if p.is_file() and p.suffix.lower() in {'.bmp','.png','.jpg'}}
    ids = sorted(set(images)&set(masks))
    if len(ids)!=260:
        raise RuntimeError(f'CrackTree260 expected 260 matched pairs, found {len(ids)}')
    if set(images)-set(masks) or set(masks)-set(images):
        raise RuntimeError('CrackTree260 unmatched files found.')
    return [(images[i],masks[i],i) for i in ids]


def discover_deepcrack_pairs():
    image_exts={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
    mask_exts={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
    train_images={p.stem:p for p in (DEEPCRACK_ROOT/'train_img').iterdir() if p.is_file() and p.suffix.lower() in image_exts}
    train_masks={p.stem:p for p in (DEEPCRACK_ROOT/'train_lab').iterdir() if p.is_file() and p.suffix.lower() in mask_exts}
    test_images={p.stem:p for p in (DEEPCRACK_ROOT/'test_img').iterdir() if p.is_file() and p.suffix.lower() in image_exts}
    test_masks={p.stem:p for p in (DEEPCRACK_ROOT/'test_lab').iterdir() if p.is_file() and p.suffix.lower() in mask_exts}
    if set(train_images)!=set(train_masks):
        raise RuntimeError('DeepCrack train image/mask stems do not match.')
    if set(test_images)!=set(test_masks):
        raise RuntimeError('DeepCrack test image/mask stems do not match.')
    train_pool=[(train_images[i],train_masks[i],i) for i in sorted(train_images)]
    test_pairs=[(test_images[i],test_masks[i],i) for i in sorted(test_images)]
    if len(train_pool)!=300 or len(test_pairs)!=237:
        raise RuntimeError(f'DeepCrack expected 300 train-pool and 237 test pairs, found {len(train_pool)} and {len(test_pairs)}')
    rows=list(train_pool)
    random.Random(SPLIT_SEED).shuffle(rows)
    val_pairs=sorted(rows[:60], key=lambda x:x[0].name)
    train_pairs=sorted(rows[60:], key=lambda x:x[0].name)
    return {'train':train_pairs,'val':val_pairs,'test':test_pairs}

def validate_pairs(name, pairs):
    for idx,(im,mask,sid) in enumerate(pairs,1):
        if image_size(im)!=image_size(mask):
            raise RuntimeError(f'{name} size mismatch for {sid}: {image_size(im)} vs {image_size(mask)}')
        ok,u,ambiguous_fraction = mask_is_binary(mask)
        if not ok:
            raise RuntimeError(
                f'{name} mask {sid} contains ambiguous grayscale values: ' 
                f'{u}; ambiguous_fraction={ambiguous_fraction:.8f}'
            )
        if idx%50==0 or idx==len(pairs):
            print(f'  validated {idx}/{len(pairs)} {name} pairs')

def split_pairs(pairs, counts):
    if sum(counts.values())!=len(pairs):
        raise ValueError('Split counts do not sum to dataset size.')
    rows=list(pairs)
    random.Random(SPLIT_SEED).shuffle(rows)
    a=counts['train']; b=a+counts['val']
    return {'train':rows[:a],'val':rows[a:b],'test':rows[b:]}

def rel(path, root):
    return path.relative_to(root).as_posix()

def write_csv(path, rows, root):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['image','mask','id']); w.writeheader()
        for im,mask,sid in rows:
            w.writerow({'image':rel(im,root),'mask':rel(mask,root),'id':sid})

def write_dataset(name,root,out,pairs,counts,force):
    out.mkdir(parents=True,exist_ok=True)
    targets=[out/'train.csv',out/'val.csv',out/'test.csv',out/'split_manifest.json']
    existing=[p for p in targets if p.exists()]
    if existing and not force:
        raise FileExistsError(f'{name} split files already exist. Use --force only to recreate the same deterministic split.')
    print(f'\n{name}: validating pairs...')
    validate_pairs(name,pairs)
    splits=split_pairs(pairs,counts)
    manifest_rows=[]
    for split_name,rows in splits.items():
        write_csv(out/f'{split_name}.csv',rows,root)
        for im,mask,sid in rows:
            manifest_rows.append({'split':split_name,'id':sid,'image':rel(im,root),'mask':rel(mask,root),'image_sha256':sha256_file(im),'mask_sha256':sha256_file(mask)})
    manifest={'dataset':name,'root':str(root),'split_seed':SPLIT_SEED,'split_strategy':'single deterministic random 70/10/20 partition; same split reused for all losses and seeds','counts':counts,'total':len(pairs),'samples':manifest_rows}
    (out/'split_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f"{name}: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    print(f'Saved to: {out}')


def write_predefined_dataset(name, root, out, splits, counts, force):
    out.mkdir(parents=True,exist_ok=True)
    targets=[out/'train.csv',out/'val.csv',out/'test.csv',out/'split_manifest.json']
    existing=[p for p in targets if p.exists()]
    if existing and not force:
        raise FileExistsError(f'{name} split files already exist. Use --force only to recreate the same deterministic split.')
    print(f'\n{name}: validating pairs...')
    for split_name,rows in splits.items():
        validate_pairs(f'{name} {split_name}',rows)
    actual={k:len(v) for k,v in splits.items()}
    if actual!=counts:
        raise RuntimeError(f'{name} counts mismatch: expected {counts}, got {actual}')
    manifest_rows=[]
    for split_name,rows in splits.items():
        write_csv(out/f'{split_name}.csv',rows,root)
        for im,mask,sid in rows:
            manifest_rows.append({'split':split_name,'id':sid,'image':rel(im,root),'mask':rel(mask,root),'image_sha256':sha256_file(im),'mask_sha256':sha256_file(mask)})
    manifest={'dataset':name,'root':str(root),'split_seed':SPLIT_SEED,'split_strategy':'published train/test preserved; fixed 60-image validation subset drawn from the 300-image training pool with seed 20260705','counts':counts,'total':sum(counts.values()),'samples':manifest_rows}
    (out/'split_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f"{name}: train={actual['train']}, val={actual['val']}, test={actual['test']}")
    print(f'Saved to: {out}')

def main():
    print('prepare_paper3_extension_splits v2.1')
    ap=argparse.ArgumentParser(); ap.add_argument('--force',action='store_true'); args=ap.parse_args()
    if not CFD_ROOT.exists(): raise FileNotFoundError(CFD_ROOT)
    if not CRACKTREE_ROOT.exists(): raise FileNotFoundError(CRACKTREE_ROOT)
    if not DEEPCRACK_ROOT.exists(): raise FileNotFoundError(DEEPCRACK_ROOT)
    write_predefined_dataset('DeepCrack',DEEPCRACK_ROOT,DEEPCRACK_SPLIT_DIR,discover_deepcrack_pairs(),DEEPCRACK_COUNTS,args.force)
    write_dataset('CFD',CFD_ROOT,CFD_SPLIT_DIR,discover_cfd_pairs(),CFD_COUNTS,args.force)
    write_dataset('CrackTree260',CRACKTREE_ROOT,CRACKTREE_SPLIT_DIR,discover_cracktree_pairs(),CRACKTREE_COUNTS,args.force)
    print('\nAll nine CSV split files were created successfully.')

if __name__=='__main__':
    main()
