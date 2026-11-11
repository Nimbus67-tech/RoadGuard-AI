"""Audit image datasets with YOLO-format detection labels."""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from collections import Counter
from pathlib import Path
from typing import Iterable
from PIL import Image

IMAGE_EXTS={".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def parse_yolo_label(path: Path, class_names: list[str]):
    rows,errors=[],[]
    try: lines=path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError: return rows,["label_not_utf8"]
    for line_no,raw in enumerate(lines,1):
        line=raw.strip()
        if not line: continue
        parts=line.split()
        if len(parts)<5:
            errors.append(f"line_{line_no}:expected_at_least_5_fields"); continue
        try:
            cf=float(parts[0])
            if not cf.is_integer(): raise ValueError
            cid=int(cf); xc,yc,w,h=map(float,parts[1:5])
        except ValueError:
            errors.append(f"line_{line_no}:non_numeric_or_invalid_class"); continue
        if not all(math.isfinite(v) for v in (xc,yc,w,h)):
            errors.append(f"line_{line_no}:non_finite_coordinate"); continue
        if not 0<=cid<len(class_names): errors.append(f"line_{line_no}:class_id_out_of_range:{cid}")
        if w<=0 or h<=0: errors.append(f"line_{line_no}:non_positive_box_size")
        if any(v<0 or v>1 for v in (xc,yc,w,h)): errors.append(f"line_{line_no}:coordinate_outside_0_1")
        if w>0 and h>0 and (xc-w/2<0 or yc-h/2<0 or xc+w/2>1 or yc+h/2>1):
            errors.append(f"line_{line_no}:box_exceeds_image_bounds")
        rows.append(dict(class_id=cid,class_name=class_names[cid] if 0<=cid<len(class_names) else "__INVALID__",x_center=xc,y_center=yc,width=w,height=h,area_fraction=w*h,line_number=line_no))
    return rows,errors

def _write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def audit_dataset(root: Path,image_dirs: Iterable[str],label_dirs: Iterable[str],class_names: list[str],out_dir: Path)->dict:
    root=Path(root).resolve(); out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    image_dirs,label_dirs=list(image_dirs),list(label_dirs)
    if len(image_dirs)!=len(label_dirs): raise ValueError("Image and label directory lists must have matching lengths.")
    rows_img,rows_ann=[],[]; counts,issues=Counter(),Counter(); hashes={}
    for idir,ldir in zip(image_dirs,label_dirs):
        ir,lr=root/idir,root/ldir
        if not ir.is_dir(): raise FileNotFoundError(f"Image directory not found: {ir}")
        if not lr.is_dir(): raise FileNotFoundError(f"Label directory not found: {lr}")
        for img in sorted(p for p in ir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            rel=img.relative_to(ir); label=(lr/rel).with_suffix(".txt")
            if not label.exists(): label=lr/(img.stem+".txt")
            try:
                with Image.open(img) as im: width,height=im.size; im.verify()
                image_error=""
            except Exception as exc:
                width=height=None; image_error=type(exc).__name__; issues["unreadable_images"]+=1
            parsed,errs=([],[]) if not label.exists() else parse_yolo_label(label,class_names)
            status="missing" if not label.exists() else ("empty_or_unparsed" if errs and not parsed else ("empty_valid" if not parsed else "present"))
            if not label.exists(): issues["missing_label_files"]+=1
            for e in errs: issues[e.split(":",1)[-1].split(":",1)[0]]+=1
            for a in parsed:
                counts[a["class_name"]]+=1
                rows_ann.append({"image":str(img.relative_to(root)),"label_file":str(label.relative_to(root)),**a})
            digest=sha256_file(img); hashes.setdefault(digest,[]).append(str(img.relative_to(root)))
            rows_img.append(dict(image=str(img.relative_to(root)),width_px=width,height_px=height,file_bytes=img.stat().st_size,sha256=digest,label_file=str(label.relative_to(root)) if label.exists() else "",label_status=status,annotation_count=len(parsed),image_error=image_error,label_errors=";".join(errs)))
    duplicates=[v for v in hashes.values() if len(v)>1]
    summary=dict(image_count=len(rows_img),annotation_count=len(rows_ann),class_names=class_names,class_counts=dict(counts),images_missing_labels=sum(x["label_status"]=="missing" for x in rows_img),images_empty_valid_labels=sum(x["label_status"]=="empty_valid" for x in rows_img),images_empty_or_unparsed_labels=sum(x["label_status"]=="empty_or_unparsed" for x in rows_img),unreadable_images=issues["unreadable_images"],exact_duplicate_groups=duplicates,exact_duplicate_group_count=len(duplicates),validation_issue_counts=dict(issues),warning="Exact SHA-256 detects byte-identical files only; this is not a semantic annotation audit.")
    _write_csv(out_dir/"images.csv",rows_img,["image","width_px","height_px","file_bytes","sha256","label_file","label_status","annotation_count","image_error","label_errors"])
    _write_csv(out_dir/"annotations.csv",rows_ann,["image","label_file","class_id","class_name","x_center","y_center","width","height","area_fraction","line_number"])
    _write_csv(out_dir/"class_counts.csv",[dict(class_name=n,box_count=counts[n]) for n in class_names],["class_name","box_count"])
    (out_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    report=["# Dataset audit report","",f"- Images: **{summary['image_count']}**",f"- Parsed annotations: **{summary['annotation_count']}**",f"- Missing labels: **{summary['images_missing_labels']}**",f"- Unparsed/invalid empty labels: **{summary['images_empty_or_unparsed_labels']}**",f"- Unreadable images: **{summary['unreadable_images']}**",f"- Exact duplicate groups: **{summary['exact_duplicate_group_count']}**","","## Class counts","","| Class | Boxes |","|---|---:|"]
    report += [f"| {n} | {counts[n]} |" for n in class_names]
    report += ["","## Validation issues",""]+[f"- {k}: {v}" for k,v in sorted(issues.items())]
    (out_dir/"audit_report.md").write_text("\n".join(report),encoding="utf-8")
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True,type=Path); p.add_argument("--images",nargs="+",required=True); p.add_argument("--labels",nargs="+",required=True); p.add_argument("--names",nargs="+",required=True); p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    if len(a.images)!=len(a.labels): p.error("--images and --labels must have the same number of directories")
    print(json.dumps(audit_dataset(a.root,a.images,a.labels,a.names,a.out),indent=2))
if __name__=="__main__": main()
