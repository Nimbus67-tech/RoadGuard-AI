from PIL import Image
from roadguard.dataset_audit import audit_dataset, parse_yolo_label

def test_parse_yolo_valid_and_invalid(tmp_path):
    p=tmp_path/"labels.txt"
    p.write_text("0 0.5 0.5 0.2 0.4\n3 0.5 0.5 0.2 0.2\n0 1.2 0.5 0.2 0.2\n")
    rows,errors=parse_yolo_label(p,["bike","no_helmet","helmet"])
    assert len(rows)==3
    assert any("class_id_out_of_range" in e for e in errors)
    assert any("coordinate_outside_0_1" in e for e in errors)

def test_audit_creates_reports(tmp_path):
    (tmp_path/"images/train").mkdir(parents=True); (tmp_path/"labels/train").mkdir(parents=True)
    Image.new("RGB",(100,80),"white").save(tmp_path/"images/train/a.jpg")
    (tmp_path/"labels/train/a.txt").write_text("0 0.5 0.5 0.2 0.3\n")
    result=audit_dataset(tmp_path,["images/train"],["labels/train"],["motorcycle"],tmp_path/"reports")
    assert result["image_count"]==1 and result["annotation_count"]==1
    assert (tmp_path/"reports/summary.json").exists()

def test_empty_label_is_valid_negative(tmp_path):
    (tmp_path/"images/train").mkdir(parents=True); (tmp_path/"labels/train").mkdir(parents=True)
    Image.new("RGB",(64,64),"black").save(tmp_path/"images/train/negative.jpg")
    (tmp_path/"labels/train/negative.txt").write_text("")
    result=audit_dataset(tmp_path,["images/train"],["labels/train"],["bike"],tmp_path/"reports")
    assert result["images_empty_valid_labels"]==1
    assert result["images_empty_or_unparsed_labels"]==0

def test_malformed_label_not_valid_negative(tmp_path):
    (tmp_path/"images/train").mkdir(parents=True); (tmp_path/"labels/train").mkdir(parents=True)
    Image.new("RGB",(64,64),"black").save(tmp_path/"images/train/bad.jpg")
    (tmp_path/"labels/train/bad.txt").write_text("not a yolo annotation\n")
    result=audit_dataset(tmp_path,["images/train"],["labels/train"],["bike"],tmp_path/"reports")
    assert result["images_empty_valid_labels"]==0
    assert result["images_empty_or_unparsed_labels"]==1
