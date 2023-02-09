import os
import json
import cv2

def region2coco(region_json_path):
    with open(region_json_path,"r") as f:
        old_ann = json.load(f)
        coco_ann = {}
        images = []
        annotations = []
        categories = [{
            "id":1,
            "name":"balloon",
            "supercategory":""
        }]
        prefix = os.path.split(region_json_path)[0]
        ann_id = 1
        for id,anno in enumerate(old_ann.values()):
            image_id = id+1
            filename = anno["filename"]
            img = cv2.imread(os.path.join(prefix,filename))
            height,width = img.shape[:2]
            images.append({
                "id":image_id,
                "width":width,
                "height":height,
                "file_name":filename
            })
            for polygon in anno["regions"].values():
                all_points_x = polygon["shape_attributes"]["all_points_x"]
                all_points_y = polygon["shape_attributes"]["all_points_y"]
                seg = []
                for x,y in zip(all_points_x,all_points_y):
                    seg.append(x)
                    seg.append(y)
                minx = min(all_points_x)
                miny = min(all_points_y)
                box_width = max(all_points_x) - minx
                box_height = max(all_points_y) - miny
                annotations.append({
                    "id":ann_id,
                    "image_id":image_id,
                    "category_id":1,
                    "segmentation":[seg],
                    "area":box_height*box_width,
                    "bbox":[minx,miny,box_width,box_height],
                    "iscrowd":0
                })
                ann_id += 1
        coco_ann = {
            "images":images,
            "annotations":annotations,
            "categories":categories
        }
        with open(os.path.join(prefix,"coco.json"),"w") as f2:         
            f2.write(json.dumps(coco_ann))

region_json_path = "./balloon/train/via_region_data.json"
# region_json_path = "./balloon/val/via_region_data.json"
region2coco(region_json_path=region_json_path)

