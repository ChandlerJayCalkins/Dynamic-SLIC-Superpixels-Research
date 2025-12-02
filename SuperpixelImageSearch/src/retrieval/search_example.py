import cv2
from src.data_pipeline.coco_opencv_loader import COCODataset
from src.features.descriptor_builder import build_descriptor
from src.retrieval.index import ImageIndex

# ---- Load dataset ----
images = "data/coco2017/images/train2017"
ann = "data/coco2017/annotations/instances_train2017.json"
ds = COCODataset(images, ann)

# ---- Build index ----
index = ImageIndex()
for i in range(50):   # build from first 50 images (for demo)
    img_bgr, info = ds.load_image_bgr(i)
    vec = build_descriptor(img_bgr)
    index.add(info["file_name"], vec)
    print(f"Indexed {info['file_name']}")

# ---- Query with another image ----
query_bgr, qinfo = ds.load_image_bgr(100)
q_vec = build_descriptor(query_bgr)

results = index.search(q_vec, top_k=5)
print("\nTop matches:")
for r in results:
    print(r)
