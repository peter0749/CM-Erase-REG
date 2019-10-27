import sys
import tqdm
import imagesize
import json
import pickle
import nltk
import argparse

def parse_args():
    parser = argparse.ArgumentParser('PFN-PIC to RefItGame converter')
    parser.add_argument('--json_path', type=str, required=True, help='path to annotation file')
    parser.add_argument('--image_dir_path', type=str, required=True, help='root to images')
    parser.add_argument('--ref_out', type=str, default='./ref.json', help='path to output referring annotation')
    parser.add_argument('--coco_out', type=str, default='./coco_fmt.json', help='path to output bbox annotation')
    parser.add_argument('--split', type=str, required=True, help='train/val')
    return parser.parse_args()

def main(args):
    '''
    input (each line):
        - image_file: '...',
        - pcd_file: '...',
        - objects: [{
            dest_box: '...',
            bbox: {
                x: int(...),
                y: int(...),
                width: int(...),
                height: int(...)
            }
        - instructions: [
            'expr_1',
            'expr_2',...
        ]
        },...]
    output:
        - ref_obj: [{
            - sent_ids: [...]
            - file_name: '...'
            - ann_id: int(...) -> coco
            - ref_id: int(...)
            - image_id: int(...) -> coco
            - split: 'train/val'
            - sentences:
                - [{tokens, raw, sent_id, sent}, ...]
            - category_id: int(...) -> coco
          },...]
        - coco_json: {
            - images: [{
                file_name: '...',
                height: int(...),
                width: int(...),
                id: int(...)
            },...]
            - type: "instances",
            - annotations:[{
                - area: int(...),
                - iscrowd: 0,
                - image_id: int(...),
                - bbox: int list [x,y,w,h],
                - category_id: 0,
                - id: int(...),
                - ignore: 0,
                - segmentation: [] <- FIXME: No segmentation annotation
            },...]
        }
    '''
    images = []
    annotations = []
    img_id = 0
    sent_id = 0
    annot_id = 0
    ref_obj = []
    with open(args.json_path, 'r') as fp:
        for line in tqdm.tqdm(fp):
            row_data = json.loads(line)
            file_name = row_data['image_file']
            real_path = args.image_dir_path+'/'+file_name
            width, height = imagesize.get(real_path)
            images.append({'file_name': file_name,
                           'height': height,
                           'width': width,
                           'id': img_id
                })
            sent_ids = []
            sentences = []
            for obj in row_data['objects']:
                bbox = obj['bbox']
                x, y, w, h = int(bbox['x']), int(bbox['y']), int(bbox['width']), int(bbox['height'])
                area = int(w*h)
                sents = obj['instructions']
                annotations.append({
                    'area': area,
                    'iscrowd': 0,
                    'image_id': img_id,
                    'bbox': [x,y,w,h],
                    'category_id': 0,
                    'id': annot_id,
                    'ignore': 0,
                    'segmentation': []
                    })
                for sent in sents:
                    p = sent.lower()
                    tokens = nltk.word_tokenize(p)
                    sentences.append({
                        'tokens': tokens,
                        'raw': sent,
                        'sent_id': sent_id,
                        'sent': p
                        })
                    sent_ids.append(sent_id)
                    sent_id += 1
                ref_obj.append({
                    'sent_ids': sent_ids,
                    'file_name': file_name,
                    'ann_id': annot_id,
                    'ref_id': annot_id,
                    'image_id': img_id,
                    'split': args.split,
                    'sentences': sentences,
                    'category_id': 0
                    })
                annot_id += 1
            img_id += 1
    with open(args.coco_out, 'w') as fp:
        fp.write(json.dumps({'images': images, 'type': 'instances', 'annotations': annotations, 'categories': [{'supercategory': 'none', 'id': 0, 'name': 'foo'},]}))
    with open(args.ref_out, 'wb') as fp:
        pickle.dump(ref_obj, fp, protocol=0)

if __name__=='__main__':
    main(parse_args())

