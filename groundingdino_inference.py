import argparse
import os
import sys

os.environ['TRANSFORMERS_CACHE'] = 'cache/'


from tqdm import tqdm
# from utilstools.pickle_handler import saveObject, loadObject
# from utilstools.json_handler import read_json, write_json

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from torchvision.ops import batched_nms
# print(sys.path)
sys.path.append(os.path.join(os.getcwd()))
# print(sys.path)

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from evaluate_map_all import evaluate_all, evaluate_all_multi_batch

import time,datetime
import json
import pickle

import concurrent.futures
import threading
import logging

import multiprocessing
from multiprocessing import Queue
# 获取当前时间
current_time = datetime.datetime.now()
print(current_time)

# 配置日志输出格式和级别
log_dir = "/picassox/sfs-mtlab-train-base/segmentation/myq1/experiment/Open-GroundingDino/train_in_my_dataset/test_on_new_split/terminal"  # 自定义日志存放目录
os.makedirs(log_dir, exist_ok=True)  # 如果目录不存在，创建它
log_file = os.path.join(log_dir, f"debug_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

# 配置日志输出
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG,
    handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]
)

# 获取 logger 实例
logger = logging.getLogger(__name__)


def saveObject(obj, path):
    """"Save an object using the pickle library on a file
    
    :param obj: undefined. Object to save
    :param fileName: str. Name of the file of the object to save
    """
    print("Saving " + path + '.pkl')
    with open(path + ".pkl", 'wb') as fid:
        pickle.dump(obj, fid)
        
def loadObject(path):
    """"Load an object from a file
    
    :param fileName: str. Name of the file of the object to load
    :return: obj: undefined. Object loaded
    """
    try:
        with open(path + '.pkl', 'rb') as fid:
            obj = pickle.load(fid)
            return obj
    except IOError:
        return None   
def read_json(file_name):
    #Read JSON file
    with open(file_name) as infile:
        data = json.load(infile)
    return data

def write_json(data, file_name):
    # Write JSON file
    with open(file_name, "w") as outfile:
        json.dump(data, outfile)
def get_category_name(id, category_id_map):
    return category_id_map[id]
        
def get_image_filepath(id, image_id_map):
    return image_id_map[id]

def adjust_out_id(output, vocabulary_id):
    for i in range(len(output['labels'])):
        output['labels'][i] = vocabulary_id[output['labels'][i]]
    return output

def calculate_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Parameters:
        bb1 (list): The first bounding box in the format [x_min, y_min, h, w].
        bb2 (list): The second bounding box in the format [x_min, y_min, h, w].

    Returns:
        float: The IoU value.
    """
    x1, y1, h1, w1 = bb1
    x2, y2, h2, w2 = bb2

    # Calculate the coordinates of the intersection rectangle
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    x_intersection_end = min(x1 + w1, x2 + w2)
    y_intersection_end = min(y1 + h1, y2 + h2)

    # If there's no intersection, return 0
    if x_intersection >= x_intersection_end or y_intersection >= y_intersection_end:
        return 0.0

    # Calculate the areas of both bounding boxes and the intersection
    area_bb1 = h1 * w1
    area_bb2 = h2 * w2
    area_intersection = (y_intersection_end - y_intersection) * (x_intersection_end - x_intersection)

    # Calculate the IoU
    iou = area_intersection / float(area_bb1 + area_bb2 - area_intersection)
    return iou


def create_vocabulary(ann, category_id_map):
    vocabulary_id = [ann['category_id']] + ann['neg_category_ids']
    vocabulary_uncleaned = [get_category_name(id, category_id_map) for id in vocabulary_id]
    
    vocabulary = []
    for voc in vocabulary_uncleaned:
        # voc = voc[:-1] if voc[-1] == '.' else voc # removing last '.'
        # voc = voc.replace('.', ',') # replace eventual ','
        if not voc.endswith("."):
            voc = voc + "."
        vocabulary.append(voc)
    
    return vocabulary, vocabulary_id

def sort_boxes_by_score(boxes, labels, scores, max_elem):
    """
    Sorts the boxes, labels, and scores arrays by the score in descending order.

    Args:
        boxes (list): List of boxes.
        labels (list): List of labels.
        scores (list): List of scores.

    Returns:
        dict: A dictionary containing the sorted boxes, labels, and scores arrays.
    """
    # Sort the boxes, labels, and scores arrays by score in descending order
    sorted_indices = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
    sorted_boxes = [boxes[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]
    # sorted_total_scores = [total_scores[i] for i in sorted_indices]

    # Create a dictionary to store the sorted arrays
    sorted_dict = {
        'boxes': sorted_boxes[:max_elem],
        'labels': sorted_labels[:max_elem],
        'scores': sorted_scores[:max_elem],
        # 'total_scores': sorted_total_scores
    }

    return sorted_dict



def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def apply_NMS(boxes, scores, labels, iou=0.5):
    indexes_to_keep = batched_nms(torch.stack(boxes, dim=0),
                       torch.FloatTensor(scores),
                       torch.IntTensor([0] * len(boxes)),
                       iou)
    
    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    deleted_boxes = []
    deleted_scores = []
    deleted_labels = []
    
    for x in range(len(boxes)):
        if x in indexes_to_keep:
            filtered_boxes.append(boxes[x])
            filtered_scores.append(scores[x])
            filtered_labels.append(labels[x])
        else:
            deleted_boxes.append(boxes[x])
            deleted_scores.append(scores[x])
            deleted_labels.append(labels[x])
    
    
    # bisognerebbe fare in modo che le box eliminate abbiano gli score appesi a quelle rimaste
    # return filtered_boxes, filtered_scores, filtered_labels, create_total_scores(filtered_boxes, filtered_scores, filtered_labels, deleted_boxes, deleted_scores, deleted_labels)

    return filtered_boxes, filtered_scores, filtered_labels


def create_total_scores(pred_boxes_filtered, pred_scores_filtered, pred_label_ids_filtered, pred_boxes, pred_scores, pred_label_ids):
    pred_total_scores = []
    count = 0
    for pred_box_filtered, pred_score_filtered, pred_label_filtered in tqdm(zip(pred_boxes_filtered, pred_scores_filtered, pred_label_ids_filtered)):
        # for each bbox, we find for each label which is the score of the deleted box with highest overlapping
        max_iou = [0] * 11
        max_score = [0] * 11
        for pred_box, pred_score, pred_label in zip(pred_boxes, pred_scores, pred_label_ids):
            if pred_label == pred_label_filtered:
                continue
            iou = calculate_iou(pred_box_filtered, pred_box)
            if max_iou[pred_label] < iou:
                max_iou[pred_label] = iou
                max_score[pred_label] = pred_score
        max_score[pred_label_filtered] = pred_score_filtered # assigning the value of the score of the prediction
        
        # in theory the highest score should be achieved by the filtered box, but we check anyway
        if max(max_score) > pred_score_filtered:
            count += 1
        pred_total_scores.append(max_score)
    
    print("%d/%d" % (count, len(pred_box_filtered)))
    return pred_total_scores

def get_grounding_output(model, image, captions, w, h, max_elem=100, iou=0.5, box_threshold=0, cpu_only=False, device = "cuda"):
    # text_threshold = 0
    # device = "cuda" if not cpu_only else "cpu"
    image = image.to(device)
    print("device in get_grounding_output function: ", device)

    
    # we need to batch the inputs, in order to make a query for each caption
    images = image[None].repeat(len(captions), 1, 1, 1)
    
    with torch.no_grad():
        outputs = model(images, captions=captions)
    
    pred_boxes = []
    pred_scores = []
    pred_label_ids = []
    
    logits_batched = [x.cpu().sigmoid() for x in outputs['pred_logits']]
    boxes_batched = [x.cpu() for x in outputs['pred_boxes']]
    for logits, boxes, label_id in zip(logits_batched, boxes_batched, range(len(captions))):
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]
        
        for logit, box in zip(logits_filt, boxes_filt):
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([w, h, w, h])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            pred_boxes.append(box)
            pred_label_ids.append(label_id)
            pred_scores.append(logit.max().item())

    # pred_boxes, pred_scores, pred_label_ids = apply_NMS(pred_boxes, pred_scores, pred_label_ids, iou)
    return sort_boxes_by_score(pred_boxes, pred_label_ids, pred_scores, max_elem)

categories_done_lock = threading.Lock()

def process_annotation(ann, model, processed_count, lock, category_id_map, image_id_map, args, queue, categories_done, device):
    try:
        
        logger.debug(f"Processing annotation: {ann['image_id']} - {ann['category_id']}")
        # 使用字典来模拟集合的功能
        if (ann['category_id'], ann["image_id"]) not in categories_done:
            categories_done[(ann['category_id'], ann["image_id"])] = None  # 在字典中设置键
            # logger.debug(f"Added to categories_done: {ann['image_id']} - {ann['category_id']}")
        else:
            # logger.debug(f"Skipping already processed pair: {ann['image_id']} - {ann['category_id']}")
            return  # Skip already processed pair

        # Check if a number of hardnegatives is set to non-default values
        vocabulary, vocabulary_id = create_vocabulary(ann, category_id_map)
        # logger.debug(f"Vocabulary created: {vocabulary}")

        vocabulary = vocabulary[:args.n_hardnegatives + 1]
        vocabulary_id = vocabulary_id[:args.n_hardnegatives + 1]
        
        image_filepath = os.path.join(args.imgs_path, get_image_filepath(ann['image_id'], image_id_map))
        # logger.debug(f"Loading image: {image_filepath}")
        
        # Load image and process it
        image_pil, imm = load_image(image_filepath)
        # logger.debug(f"Image loaded: {image_pil.size[0]}x{image_pil.size[1]}")
        
        output = get_grounding_output(model, imm, vocabulary, image_pil.size[0], image_pil.size[1], device=device)
        if output is not None:
            output['category_id'] = ann['category_id']
            output['vocabulary'] = vocabulary_id
            output['image_filepath'] = get_image_filepath(ann['image_id'], image_id_map)
            output = adjust_out_id(output, vocabulary_id)
            def convert_tensor(t):
                if isinstance(t, torch.Tensor):
                    return t.cpu().numpy()
                elif isinstance(t, dict):
                    return {k: convert_tensor(v) for k, v in t.items()}
                elif isinstance(t, list):
                    return [convert_tensor(x) for x in t]
                return t

            # 转换整个输出
            output = convert_tensor(output)
            queue.put(output)
            # 更新进度条
            # logger.debug(f"Grounding output received for {image_filepath}")

            # progress_bar.update(1)
        else:
            # queue.put(None)  # 显式存入 None，方便调试
            # logger.warning(f"process_annotation: output is None for annotation {ann['image_id']}")
            logger.warning(f"Output is None for {image_filepath} and {vocabulary} of {ann['image_id']}")
        # 更新处理数量
        with lock:  # 加锁避免多进程竞争
            processed_count.value += 1
            print(f"Processed {processed_count.value} annotations")

    except Exception as e:
        logger.error(f"Error processing annotation {ann['id']}: {e}", exc_info=True)
        raise

def process_dataset(dataname, genfolder, args):
    logger.info(f"--------------------- Start processing dataset: {dataname} ---------------------\n")
    start_timestamp = int(time.time())
    logger.info(f"Start time: {datetime.datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d %H:%M:%S')}")

    dataset_path = os.path.join(genfolder, dataname)
    data = read_json(dataset_path)
    logger.debug(f"Dataset {dataname} loaded with {len(data['annotations'])} annotations")

    for category in data['categories']:
        category_id_map[category['id']] = category['caption']
    for image in data['images']:
        image_id_map[image['id']] = image['filename']

    num_gpus = torch.cuda.device_count()
    logger.info(f"Number of available GPUs: {num_gpus}")
    num_processes_per_gpu = 2
    total_processes = num_gpus * num_processes_per_gpu

    with multiprocessing.Manager() as manager:
        processed_count = manager.Value('i', 0)
        categories_done = manager.dict()
        queue = manager.Queue()
        lock = manager.Lock()
        
        annotations = data['annotations']
        chunk_size = len(annotations) // total_processes
        annotation_chunks = [annotations[i * chunk_size: (i + 1) * chunk_size] for i in range(total_processes)]
        
        processes = []
        for gpu_id in range(num_gpus):
            for i in range(num_processes_per_gpu):
                process_index = gpu_id * num_processes_per_gpu + i
                if process_index < len(annotation_chunks):
                    p = multiprocessing.Process(target=process_worker, args=(
                        annotation_chunks[process_index], args, gpu_id, processed_count, lock, 
                        category_id_map, image_id_map, args, queue, categories_done
                    ))
                    processes.append(p)
                    p.start()
        
        for p in processes:
            p.join()
        
        complete_outputs = []
        while not queue.empty():
            item = queue.get()
            if item is not None:
                complete_outputs.append(item)
            # complete_outputs.append(queue.get())
    
    end_timestamp = int(time.time())
    logger.info(f"End time: {datetime.datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total time taken: {(end_timestamp - start_timestamp) // 60} minutes")

    dataname_woext, ext = os.path.splitext(dataname)
    out_path = os.path.join(args.map_out, f"benchmark_{dataname_woext}_neg{str(args.n_hardnegatives)}")
    saveObject(list(complete_outputs), out_path)
    logger.info(f"category_len: {len(data['categories'])}\n image_len: {len(data['images'])}\n annotation_len: {len(data['annotations'])} \n complete_outputs_len: {len(complete_outputs)}")

    logger.info(f"Results saved to {out_path}")
    
    map_out_path = os.path.join(args.map_out, os.path.basename(out_path) + ".txt")
    evaluate_all_multi_batch(pred_path=out_path, dataset_path=dataset_path, out=map_out_path)
    logger.info(f"Evaluation completed and results saved to {map_out_path}")

def process_worker(annotations, model_args, gpu_id, processed_count, lock, category_id_map, image_id_map, args, queue, categories_done):
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    torch.cuda.set_device(gpu_id)  # 显式设置 GPU 设备
    device = torch.device(f'cuda:{gpu_id}')
    print(f"Process worker on GPU {gpu_id}")
    print(f"Device: {device}")
    # print(f"Available GPUs: {torch.cuda.device_count()}")
    # print(f"Current CUDA device: {torch.cuda.current_device()}")
    # print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


    # load model
    config_file = args.config_file
    checkpoint_path = args.checkpoint_path

    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)
    # device = "cuda" if not args.cpu_only else "cpu"
    model = model.to(device)

    # 重新实例化模型
    # model = model_class(*model_args).to(device)
    model.eval()

    for ann in annotations:
        result = process_annotation(ann, model, processed_count, lock, category_id_map, image_id_map, args, queue, categories_done, device)
        queue.put(result)

def set_start_method():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        print("RuntimeError: start method already set")
        pass  # 已经设置过了，不做重复设置




# 主逻辑
def main():
    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", default="", type=str, help="path to config file")
    parser.add_argument("--checkpoint_path", "-p", type=str, default="", help="path to checkpoint file")
    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    # parser.add_argument('--dataset_path', "--dataset", type=str,  default="/picassox/obs-mtlab-train-base/segmentation/myq1/process_pixcap_dataset/step9_final_dataset_with_negatives/before_artifical_filter/test_dataset/no_text_image_info_have_bbox_we_need_with_boxes_relationship_and_attribute_negatives_coco_format.json",help='Path of the json to process')
    parser.add_argument("--imgs_path", default="pixmo_cap_image", type=str, help="Path of the images to process")
    # parser.add_argument('--out', type=str, required=True, help='Output file')
    parser.add_argument('--n_hardnegatives', type=int, default=5, help="Number of hardnegatives in each vocabulary")
    parser.add_argument('--map_out',default="", help="path to output folder")
    parser.add_argument('--genfolder',default="", help="path to annotation folder")

    args = parser.parse_args()

    genfolder = args.genfolder

    dataset_names = [ "attribute_shape.json", "attribute_material.json","relationship_spatial.json", "relationship_action.json", "text.json", "attribute_color.json", "attribute_other.json" ]
    
    set_start_method()  # 确保设置了 spawn 启动方法
    print("=================================")
    print("checkpoint_path: ", args.checkpoint_path)
    print("config_file: ", args.config_file)
    # print("imgs_path: ", args.imgs_path)
    print("map_out: ", args.map_out)
    print("test_folder: ", genfolder)
    print("=================================")

    args.n_hardnegatives = 5

    # 对每个dataset并行处理
    for dataname in dataset_names:

        process_dataset(dataname, genfolder,  args)

        # 清理 GPU 资源
        torch.cuda.empty_cache()


if __name__ == "__main__":

    main()
