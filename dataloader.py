import json
import parser
from torch.utils.data import Dataset
from PIL import Image
import cv2 
import torch
from transformers import LayoutLMv2Processor
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")



#classes
train_config=parser.train_config
class_path=train_config["class_path"]
with open(class_path,'r') as file:
    classes=file.read().split('\n')



#Label ENcoding
label_index={}
for index,label in enumerate(classes):
  label_index[label]=index

###########################Using textract OCR##################################


class Preprocess(Dataset):
    def __init__(self,df):
        # self.device=device
        self.df=df


    def get_text_elements(self,ocr_file_path, img_shape):
  
        final_words, final_bboxes = [], []


        with open(ocr_file_path) as file:
            data = json.load(file)

  
        if type(data) == list:
            data = data[0]



        for block in data["Blocks"]:
                    block["Geometry"]["BoundingBox"]["Left"] = int(
                        img_shape[0] * block["Geometry"]["BoundingBox"]["Left"]
                    )
                    block["Geometry"]["BoundingBox"]["Width"] = int(
                        img_shape[0] * block["Geometry"]["BoundingBox"]["Width"]
                    )
                    block["Geometry"]["BoundingBox"]["Top"] = int(
                        img_shape[1] * block["Geometry"]["BoundingBox"]["Top"]
                    )
                    block["Geometry"]["BoundingBox"]["Height"] = int(
                        img_shape[1] * block["Geometry"]["BoundingBox"]["Height"]
                    )


        for block in data['Blocks']:
            if block['BlockType']=='WORD':
        
                try:
                    top = block["Geometry"]["BoundingBox"]["Top"]
                    left = block["Geometry"]["BoundingBox"]["Left"]
                    bottom = block["Geometry"]["BoundingBox"]["Height"] + top
                    right = block["Geometry"]["BoundingBox"]["Width"] + left

                    x1 = left
                    y1 = top
                    x2 = right
                    y2 = bottom
                    box=[x1,y1,x2,y2]


                    final_words.append(block["Text"])

                    final_bboxes.append(box)

                except:
                    continue
        return final_words, final_bboxes


#############################
    def fix_invalid_boxes(self,unnormalized_word_boxes):
        for bbox in unnormalized_word_boxes:
            if bbox[0] > bbox[2]:
                bbox[0], bbox[2] = bbox[2], bbox[0]
            if bbox[1] > bbox[3]:
                bbox[1], bbox[3] = bbox[3], bbox[1]
##############################
    def normalize_box(self, box, width, height):
        return [
            int(1000 * (box[0] / width)),
            int(1000 * (box[1] / height)),
            int(1000 * (box[2] / width)),
            int(1000 * (box[3] / height)),
            ]

##############################
    def __len__(self):
        return len(self.df)
##############################
    def __getitem__(self,idx):
        ocr_path=self.df.ocrs[idx]
        image_path=self.df.images[idx]
        lab=self.df.labels[idx]
        label=label_index[lab]

        full_image = cv2.imread(image_path)
        height, width, _ = full_image.shape
        img_shape = (width, height)
        original_image = Image.fromarray(full_image).convert("RGB")

        (words,unnormalized_word_boxes,) = self.get_text_elements(ocr_path, img_shape)
        self.fix_invalid_boxes(unnormalized_word_boxes)

        width, height = original_image.size
        normalized_word_boxes = [self.normalize_box(bbox, width, height) for bbox in unnormalized_word_boxes]
        assert len(words) == len(normalized_word_boxes)
        # print(len(words))
        # print(len(normalized_word_boxes))

     


        image = Image.open(image_path).convert("RGB")
        # print(normalized_word_boxes)
        # print(len(normalized_word_boxes))
        # print("*************************************************")
        encoding = processor(image, words, boxes=normalized_word_boxes, return_tensors="pt",padding='max_length',truncation=True,max_length=512)
        # print(encoding1.keys())
        sequence_label = torch.tensor([label])
        encoding["labels"]=sequence_label
        return encoding
# dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'image'])
