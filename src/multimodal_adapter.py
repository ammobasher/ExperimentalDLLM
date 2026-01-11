import numpy as np
from datasets import load_dataset
from PIL import Image
import io
from src.text_adapter import TextAdapter

class MultimodalAdapter:
    """
    Handles streaming Image-Text pairs.
    Fallbacks: 'cifar10' (Public, Synthetic Captions)
    """
    def __init__(self, dataset_name="cifar10", split="train", batch_size=4, image_size=256):
        self.batch_size = batch_size
        self.image_size = image_size
        self.dataset_name = dataset_name
        
        print(f"[MultimodalAdapter] Loading {dataset_name}...")
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)
        self.iterator = iter(self.dataset)
        
        # CIFAR Labels
        self.labels = [
            "airplane", "automobile", "bird", "cat", "deer", 
            "dog", "frog", "horse", "ship", "truck"
        ]
        
        # We reuse TextAdapter's tokenizer logic
        self.text_adapter = TextAdapter(seq_len=64, batch_size=batch_size, dataset_name="wikitext") 
        self.tokenizer = self.text_adapter.tokenizer

    def process_image(self, image_obj):
        """
        Convert PIL Image / Bytes to [H, W, 3] Normalized Float Array
        """
        if not isinstance(image_obj, Image.Image):
            # Try converting if bytes
            try:
                image_obj = Image.open(io.BytesIO(image_obj)).convert("RGB")
            except:
                # Dummy black image on failure
                return np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
                
        img = image_obj.resize((self.image_size, self.image_size))
        img_arr = np.array(img).astype(np.float32) / 255.0 # [0, 1]
        
        # Ensure 3 channels
        if img_arr.ndim == 2:
            img_arr = np.stack([img_arr]*3, axis=-1)
        elif img_arr.shape[-1] == 4:
            img_arr = img_arr[..., :3]
            
        return img_arr

    def get_batch(self):
        """
        Returns:
            images: [Batch, H, W, 3]
            tokens: [Batch, Seq]
        """
        images = []
        texts = []
        
        while len(images) < self.batch_size:
            try:
                item = next(self.iterator)
                # Item keys vary.
                # CIFAR10: 'img', 'label'
                
                if 'img' in item:
                    img_raw = item['img']
                elif 'image' in item:
                    img_raw = item['image']
                else:
                    raise ValueError("Unknown image key")

                # Image
                img = self.process_image(img_raw)
                
                # Text
                if 'text' in item:
                    txt = item['text']
                elif 'label' in item:
                    # Synthetic
                    label_idx = item['label']
                    label_name = self.labels[label_idx] if label_idx < len(self.labels) else "object"
                    txt = f"A photo of a {label_name}."
                else:
                    txt = "A photo of something."
                
                images.append(img)
                texts.append(txt)
                
            except StopIteration:
                self.iterator = iter(self.dataset)
                
        # Tokenize Text
        encodings = self.tokenizer(
            texts, 
            truncation=True, 
            padding="max_length", 
            max_length=64, 
            return_tensors="np"
        )
        
        batch_images = np.array(np.stack(images))
        batch_tokens = np.array(encodings['input_ids'])
        
        return batch_images, batch_tokens
