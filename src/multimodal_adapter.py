import torch
import numpy as np
from typing import Tuple, Dict, List
from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms
from diffusers import AutoencoderKL
from datasets import load_dataset
from src.config import Config
from src.text_adapter import TextAdapter

class MultimodalAdapter:
    """
    Real VAE Adapter. Supports two modes:
    1. Cached Mode: Downloads 4 sample images, pre-encodes them. (Fast, Low Diversity)
    2. Streaming Mode: Streams real COCO data from HuggingFace. (Slower, High Diversity)
    """
    def __init__(self, batch_size=4, image_size=128, seq_len=64, device='cpu', streaming=False):
        self.batch_size = batch_size
        self.image_size = image_size 
        self.seq_len = seq_len
        self.vocab_size = Config.vocab_size
        self.device = device
        self.streaming = streaming
        
        # 1. Load VAE
        print(f"[MultimodalAdapter] Loading VAE (stabilityai/sd-vae-ft-mse)...")
        try:
            self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
            self.vae.eval() 
            for param in self.vae.parameters():
                param.requires_grad = False
            print("[MultimodalAdapter] VAE Loaded Successfully.")
        except Exception as e:
            print(f"[MultimodalAdapter] !! Error loading VAE: {e}")
            self.vae = None

        # 2. Setup Transform
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ])
        
        # 3. Setup Data Source
        if self.streaming:
            print("[MultimodalAdapter] Mode: STREAMING (COCO)")
            # Using a reliable Ungated dataset (CIFAR-10) as robust fallback
            try:
                # cifar10 is native Parquet/Arrow, no scripts.
                self.dataset = load_dataset("cifar10", split="train", streaming=True)
                self.data_iter = iter(self.dataset)
                # Need text tokenizer for captions
                self.tokenizer = TextAdapter(split="test").tokenizer 
                self.cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            except Exception as e:
                print(f"[MultimodalAdapter] Failed to load Dataset streaming: {e}")
                print("Fallback to Cached Mode.")
                self.streaming = False
                
        if not self.streaming:
            print("[MultimodalAdapter] Mode: CACHED (4 Samples)")
            self.images = self._load_sample_images()
            self.cached_latents = self._pre_encode_latents(device)
            self.tokenizer = TextAdapter(split="test").tokenizer

    def _pre_encode_latents(self, device):
        if not self.images:
            return []
        print("[MultimodalAdapter] Pre-encoding latents...")
        latents_list = []
        with torch.no_grad():
            for img in self.images:
                img_batch = img.unsqueeze(0).to(device)
                l = self.vae.encode(img_batch).latent_dist.sample() * 0.18215
                latents_list.append(l.cpu())
        return latents_list

    def _load_sample_images(self) -> List[torch.Tensor]:
        urls = [
            "http://images.cocodataset.org/val2017/000000039769.jpg",
            "http://images.cocodataset.org/val2017/000000050000.jpg",
            "http://images.cocodataset.org/val2017/000000100000.jpg",
            "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
        ]
        tensors = []
        print(f"[MultimodalAdapter] downloading {len(urls)} sample images...")
        for url in urls:
            try:
                response = requests.get(url, timeout=5)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                tensor = self.transform(img)
                tensors.append(tensor)
            except Exception as e:
                pass # Silent fail for samples
        
        if len(tensors) == 0:
            print("[MultimodalAdapter] WARNING: Generating noise images.")
            tensors = [torch.randn(3, self.image_size, self.image_size) for _ in range(4)]
        return tensors

    def get_batch(self, device: torch.device) -> Dict[str, torch.Tensor]:
        
        if self.streaming:
            # Fetch real data
            images = []
            captions = []
            
            for _ in range(self.batch_size):
                try:
                    item = next(self.data_iter)
                    # Item typically has 'image' and 'caption'
                    # Check keys
                    # item has 'img' and 'label'
                    img = item.get('img')
                    label_idx = item.get('label', 0)
                    class_name = self.cifar_classes[label_idx]
                    txt = f"A low resolution photo of a {class_name}"
                    
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                        
                    images.append(self.transform(img))
                    captions.append(txt)
                except StopIteration:
                    # Reset iterator
                    self.dataset = load_dataset("cifar10", split="train", streaming=True)
                    self.data_iter = iter(self.dataset)
                    
            # Encode Images
            batch_imgs = torch.stack(images).to(device)
            with torch.no_grad():
                latents = self.vae.encode(batch_imgs).latent_dist.sample() * 0.18215
                
            # Tokenize Captions
            # We must output [B, Seq_Len] ids
            encoding = self.tokenizer(
                captions, 
                max_length=self.seq_len, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )
            input_ids = encoding.input_ids.to(device)
            
            return {
                "visual_latents": latents,
                "input_ids": input_ids
            }
            
        else:
            # Cached Mode (Synthetic Captions or Cached)
            if self.cached_latents:
                indices = np.random.choice(len(self.cached_latents), self.batch_size)
                batch_latents = torch.cat([self.cached_latents[i].to(device) for i in indices])
            else:
                 batch_latents = torch.randn(self.batch_size, 4, 16, 16, device=device)

            input_ids = torch.randint(
                0, 
                self.vocab_size, 
                (self.batch_size, self.seq_len), 
                device=device
            )
            
            return {
                "visual_latents": batch_latents, 
                "input_ids": input_ids
            }
