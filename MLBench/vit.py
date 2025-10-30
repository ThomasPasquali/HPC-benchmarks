from transformers import ViTImageProcessor, ViTModel, AutoImageProcessor, ViTForImageClassification
import torchvision
import argparse
import torch
import os

def load_model(model_name:str, num_labels:int):
    '''Load the ViT model and image processor from Hugging Face Transformers library'''

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_labels)

    return image_processor, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLBench")
    parser.add_argument("--model", type=str, default="google/vit-base-patch16-224-in21k",
                        help="Name of the ViT model to use")
    
    args = parser.parse_args()

    model_name = args.model
    image_processor, model = load_model(model_name, num_labels=100) # 101 depends on how many classes there are

    rand_gen = torch.Generator()
    image=torch.rand(3, 3, 224, 224)

    input_sample = image_processor(images=image, return_tensors="pt", do_rescale=False)

    model.eval()
    with torch.no_grad():
        out = model(**input_sample).logits

        print(out)