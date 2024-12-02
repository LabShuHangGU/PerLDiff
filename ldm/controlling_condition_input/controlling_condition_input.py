import os 
import torch as th 

class EncodingNetInput:
    def __init__(self, img_size):
        self.set = False 
        self.img_size = img_size

    def prepare(self, batch):
        """
        batch should be the output from dataset.
        Please define here how to process the batch and prepare the 
        input only for the ground tokenizer. 
        """

        self.set = True
        
        box = batch['box'] 
        box_mask = batch['box_mask']
        positive_embeddings = batch["box_text_embedding"] 
        self.in_dim = positive_embeddings.shape[-1]

        perl_box_masking_map = batch['perl_box_masking_map']
        perl_road_masking_map = batch['perl_road_masking_map']

        self.uroad_map_embedding = batch["uroad_map_embedding"]

        road_map_embedding = batch["road_map_embedding"]

        self.ucontext = batch['ucontext']

        context = batch['context']

        self.batch, self.num_camera, self.max_box, _ = box.shape
        self.device = positive_embeddings.device
        self.dtype = positive_embeddings.dtype

        return {"box":box, "box_mask":box_mask, "positive_embeddings":positive_embeddings, "perl_box_masking_map":perl_box_masking_map, "perl_road_masking_map":perl_road_masking_map, "road_map_embedding":road_map_embedding, "context":context}


    def get_null_input(self, batch=None, device=None, dtype=None):
        """
        Guidance for training (drop) or inference, 
        please define the null input for the grounding tokenizer 
        """

        assert self.set, "not set yet, cannot call this funcion"
        batch =  self.batch  if batch  is None else batch
        device = self.device if device is None else device
        dtype = self.dtype   if dtype  is None else dtype

        box = th.zeros(batch, self.num_camera, self.max_box, 16,).type(dtype).to(device) 
        box_mask = th.zeros(batch, self.num_camera, self.max_box).type(dtype).to(device) 
        positive_embeddings = th.zeros(batch, self.num_camera, self.max_box, self.in_dim).type(dtype).to(device) 
        
        perl_box_masking_map = th.zeros(batch, self.num_camera,self.max_box, self.img_size[0], self.img_size[1]).type(dtype).to(device) 
        perl_road_masking_map = th.zeros(batch, self.num_camera, self.img_size[0], self.img_size[1]).type(dtype).to(device) 

        return {"box":box, "box_mask":box_mask, "positive_embeddings":positive_embeddings, "perl_box_masking_map":perl_box_masking_map, "perl_road_masking_map":perl_road_masking_map, "road_map_embedding":self.uroad_map_embedding, "context":self.ucontext}

