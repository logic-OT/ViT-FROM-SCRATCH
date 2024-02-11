# Simple Vision Transformer Implementation With Pytorch


This  Repo implements a simple <b>Vision Transformer(ViT)</b> for a dummy classification task of predicting whether a person is wearing a <b> Hat or not</b>.
## Key Components


1.   <b> Image patcher and depatcher </b>
2.   <b> Positional Encoding and Transformer Encoder (My implementation is from "Attention is All You Need")</b>
3.   <b> Model Architecture</b>
4.   <b> Attention Visualisation</b>


![Architecture design](https://github.com/logic-OT/TRANSFORMER-FROM-SCRATCH/assets/61668807/cf05acb6-e928-4e99-a7a0-623fbdd8c2f0)



# Repo Content
This Repo contains 2 files: <b> transformer_utils.py </b> and <b> ViT Experiments.py</b>
1.  <b> transformer_utils.py </b>: Contains key components of the transformer encoder. In this file, you would find:  
    *   A Single-head self attention layer implementation
    *   A Multi-head self attention layer
    *   A Positional encoder
    *   Transformer Encoder
    Which you can download and import to quickly build your own architecture

2. <b> ViT Experiments.py</b>: The notebook where I trained my transformer model to classify <b>Hat</b> or <b>No hat</b> images. All preprocessing and visualisations are done here.
   
4. Notes: These are a summary of observations corrections I made which i think would help with better undersatnding

# Data
This is a small dataset of 471 images. Link to the data is: [here](https://drive.google.com/drive/folders/1G_ok5crD1EXH4tjZ3yLpyAZr6lAvNBx1?usp=sharin#g)

# Encoder Design

![encoder1 design](https://github.com/logic-OT/TRANSFORMER-FROM-SCRATCH/assets/61668807/fd770a19-cce2-4de0-bbd7-d1a38cc3a4ff)


# Performance
The emphasis if this Repo was more on model Architecture rather than performance. That being said, the result were good
* <b> EPOCHS:</b> 100
* <b> TRAIN:</b> 0.93
* <b> TEST:</b> 0.91

# Attention Visualisation Examples
![download (15)](https://github.com/logic-OT/TRANSFORMER-FROM-SCRATCH/assets/61668807/b4065cf1-d08d-4d16-8fd8-d1ca9e757c4e)

![download (18)](https://github.com/logic-OT/TRANSFORMER-FROM-SCRATCH/assets/61668807/c6edd6f0-e84d-4f15-bc8b-27721e2aaa05)

![download (17)](https://github.com/logic-OT/TRANSFORMER-FROM-SCRATCH/assets/61668807/945527df-9653-4065-8bdb-49ba0e57a0ce)

![download (16)](https://github.com/logic-OT/TRANSFORMER-FROM-SCRATCH/assets/61668807/630e22ff-e5bd-41ea-a3ce-501b614af951)


       
