from sentence_transformers import CrossEncoder
from typing import List, Optional





class LocalReranker:
    #本地重排序 基于 Cross-Encoder 实现

  def __init__(self,model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2" , device  = 'cuda'):
    self.model_name =model_name
    self.device = device 
    self.model = None

  #首次加载模型
  def load_mode(self):
      if self.model is None:
         self.model = CrossEncoder(self.model_name, device=self.device)
         return self.model

  #重排
  def rerank(self,query:str,docs:Optional[list],top_n:Optional[int])-> Optional[list]:


      print (f"重排序的doc：",docs)
      