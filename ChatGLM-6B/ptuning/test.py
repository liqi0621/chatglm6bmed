import numpy as np 
from tensorflow.keras import Input,Model,layers,losses,callbacks 

logdir="/mandapeng16/lq/chatglm-6b"  
tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)#——————1——————
class TestModel(Model):#——————2——————
  def __init__(self):
    super().__init__()
    self.layer1 = layers.Dense(10) 
    self.layer2 = layers.Dense(1)
  def call(self,inputs):
    x = self.layer1(inputs) 
    x = self.layer2(x)
    return x
model = TestModel()#——————3——————
model.compile(optimizer='rmsprop',loss='mse')#——————4——————
model.fit(np.ones([3,10]),
          np.ones([3,1]),
          callbacks=[tensorboard_callback]) #——————5——————
