# thesis

참고 :
- https://pytorch.org/docs/stable/nn.html?highlight=convtranspose#torch.nn.ConvTranspose1d 
ConvTranspose1d 인풋이 (1024, 512, 2) 이면 1024 -> in_channels, 512 -> out_channels, 2 -> kernel_size 되는 것으로 알고 있는데요. 
- https://github.com/pytorch/tutorials/blob/master/beginner_source/dcgan_faces_tutorial.py
참고한 코드 중에 pytorch tutorial 코드
- https://towardsdatascience.com/whats-wrong-with-spectrograms-and-cnns-for-audio-processing-311377d7ccd
-> Convolution 2D 이용하지 않은 이유 
- https://github.com/rafalencar1997/Audio_Generation/blob/master/ganModels.py
-> Tensorflow로 되어 있는데 이 코드도 참고했습니다
- https://github.com/rbracco/fastai2_audio/blob/master/nbs/02_tutorial.ipynb
--> fastai_audio version 2 (experimental)인데 보니까 cnn model이 1 channel input 받는 셀이 만들 수 있더라고요
