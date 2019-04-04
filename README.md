# style-transfer-tensorflow-2.0
Style transfer from Gatys et al and histogram loss added from https://arxiv.org/abs/1701.08893. Written in tensorflow2.0 with eager execution. A very minimalistic implementation.

## Usage 
To synthesize a new output image by applying a style over a content image, use this command. 
Dimensions (width x height) of both files should be the same.

`python synthesis.py --content_img 'path_to_content_img.png' --style_img 'path_to_style_img.png'`

To change some hyper parameters use the file `params.py`
