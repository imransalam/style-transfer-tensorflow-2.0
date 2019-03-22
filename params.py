
num_iterations = 1000
hist_ratio = 100
content_weight = 1e3
style_weight = 1e-2

content_layers = ['block5_conv2'] 
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]
histogram_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]


num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
