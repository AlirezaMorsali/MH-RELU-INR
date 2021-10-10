import numpy as np




def calculate_parameters(in_features, n_nerouns, n_layers, n_heads, alpha, use_bias_last=False): 
    parameters = n_nerouns * (in_features + 1) + (n_layers - 1) * n_nerouns * (n_nerouns + 1) + alpha * n_heads
    if use_bias_last:
        parameters += n_heads
    return parameters




def calculate_alpha(n_parameters, in_features, n_nerouns_init, n_layers, n_heads, alpha_init, use_bias_last=False):
    buff = calculate_parameters(in_features, n_nerouns_init, n_layers, n_heads, alpha_init, use_bias_last)
    
    flag = False
    while buff <= n_parameters:
        alpha_init += 1
        flag = True
        
        if alpha_init > n_nerouns_init:
            break
        
        buff = calculate_parameters(in_features, n_nerouns_init, n_layers, n_heads, alpha_init, use_bias_last)
        #print(buff)
    
    if flag:
        alpha_init -= 1
    
    return alpha_init




def calculate_hyperparameter(n_parameters, in_features, n_nerouns_init, n_layers, n_heads, alpha_init, use_bias_last=False):
    buff = calculate_parameters(in_features, n_nerouns_init, n_layers, n_heads, alpha_init, use_bias_last)
    
    flag = False
    while buff <= n_parameters:
        alpha_init = calculate_alpha(n_parameters, in_features, n_nerouns_init, n_layers, n_heads, alpha_init, use_bias_last)
        n_nerouns_init += 1
        flag = True
        
        buff = calculate_parameters(in_features, n_nerouns_init, n_layers, n_heads, alpha_init, use_bias_last)
        #print(buff)
        
    if flag:
        n_nerouns_init -= 1
        
    
    print("Neuron : ", n_nerouns_init, ", alpha : ", alpha_init)
    return n_nerouns_init, alpha_init




def PSNR(loss_value, max_value=1.0):
    return 20 * np.log10(max_value / np.sqrt(loss_value))




def calculate_flops(input_shape, in_features, n_nerouns, n_layers, n_heads, use_bias_last=False, sigma=1): 
    flops = 2 * n_nerouns * in_features + n_nerouns + \
            2 * (n_layers - 1) * n_nerouns * n_nerouns + (n_layers - 1) * n_nerouns + \
            2 * n_nerouns * n_heads * sigma
    if use_bias_last:
        flops += n_heads
    return flops * (input_shape[0] * input_shape[1]) / n_heads




def interpolant(t):
    return t*t*t*(t*(t*6 - 15) + 10)


def generate_perlin_noise_2d(
        shape, res, tileable=(False, False), interpolant=interpolant
):
    """Generate a 2D numpy array of perlin noise.
    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).
    Returns:
        A numpy array of shape shape with the generated noise.
    Raises:
        ValueError: If shape is not a multiple of res.

    references : https://github.com/pvigier/perlin-numpy/blob/master/perlin_numpy/perlin2d.py
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]]\
             .transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1,:] = gradients[0,:]
    if tileable[1]:
        gradients[:,-1] = gradients[:,0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[    :-d[0],    :-d[1]]
    g10 = gradients[d[0]:     ,    :-d[1]]
    g01 = gradients[    :-d[0],d[1]:     ]
    g11 = gradients[d[0]:     ,d[1]:     ]
    # Ramps
    n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11

    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)




def perlin_noise(shape, n_random_image):

	Random_Image = np.zeros((n_random_image, shape[0], shape[1]))
	for counter in range(n_random_image):
	    noise = generate_perlin_noise_2d(shape, (2 ** counter, 2 ** counter))
	    noise = (noise - noise.min()) / (noise.max() - noise.min())
	    Random_Image[counter] = noise

	return Random_Image