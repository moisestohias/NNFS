# processing.py

""" Maybe we need to implement some data augmentation methods like: random rotation, crop, noises ... """

# !wget https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg/800px-Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg -O cat.jp
# img = imread("cat.jp")
# img.shape

class Transform:
  def __call__(self, x): return self.forward(x)
  def __repr__(self): return f"{self.layers_name}(Z)"
  def forward(self, input): raise NotImplementedError

class Resize(Transform):
  def __init__(self, crop_h, crop_w=None):
    if crop_w is None: crop_w = crop_h
    self.out_h = out_h
    self.out_w = out_w
  def forward(self, input): 
    return resize_bilinear(input, self.out_h, self.out_w)

class CenterCrop(Transform):
  def __init__(self, crop_h, crop_w=None):
    if crop_w is None: crop_w = crop_h
    self.crop_h = crop_h
    self.crop_w = crop_w
  def forward(self, input): 
    return image_center_crop(input, self.crop_h, self.crop_w)

class Normalize(Transform):
  def __init__(self, mean, std):
    self.mean, self.std = mean, std
  def forward(self, input): 
    return (input - input.mean())/input.std()

def image_center_crop(img, crop_h, crop_w=None):
  if crop_w is None: crop_w = crop_h
  if len(img.shape) == 3:
    # regardless of whether it's (H, W, C) or (C, H, W)
    H, W = img.shape[:2] if (img.shape[2] == 3 or img.shape[2] == 1) else img.shape[-2:]
  elif len(img.shape) == 2: H, W = img.shape
  else: raise ValueError("Image must be of the shape (H,W) or (N, H, W) or (H,W, N)")

  if crop_h > H: raise ValueError("Image Height must be larger than the crop_h")
  if crop_w > W: raise ValueError("Image Width must be larger than the crop_w")

  h_start = (H - crop_h) // 2
  w_start = (W - crop_w) // 2

  if len(img.shape) == 3:
    if img.shape[2] == 3: return img[h_start:h_start + crop_h, w_start:w_start + crop_w, :]
    else: return img[:, h_start:h_start + crop_h, w_start:w_start + crop_w]
  else: return img[h_start:h_start + crop_h, w_start:w_start + crop_w]

def resize_bilinear(input_image, out_h:int, out_w:int=None):
    in_h, in_w, channels = input_image.shape
    if out_w is None: out_w = out_h

    h_scale = in_h / out_h
    w_scale = in_w / out_w

    y_indices = np.arange(out_h).reshape(-1, 1) * h_scale
    x_indices = np.arange(out_w) * w_scale

    y_low = np.floor(y_indices).astype(int)
    y_high = np.minimum(y_low + 1, in_h - 1)

    x_low = np.floor(x_indices).astype(int)
    x_high = np.minimum(x_low + 1, in_w - 1)

    y_fraction = y_indices - y_low
    x_fraction = x_indices - x_low

    y_fraction_complement = 1 - y_fraction
    x_fraction_complement = 1 - x_fraction

    top_left = input_image[y_low, x_low]
    top_right = input_image[y_low, x_high]
    bottom_left = input_image[y_high, x_low]
    bottom_right = input_image[y_high, x_high]

    interpolated_values = (y_fraction_complement * x_fraction_complement)[:, :, None] * top_left + \
                          (y_fraction_complement * x_fraction)[:, :, None] * top_right + \
                          (y_fraction * x_fraction_complement)[:, :, None] * bottom_left + \
                          (y_fraction * x_fraction)[:, :, None] * bottom_right

    return interpolated_values.astype(np.uint8)


