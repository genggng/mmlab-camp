from mmdet.apis import init_detector,inference_detector
import cv2
import skimage
import numpy as np

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, video_path,score_thr=0.3):
    vcapture = cv2.VideoCapture(video_path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)

    # Define codec and create video writer
    file_name = "splash.mp4"
    vwriter = cv2.VideoWriter(file_name,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps, (width, height))

    count = 0
    success = True
    while success:
        print("frame: ", count)
        # Read next image
        success, image = vcapture.read()
        if success:
            # OpenCV returns images as BGR, convert to RGB
            image = image[..., ::-1]
            # Detect objects
            bbox,seg = inference_detector(model,image)
            mask = []
            for i in range(100):
                if bbox[0][i][4]>score_thr:
                    mask.append(seg[0][i])
            mask = np.array(mask).transpose(1,2,0)
            # Color splash
            splash = color_splash(image, mask)
            # RGB -> BGR to save image to video
            splash = splash[..., ::-1]
            # Add image to video writer
            vwriter.write(splash)
            count += 1
    vwriter.release()
    print("Saved to ", file_name)




config_file = "./ballon.py"
checkpoint_file = "./balloon_epoch_24.pth"
video_path  = "./test_video.mp4"
score_thr = 0.3

if __name__ == "__main__":
    model = init_detector(config_file,checkpoint_file)
    detect_and_color_splash(model, video_path,score_thr)