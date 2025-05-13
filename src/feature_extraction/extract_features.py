def extract_features(image_path):
    import cv2
    import numpy as np

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f'Failed to load image: {image_path}')

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate centering deviation
    height, width = gray.shape
    moments = cv2.moments(gray)
    cx = moments['m10'] / moments['m00'] if moments['m00'] != 0 else width // 2
    cy = moments['m01'] / moments['m00'] if moments['m00'] != 0 else height // 2
    centering_deviation = np.sqrt((cx - width // 2) ** 2 + (cy - height // 2) ** 2)

    # Calculate edge-gradient profiles
    edges = cv2.Canny(gray, 100, 200)
    edge_gradient = np.mean(edges)

    # Calculate corner-sharpness metrics
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corner_sharpness = len(corners) if corners is not None else 0

    # Calculate surface-texture descriptors using Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_descriptor = laplacian.var()

    return {
        'centering_deviation': centering_deviation,
        'edge_gradient': edge_gradient,
        'corner_sharpness': corner_sharpness,
        'texture_descriptor': texture_descriptor
    }