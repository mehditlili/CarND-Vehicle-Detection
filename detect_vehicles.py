import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from matplotlib import gridspec

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
# I just multiply my image with 255
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True,
                     for_plotting=False):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            elif color_space == 'Lab':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        else:
            feature_image = np.copy(image)
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image * 255, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
                hog_features[np.isnan(hog_features)] = 0
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        if for_plotting:
            features.append(file_features)
        else:
            features.append(np.concatenate(file_features))
    # Return list of feature vectors, if for plotting purpose, return each feature separately
    return features


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan / nx_pix_per_step) - 1
    ny_windows = np.int(yspan / ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = np.int(xs * nx_pix_per_step + x_start_stop[0])
            endx = startx + xy_window[0]
            starty = np.int(ys * ny_pix_per_step + y_start_stop[0])
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, tuple(map(int, list(bbox[0]))), tuple(map(int, list(bbox[1]))), color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True,
                        for_plotting=False):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        elif color_space == 'Lab':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image * 255, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=for_plotting, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=for_plotting, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    if for_plotting:
        return img_features, feature_image
    else:
        return np.concatenate(img_features)


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


def display_sliding_windows(dpi=96):
    image = mpimg.imread("test_images/test1.jpg")
    image = cv2.resize(image, (0, 0), fx=1.0 / scale_factor, fy=1.0 / scale_factor)
    # image = image.astype(np.float32) / 255.0
    windows_1 = slide_window(image, x_start_stop=[200 / scale_factor, None], y_start_stop=y_start_stop,
                             xy_window=(100 / scale_factor, 100 / scale_factor), xy_overlap=(0.5, 0.5))
    windows_2 = slide_window(image, x_start_stop=[200 / scale_factor, None], y_start_stop=y_start_stop,
                             xy_window=(75 / scale_factor, 75 / scale_factor), xy_overlap=(0.5, 0.5))
    window_img_1 = draw_boxes(image, windows_1, color=(0, 0, 255), thick=2)
    window_img_2 = draw_boxes(image, windows_2, color=(0, 255, 0), thick=2)

    fig1 = plt.figure(4, figsize=(700 / dpi, 400 / dpi), dpi=dpi)
    plt.imshow(window_img_1)
    fig2 = plt.figure(5, figsize=(700 / dpi, 400 / dpi), dpi=dpi)
    plt.imshow(window_img_2)
    fig1.savefig("output_images/big_search_windows.png")
    fig2.savefig("output_images/small_search_windows.png")


def display_features(dpi=96):
    # Read in cars and notcars
    cars = glob.glob('vehicles/**/*.png', recursive=True)
    notcars = glob.glob('non-vehicles/**/*.png', recursive=True)
    # choose random 5 car images and 3 not car images
    rand_idx = np.random.randint(0, min(len(notcars), len(cars)), 5)
    cars = np.array(cars)[rand_idx]
    notcars = np.array(notcars)[rand_idx]
    for idx, (car, notcar) in enumerate(zip(cars, notcars)):
        car_image = mpimg.imread(car)
        noncar_image = mpimg.imread(notcar)
        car_features, feat_img_car = single_img_features(car_image, color_space=color_space,
                                                         spatial_size=spatial_size, hist_bins=hist_bins,
                                                         orient=orient, pix_per_cell=pix_per_cell,
                                                         cell_per_block=cell_per_block,
                                                         hog_channel=hog_channel, spatial_feat=spatial_feat,
                                                         hist_feat=hist_feat, hog_feat=hog_feat,
                                                         for_plotting=True)

        noncar_features, feat_img_not_car = single_img_features(noncar_image, color_space=color_space,
                                                                spatial_size=spatial_size, hist_bins=hist_bins,
                                                                orient=orient, pix_per_cell=pix_per_cell,
                                                                cell_per_block=cell_per_block,
                                                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                                                hist_feat=hist_feat, hog_feat=hog_feat,
                                                                for_plotting=True)
        # Plot the examples
        fig = plt.figure(figsize=(1080 / dpi, 960 / dpi), dpi=dpi)
        gs = gridspec.GridSpec(8, 8)
        plt.subplot(gs[:2, :2])
        plt.imshow(car_image)
        plt.title('Example Car Image')
        plt.subplot(gs[:2, 4:6])
        plt.imshow(noncar_image)
        plt.title('Example Not-car Image')

        # Dsiplay hog features
        plt.subplot(gs[:2, 2:4])
        plt.imshow(car_features[2][1], cmap='jet')
        plt.title('Car HOG')
        plt.subplot(gs[:2, 6:8])
        plt.imshow(noncar_features[2][1], cmap='jet')
        plt.title('Not-car HOG')

        # Display all three channels of the color space used
        plt.subplot(gs[2:4, :4])
        plt.imshow(np.hstack((feat_img_car[:, :, 0], feat_img_car[:, :, 1], feat_img_car[:, :, 2])))
        plt.title("YCrCb Channels")
        plt.subplot(gs[2:4, 4::])
        plt.imshow(np.hstack((feat_img_car[:, :, 0], feat_img_car[:, :, 1], feat_img_car[:, :, 2])))
        plt.title("YCrCb Channels")

        # Plot a figure with all three bar charts
        hist = car_features[1]
        bin_edges = np.arange(hist.shape[0]) + 1
        plt.subplot(gs[4:6, 0:4])
        plt.bar(bin_edges, hist)
        plt.title('Car Color Histogram')

        hist = noncar_features[1]
        bin_edges = np.arange(hist.shape[0]) + 1
        plt.subplot(gs[4:6, 4:])
        plt.bar(bin_edges, hist)
        plt.title('Not Car Color Histogram')

        spacial = car_features[0]
        plt.subplot(gs[6:8, 0:4])
        plt.plot(spacial)
        plt.title("Car binned spatial color")

        spacial = noncar_features[0]
        plt.subplot(gs[6:8, 4:])
        plt.plot(spacial)
        plt.title("Not-Car binned spatial color")
        plt.tight_layout()
        fig.savefig("output_images/example_features_%s.png" % idx)
        plt.close(fig)


def train_model():
    # Read in cars and notcars
    cars = glob.glob('vehicles/**/*.png', recursive=True)
    notcars = glob.glob('non-vehicles/**/*.png', recursive=True)
    # If n_samples is defined, use it, otherwise keep all data
    if n_samples > 0:
        rand_idx = np.random.randint(0, len(cars), n_samples)
        cars = np.array(cars)[rand_idx]
        notcars = np.array(notcars)[rand_idx]
    print("Cars training set size: %s" % len(cars))
    print("Notars training set size: %s" % len(notcars))

    # Extract car features
    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    # Extract non car features
    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # svc = SVC(probability=True)
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    return svc, X_scaler


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    new_heatmap = heatmap.astype(np.uint8)
    new_heatmap[heatmap < threshold] = 0
    new_heatmap[heatmap >= threshold] = 1
    # Return thresholded map
    return new_heatmap


def find_cars(image, independent_images=False):
    global heatmap
    if heatmap is None:
        # Initialize heatmap if non existent
        heatmap = np.zeros((image.shape[0], image.shape[1], heatmap_history), dtype=np.int16)

    start = time.time()
    image = cv2.resize(image, (0, 0), fx=1.0 / scale_factor, fy=1.0 / scale_factor)
    draw_image = np.copy(image)
    # This caused me a lot of trouble, I didn't see it at the beginning
    image = image.astype(np.float32) / 255.0

    windows_1 = slide_window(image, x_start_stop=[200 / scale_factor, None], y_start_stop=y_start_stop,
                             xy_window=(120 / scale_factor, 120 / scale_factor), xy_overlap=(0.5, 0.5))
    windows_2 = slide_window(image, x_start_stop=[200 / scale_factor, None], y_start_stop=y_start_stop,
                             xy_window=(90 / scale_factor, 90 / scale_factor), xy_overlap=(0.5, 0.5))

    print("total number of windows: %s" % (len(windows_1) + len(windows_2)))
    hot_windows_1 = search_windows(image, windows_1, svc, X_scaler, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)

    hot_windows_2 = search_windows(image, windows_2, svc, X_scaler, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)

    if independent_images:
        #  If processing single test images, reinit heatmap at every frame
        heatmap *= 0
    else:
        # If video, update the queue and get the last heatmap in the first position
        heatmap[:, :, 1:] = heatmap[:, :, :-1]
        heatmap[:, :, 0] *= 0
    heatmap[:, :, 0] = add_heat(heatmap[:, :, 0], hot_windows_1)
    heatmap[:, :, 0] = add_heat(heatmap[:, :, 0], hot_windows_2)
    if independent_images:
        # If single images, accept any detection as positive
        heatmap_disp = apply_threshold(np.sum(heatmap, axis=2), 1)
    else:
        # Other wise apply the defined threshold
        heatmap_disp = apply_threshold(np.sum(heatmap, axis=2), heat_thresh)
    labels = label(heatmap_disp)
    print(labels[1], 'cars found')
    draw_image = draw_labeled_bboxes(draw_image, labels)
    print("Frame processing took %s" % (time.time() - start))
    if independent_images:
        return draw_image, labels[0]
    else:
        return draw_image


def test_on_images():
    for idx, img in enumerate(test_images):
        image = mpimg.imread(img)
        # Use the independent_images param to avoid stacking up heatmaps
        result, labels = find_cars(image, independent_images=True)
        mpimg.imsave("output_images/test_result_%s.png" % (idx + 1), result)
        mpimg.imsave("output_images/test_result_label_%s.png" % (idx + 1), labels)


def test_on_videos():
    video = "project_video.mp4"
    clip1 = VideoFileClip(video)
    white_clip = clip1.fl_image(find_cars)
    white_clip.write_videofile("output_images/result_project.mp4", audio=False)


color_space = "YCrCb"  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
test_images = glob.glob("test_images/*.jpg")
image = mpimg.imread(test_images[0])
scale_factor = 1  # Was used to reduce image size, but didnt make features smaller so I discarded it
y_start_stop = [image.shape[0] / (scale_factor * 2),
                (image.shape[0] - 30) / scale_factor]  # Min and max in y to search in slide_window()

n_samples = 0  # how many samples to use (0 means all)
heatmap = None
heatmap_history = 15  # How many heatmaps to keep in the queue
heat_thresh = 3  # min number of detections in the last 15 frams to count as valid positive

# Create images that display the kind of features used
display_features()
# Create images that display the used sliding windows
display_sliding_windows()
display_sliding_windows()
display_sliding_windows()
new_train = False  # Train or load model
model_name = "neo_model.pkl"
# If train new model or model is not existing
if len(glob.glob(model_name)) == 0 or new_train:
    svc, X_scaler = train_model()
    print("Saving model")
    joblib.dump((svc, X_scaler), model_name, compress=1)
# Otherwise load model
else:
    print("loading existing model")
    svc, X_scaler = joblib.load(model_name)

# Run code on single images
test_on_images()
# Run code on video
#test_on_videos()
