import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt


def import_video(videofile):
    """Returns the given videofile as a VideoCapture"""
    return cv2.VideoCapture(videofile)


def play_video(video):
    """
    Simple method to play the given video.
    Will stop if no more frames are found or if 'q' is pressed on keyboard
    """
    while video.isOpened():
        ok, frame = video.read()
        if ok:
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    # Clean up after playing video
    video.release()
    cv2.destroyAllWindows()


def tracking(video):
    """
    Executes the tracking of the cable
    MOSSE tracker from OpenCV is used to locate the pantograph
    """

    # Initialize the figure for graph plotting
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    xs = []
    ys = []

    # Set up the tracker
    tracker = cv2.TrackerMOSSE_create()

    try:
        # Stops program if there is any problem with the video
        if not video.isOpened():
            sys.exit()

        # Reads the first frame so we can create the bounding box
        ok, frame = video.read()
        if not ok:
            sys.exit()

        # selectROI lets us select an area in the first frame that is the
        # region of interest
        bounding_box = cv2.selectROI(
            "Frame", frame, fromCenter=False, showCrosshair=True)

        # Initialize the MOSSE tracker
        tracker.init(frame, bounding_box)
        frame_number = 0

        # Run loop as long as there are more frames in the video
        while True:
            ok, frame = video.read()
            if not ok:
                break

            # Update the tracker and the bounding box for each frame
            ok, bounding_box = tracker.update(frame)
            if ok:
                # Crop the frame above the bounding box, find the contour
                cropped = crop(frame, bounding_box)
                contour = find_contour(cropped)
                transposed_contour = transpose(contour, bounding_box)

                # Find the intersection point and draw it on the frame
                intersection_point = (contour[1][0][0], contour[1][0][1])

                cv2.drawContours(
                    frame, [transposed_contour], -1, (0, 255, 0), thickness=3)
                cv2.circle(
                    frame, intersection_point, 8, (128, 0, 255), -1)
                cv2.imshow('frame', frame)

                # Set up plot to call animate() function for every fifth point
                frame_number += 1
                if frame_number % 5 == 0:
                    point = contour[1][0][0]
                    animate(frame_number, xs, ys, point, fig, ax)

                # Quit when no more frames or on pressed 'q'
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        # Clean up
        video.release()
        cv2.destroyAllWindows()

    # Exit if there is an error with leading the video
    except BaseException:
        sys.exit()


def transpose(contour, bounding_box):
    """
    Because the contour of the pantograph is found in a cropped frame,
    we have to transpose the contour values for height and width to fit
    the original frame
    """
    c = contour
    c[0][0][0] += bounding_box[0]
    c[1][0][0] += bounding_box[0]
    return c


def animate(frame_count, xs, ys, point, fig, ax):
    """
    Creates the graph tracking the position of the cable
    Updates the graph with one point each time it is executed
    """
    # Add x and y to lists
    xs.append(point)
    ys.append(frame_count)

    # Limit x and y lists to 300 items, divide by 5 because we only read every
    # fifth point
    xs = xs[-(300 // 5):]
    ys = ys[-(300 // 5):]

    # Draw x and y lists
    ax.clear()
    ax.set_ylabel("Frames processed")
    ax.set_xlabel("x-position of intersection")
    ax.set_title("Pantograph intersection tracker ")
    if frame_count > 300:
        ax.axis([400, 600, ys[0], ys[-1]])
    else:
        ax.axis([400, 600, 0, 300])
    ax.plot(xs, ys)

    fig.canvas.draw()
    graph = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    graph = graph.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    graph = cv2.cvtColor(graph, cv2.COLOR_RGB2BGR)
    cv2.imshow("Graph", graph)


def find_contour(frame):
    """
    Process each frame and draws a contour over the cable
    """
    processed = processing(frame)
    contours, _ = cv2.findContours(processed, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # The cable is chosen as the top-1 contour sorted on biggest area
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    # Creates an approximation of the cable line with fewer points
    epsilon = 0.1 * cv2.arcLength(contour[0], True)
    approx = cv2.approxPolyDP(contour[0], epsilon, True)

    return approx


def processing(frame):
    """
    Process one frame of the video using gray scale, masking, threshold,
    dilation and erosion
    """
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = 255 - image
    _, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    kernel_dil = np.ones((7, 7), np.uint8)
    kernel_ero = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(thresh, kernel_dil, iterations=1)
    erosion = cv2.erode(dilation, kernel_ero, iterations=1)

    return erosion


def hough_transformP(input_frame, min_length=50, min_gap=10):
    """
    Probabilistic Hough transform using OpenCV
    Input frame should be binary, that's why Canny edge detector is used
    The probabilistic version analyzes lines as subset of points and
    estimates the probability of these points to belong to the same line
    """
    copy = input_frame.copy()
    edges = cv2.Canny(copy, 70, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, min_length, min_gap)
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(copy, (x1, y1), (x2, y2), (255, 128, 0), 2)

    return copy


def crop(frame, bounding_box):
    """
    Crops the frame above the bounding box of the object tracking
    """
    copied = frame.copy()
    cropped = copied[2: int(bounding_box[1]), int(
        bounding_box[0]): int(bounding_box[0]) + int(bounding_box[2])]
    return cropped


def run_tracker(videofile):
    """
    Main method for running the program
    """
    video = import_video(videofile)
    tracking(video)


run_tracker('Eric2020.mp4')
