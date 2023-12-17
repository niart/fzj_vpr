# This script is to preprocess the RGB frames in the same pipeline. No need "align.py", because we already know the starting point.
import dv_processing as dv
import cv2 as cv
import os
import csv

# Open a file
reader = dv.io.MonoCameraRecording("/Users/nwang/all_preprocessed_dataset/2nd_record/0_2nd_record.aedat4")
label_file = '/Users/nwang/all_preprocessed_dataset/2nd_record/0_label.csv'

# Variable to store the previous frame timestamp for correct playback
lastTimestamp = None
firstFrame = True

# Define the directory to save frames
saveDirectory = "/Users/nwang/all_preprocessed_dataset/rgb"  # update with your directory path

# Ensure the directory exists
os.makedirs(saveDirectory, exist_ok=True)

# Size for cropping and pooling
crop_size = 256
pool_size = (2, 2)  # Pooling window size

# Run the loop while camera is still connected
while reader.isRunning():
    # Read a frame from the camera
    frame = reader.getNextFrame()

    if frame is not None:
        # Print the timestamp of the received frame
        print(f"Received a frame at time [{frame.timestamp}]")
        if firstFrame:
            starting = frame.timestamp
            firstFrame = False

        # Get the image from the frame
        image = frame.image

        # Calculate coordinates for center crop
        center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
        startx = center_x - (crop_size // 2)
        starty = center_y - (crop_size // 2)

        # Crop the image to the center 256x256
        cropped_image = image[starty:starty+crop_size, startx:startx+crop_size]

        # Apply average pooling
        pooled_image = cv.blur(cropped_image, pool_size)

        # Save the frame if the current timestamp is greater than or equal to the start timestamp

        label_time = frame.timestamp
        label_index = 1514 + int((label_time - starting - 50 * 2000 * 211)*100/(10**6))  
        
        if label_index > 1519 and label_index < 17615:  
            # Open the CSV file and read its contents
            with open(label_file, 'r') as csv_file:
                # Create a CSV reader
                csv_reader = csv.reader(csv_file)
                # Skip the header row (row 1)
                next(csv_reader)
                # Loop through the rows
                for index, row in enumerate(csv_reader):
                    if index == label_index:
                        # Extract the label from the third column (index 2)
                        label = row[2]
                        break  # Exit the loop once the desired row is found

            # Construct the file path
            framePath = os.path.join(saveDirectory, f"label_{label}_timestamp_{frame.timestamp}.png")
            # Save the frame
            cv.imwrite(framePath, pooled_image)

            # Calculate the delay for correct playback pacing (optional, can be removed if not needed)
            delay = (2 if lastTimestamp is None else (frame.timestamp - lastTimestamp) / 1000)
            # Perform the sleep (optional, can be removed if not needed)
            cv.waitKey(int(delay))

            # Store timestamp for the next frame
            lastTimestamp = frame.timestamp