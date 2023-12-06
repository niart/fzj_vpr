# This is the file used to generate samples (a lot of .npy files) from .aedat4 files
# each .npy file contains a dictionary, {data, label} 
from dv import AedatFile
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import csv

# 读取文件
file = AedatFile("/Users/nwang/all_preprocessed_dataset/2nd_record/0_2nd_record.aedat4")
# 设置多个事件保存步长
frames_num = 50 # how many frames in a sample?
# 设置一个事件读取步长
event_frame_time = 2 * 1000 # 2ms # for how long to make one frame?
event = next(file['events'])
start_time = event.timestamp
starting = start_time
#first_event = copy.deepcopy(event)
blank_img = np.zeros((frames_num,) + (2,) + file['events'].size, dtype=np.float32)
save_count = 0 # count number of samples being saved from one dataset
num_events_in_a_sample = 0
c = 0 # count number of frames being saved in one sample, 0=====>frames_num

# Calculate the center and starting and ending indices for cropping
height, width = file['events'].size
#print(height, width)
center_y, center_x = height // 2, width // 2
start_y, end_y = center_y - 128, center_y + 128
start_x, end_x = center_x - 128, center_x + 128
plt_num_events_in_a_sample = []
plt_save_count = []
label = None
label_file = '/Users/nwang/all_preprocessed_dataset/2nd_record/0_label.csv'
row_index = 2
way_point_number = -1

while True:
    try:
        if c == 0:
            event_frame = blank_img.copy()
        while event.timestamp - start_time < event_frame_time:
            if event.polarity:
                event_frame[c, 0, event.y, event.x] += 1
            else:
                event_frame[c, 1, event.y, event.x] += 1
            # if event_1.polarity:
            #     event_frame[c, 2, event_1.y, event_1.x] += 1
            # else:
            #     event_frame[c, 3, event_1.y, event_1.x] += 1                
            num_events_in_a_sample +=1
            event = next(file['events'])
            # event_1 = next(file['events_1'])
        start_time = event.timestamp # the first event in the new frame
        c += 1
        if c == frames_num/2 + 1:
            label_time = event.timestamp # the timestamp of the first event in 101th frame
            #sample from 0th, increase in 6th!!   
            #print('label_time, starting, event_frame_time', label_time, starting, event_frame_time)
            label_index = 1514 + int((label_time - starting - frames_num * event_frame_time * 211)*100/(10**6)) 

            
            label_index = 1514 + int((label_time - starting - 50 * 2000 * 211)*100/(10**6))  
            if label_index > 1519 and label_index < 17615:  
            # if save_count < 211:
            #     label = 10  #'K'
            # else:
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
                            '''
                            if label is not None:
                                print("Label at row {row_index + 1}: {label}")
                            else:
                                print("Row {row_index + 1} not found in the CSV file.")
                            '''
                            break  # Exit the loop once the desired row is found
                print('the {}th sample has label'.format(save_count+1), label)
            else:
                label = None
                print('the {}th sample has label'.format(save_count), label)
            
        if c == frames_num: 
            # #plot num_events_in_a_sample -- save_count
            # plt_num_events_in_a_sample.append(num_events_in_a_sample)
            # plt_save_count.append(save_count)
            # plt.plot(plt_save_count, plt_num_events_in_a_sample, marker='o', linestyle='-')

            # # Add labels and a title
            # plt.xlabel('Sample Index (starting from 1)')
            # plt.ylabel('Number of events in a Sample_trip0_firstcamera')
            # plt.grid(True)
            # plt.savefig("Number of Events in a Sample_trip0_firstcamera")
            
            num_events_in_a_sample = 0              
            c = 0  # count number of samples being saved from one dataset

            if label != None:           
                # Crop the central 128*128 region
                event_frame_cropped = event_frame[:, :, start_y:end_y, start_x:end_x]
                event_img_squeeze = np.mean(event_frame_cropped.reshape(50, 2, 128, 2, 128, 2), axis=(3, 5))  
                # 保存事件图像
                if save_count % 50 == 0:
                    way_point_number +=1
                    connection_number = way_point_number
                    print('the {}th sample has the waypoint number '.format(save_count+1), way_point_number, '!!!!!!!!!!!!!!!')
                else:
                    connection_number = None
                        
                data_dict = {
                    'data': event_img_squeeze,
                    'label': label,
                    'connection_number': connection_number
                }
                # Save the data and label to the same file, overwriting the existing file
                np.save('/Users/nwang/all_preprocessed_dataset/chop_still_50_firstcamera_0/50_2nd_record_trip0_firstcamera_{}.npy'.format(save_count), data_dict)
                print("50_2nd_record_trip0_firstcamera_{}.npy is saved!!!".format(save_count), 'number of events is {}!!!!!!!!!'.format(num_events_in_a_sample))
                save_count += 1
  

    except StopIteration:
        break
print('number of samples saved from this dataset is...........', save_count)