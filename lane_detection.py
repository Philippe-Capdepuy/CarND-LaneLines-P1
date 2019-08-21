#!/usr/bin/env python3
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math

"""
test images: 960x540
challenge: 1280x738
"""

def display_lines_on_image(lines,image):
    # Iterate over the output line structures and draw lines on a blank image
    line_image = np.copy(image)*0 # creating a blank to draw lines on
    for line in lines:
        cv2.line(line_image,(int(line.x1),int(line.y1)),(int(line.x2),int(line.y2)),(255,0,0),10)

    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0) 
    return lines_edges

def display_raw_lines_on_image(lines,image):
    # Iterate over the output "lines" and draw lines on a blank image
    line_image = np.copy(image)*0 # creating a blank to draw lines on
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0) 
    return lines_edges


def l2norm(values):
    return math.sqrt(sum([v**2 for v in values]))

def mix(a,b,ratio_a):
    return a*ratio_a + b*(1-ratio_a)

class Line:
    """
    Store the representation of a line as segment coordinates, direction vector, mean position, length and score
    Provides merging functionalities, extrapolation
    """

    def __init__(self,x1,y1,x2,y2):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.compute_from_coordinates()

    def reorient_coordinates(self):
        """
        Force y1 to be minimum and y2 maximum
        """
        if self.y1 > self.y2:
            self.x2,self.x1 = self.x1,self.x2
            self.y2,self.y1 = self.y1,self.y2

    def compute_from_coordinates(self):
        """
        Recompute line properties from its segment coordinates
        """
        self.reorient_coordinates()
        self.vector = [self.x2 - self.x1 , self.y2 - self.y1]
        self.length = l2norm(self.vector)
        self.angle = math.atan2(self.vector[1],self.vector[0])        
        self.x = (self.x1+self.x2)/2
        self.y = (self.y1+self.y2)/2
        self.score = self.length

    def get_x_given_y(self,y):
        """
        Compute X coordinate for a given Y coordinate based on mean position and vector
        """
        if self.vector[1]==0:
            dy = 0
        else:
            dy = (y - self.y) / self.vector[1]
        dx = dy * self.vector[0]
        x = self.x + dx        
        return x

    def merge(self,line):
        """
        Merge current line with another one based on score
        """

        # Compute weight of each line based on respective scores
        rself = self.score / (self.score+line.score)
        
        # Compute coordinates of merged line (minimum and maximum Y, extrapolate for X, mix with ratio)
        min_y = min(self.y1,line.y1)
        max_y = max(self.y2,line.y2)
        sx1 = self.get_x_given_y(min_y)
        lx1 = line.get_x_given_y(min_y)
        sx2 = self.get_x_given_y(max_y)
        lx2 = line.get_x_given_y(max_y)
        self.y1 = min_y
        self.y2 = max_y
        self.x1 = mix(sx1,lx1,rself)
        self.x2 = mix(sx2,lx2,rself)
        
        # Recompute new line, new score is sum of both lines scores
        score = self.score + line.score
        self.compute_from_coordinates()        
        self.score = score

    def __str__(self):
        return "%s length %f angle %f score %f"%(str(self.vector),self.length,self.angle,self.score)

def get_angle_difference(a1,a2):
    """
    Return smallest angle difference
    """
    a = a1 - a2
    while a>math.pi: a-=math.pi*2
    while a<-math.pi: a+=math.pi*2
    return a

def are_lines_similar(l1,l2):
    # Check angular difference is not too large
    delta_angle = get_angle_difference(l1.angle,l2.angle)
    if abs(delta_angle) > 0.1:
        return False

    # Check horizatonal distance at mean Y coordinates is not too large
    avgy = (l1.y + l2.y) / 2
    x1 = l1.get_x_given_y(avgy)
    x2 = l2.get_x_given_y(avgy)
    if abs(x1-x2)>25:
        return False
    return True

class LanesFinder:
    """
    Analyses successive frames to detect lanes and display them
    """

    def __init__(self):
        self.initialized = False        

    def process_image(self,image):
        """
        Process a frame and return the image with detected lanes drawn on it
        """

        # On first frame, compute image sizes and region of interest (as ratio of sizes)
        if not self.initialized:
            self.last_lines = []
            self.ysize = image.shape[0]
            self.xsize = image.shape[1]
            self.yregion = int(self.ysize*0.60)
            self.xregion_left = int(self.xsize*0.47)
            self.xregion_right = int(self.xsize*0.53)
            self.initialized = True

        # Extract raw lines
        lines , edges = self.find_lines_on_image(image)

        # Compute lane lines
        plines = self.group_lines(lines)
   
        # Draw lines on image
        result_image = display_lines_on_image(plines,image)
        
        return result_image

    def find_lines_on_image(self,image):
        """
        Extract raw lines on image (gaussian smoothing + canny + polygon mask +  hough)
        Parameters are more or less same as exercices
        """

        # Convert to gray
        gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

        # Define a kernel size and apply Gaussian smoothing
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

        # Define our parameters for Canny and apply
        low_threshold = 50
        high_threshold = 150
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

        # Next we'll create a masked edges image using cv2.fillPoly()
        mask = np.zeros_like(edges)   
        ignore_mask_color = 255   

        # This time we are defining a four sided polygon to mask
        imshape = image.shape
        vertices = np.array([[(0,self.ysize),(self.xregion_left, self.yregion), (self.xregion_right, self.yregion), (self.xsize,self.ysize)]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on
        rho = 2 # distance resolution in pixels of the Hough grid
        theta = np.pi/90 # angular resolution in radians of the Hough grid
        threshold = 15     # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 40 #minimum number of pixels making up a line
        max_line_gap = 20    # maximum gap in pixels between connectable line segments

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                    min_line_length, max_line_gap)

        return lines,edges



    def group_lines(self,lines):
        """
        Process all raw lines and try to merge them together and with previously identified lanes
        """
        
        # Transform relevant raw lines to Line instances
        plines = []
        for line in lines:
            for x1,y1,x2,y2 in line:
                l = Line(x1,y1,x2,y2)
                # Filter out lines with angles out of relevant ranges
                if l.angle>math.pi/3 and l.angle<2*math.pi/3:
                    continue
                if l.angle<math.pi/8 or l.angle>7*math.pi/8:
                    continue
                plines.append(l)
        
        # Initialize groups with previously found lanes
        groups = self.last_lines
        
        # Process all lines and merge to existing groups or add them as new groups
        # print("\nLines")
        for l in plines:
            # print(l)
            for g in groups:
                if are_lines_similar(l,g):
                    g.merge(l)
                    break
            else:
                groups.append(l)

        # Sort by reverse score and keep best two detections
        groups = sorted(groups,reverse=True,key=lambda v:v.score)
        # print ("\nGroups")        
        # for l in groups:
            # print (l)
        if len(groups)>2:
            groups = groups[:2]

        # Force high score in selected lanes for merging at next frame (similar to a moving average)
        for l in groups:
            l.score = 2000
        self.last_lines = groups

        return groups

if __name__=="__main__":
    # Process all images
    for img_file in os.listdir("test_images"):
        lanes_finder = LanesFinder()
        image = mpimg.imread("test_images/"+img_file)    
        result_image = lanes_finder.process_image(image)
        #mpimg.imsave("test_images_output/"+img_file,result_image)    

    # Process all videos
    for video_file in os.listdir("test_videos"):
        lanes_finder = LanesFinder()
        input_file = "test_videos/"+video_file
        output_file = "test_videos_output/"+video_file
        input_clip = VideoFileClip(input_file)
        output_clip = input_clip.fl_image(lanes_finder.process_image)
        output_clip.write_videofile(output_file, audio=False)
        