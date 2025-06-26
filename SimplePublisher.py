import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image as I_MSG
from cv_bridge import CvBridge
from vmbpy import VmbSystem
import threading
import time
import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple
from io import BytesIO
import base64
import os
import matplotlib.pyplot as plt
import pandas as pd



class FabricPositionPublisher(Node):
    def __init__(self):
        super().__init__('fabric_position_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'fabric_position', 10)
        self.timer = self.create_timer(0.1, self.publish_pose)  # 10 Hz
        self.pose = Float32MultiArray()
        self.pose.data = [float(1352),float(646),float(-45)]
         

    def publish_pose(self):
        #self.get_logger().info(f"Publishing: x={self.pose.x}, y={self.pose.y}, theta={self.pose.theta}")
        self.publisher_.publish(self.pose)

#rclpy.init() #?
F_pub=0
class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            I_MSG,
            '/camera/camera/color/image_raw',
            self.listener_callback,
            10)
        self.bridge = CvBridge()
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        global rob_msg_count
        rob_msg_count+=1   
        print("got_an_image")
        image=self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        generate_edge_movement(image)
        #self.get_logger().info('I heard: "%s"' % msg.data)


def image_callback(msg):
    print

def main(args=None):
    global F_pub
    print("starting main")
    rclpy.init(args=args)
    F_pub = FabricPositionPublisher()
    node2 = MinimalSubscriber()
    F_pub.get_logger().debug("starting nodes")
    print("starting nodes")
   
    try:
        #rclpy.spin(F_pub)
        rclpy.spin(node2)
    except KeyboardInterrupt:
        pass
    finally:
        F_pub.destroy_node()
        node2.destroy_node()
        rclpy.shutdown()
# import image processing libraries


#import rospy

# load images from directiory data_1 into array

# load images from directory data_1 into array if their name starts with dat
def load_images_from_directory(directory: str,startswith: str) -> List[np.ndarray]:
    images = []
    for filename in os.listdir(directory):
        if filename.startswith(startswith) and filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            # read image as gayscale
             
            img =  cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Convert the image to Greyscale format
                images.append(img)
    return images
def load_image_names_directory(directory: str,img_names) -> List[np.ndarray]:
    images = []
    # load images from directory into a list, if their name is in the img_names list
    # order the images such that their index in the list is the same as their index in the img_names list
    for x in image_names:
        for filename in os.listdir(directory):
            if filename.startswith(x) and filename.endswith(".png"):
                img_path = os.path.join(directory, filename)
                # read image as gayscale
                img =  cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Convert the image to Greyscale format
                    images.append(img)
    # sort the images such that their index in the list is the same as their index in the img_names list
     
    return images
 
def cross_pattern(kern_size,line_width,rotation=0):
    # create a cross pattern kernel
    kernel = np.zeros((kern_size, kern_size), dtype=np.uint8)
    cv2.line(kernel, (0, kern_size//2), (kern_size-1, kern_size//2), 255, line_width)
    cv2.line(kernel, (kern_size//2, 0), (kern_size//2, kern_size-1), 255, line_width)
    # rotate the kernel by the specified angle
    if rotation != 0:
        kernel = cv2.getRotationMatrix2D((kern_size//2, kern_size//2), rotation, 1)
        kernel = cv2.warpAffine(kernel, kernel, (kern_size, kern_size))
    return kernel


##print the directory the code is running from
#print("Current working directory:", os.getcwd())
 # #print THE CONTENTS OF THE WORKING DIRECTORY
#print("Contents of the working directory:")
for filename in os.listdir("."):
    print(filename)

directory = "./data_2/IR_light_setup_final/"


# load the labeled data from the directory /data_1/ir_light_setup_final_2/labeled_data/ as a csv using pandas
labled_data = []
labled_data = pd.read_csv(os.path.join(directory+"labels/", "lab_data_1.csv"))
# extract the image names from the image column
 
image_names = labled_data["image"].tolist()
for i in range(len(image_names)):
    image_names[i] = image_names[i].split("-")[1] 
image_contours = labled_data["label"].tolist()
image_class= labled_data["answer_relevancy"].tolist()

# split each image name by the "-" character and take the second part
# remove the first part of the image name
 
#print(image_names)
# load images from the directory /data_1/ir_light_setup_final_2/ if their name is in the image_names list


images=load_image_names_directory(directory, image_names)
base_layers= load_images_from_directory(directory,"BASE_layer_")

blured_baselayers =[ ]
for x in range(len(base_layers)):
    layer= base_layers[x]
    layer= cv2.medianBlur(layer, 11)
    layer= layer.astype(np.float32)/255
    #layer = cv2.medianBlur(layer, 5)
    #layer= cv2.medianBlur(layer, 7)
    #layer= cv2.medianBlur(layer, 11)
    applyx=21
    for y in range(applyx):
        layer = cv2.GaussianBlur(layer, (21, 21), 0)
    #layer = ~layer
    gain=0
    weight=0.5
    layer=gain+ layer*weight
    layer = 1-layer

    blured_baselayers.append(layer)
    #cv2.imshow("bl", blured_baselayers[x])
    #cv2.waitKey(0)

print("set up base")
 


textiles= load_images_from_directory(directory,"T")
# load the imgae called mask_1.png as a binary mask
mask= cv2.imread(os.path.join(directory, "mask_1.png"), cv2.IMREAD_GRAYSCALE)
# invert the mask
mask=cv2.bitwise_not(mask)
# do opening to remove edges
mask = cv2.erode(mask, np.ones((11, 11), np.uint8), iterations=4)
mask = cv2.threshold(mask, 30, 1, cv2.THRESH_BINARY)[1]

def extract_textile_cover_features(image: np.ndarray):
    textiledata = []
    for i in  textiles:
        text_rep=i
        tr2=i
        # mask image to isolate workspace  
        #multiply the image by the mask
        text_rep = text_rep * mask
        #text_rep = cv2

        # set values above 250 to 0 otherwise 255
        mask2 = cv2.inRange(text_rep, (0 ), (254))
        text_rep=cv2.bitwise_and(text_rep, text_rep, mask=mask2)

        cover = cv2.dilate(text_rep, np.ones((3,3), np.uint8), iterations=3)
        cover = cv2.erode(cover, np.ones((3,3), np.uint8), iterations=4)
        # set values above 0 to 255 otherwise 0 
        mask3 = cv2.threshold(cover, 30, 255, cv2.THRESH_BINARY)[1]
         
        

        # count nozero pixels
        count = cv2.countNonZero(mask3)
        # creat an image of the edges
        edges = cv2.Canny(mask3, 100, 200)
         
        
        
         
        # find the mean area rect rectangle of shape
        contours, _ = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        biggest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect( biggest_contour )
        # find the contours of the image

        template =cv2.boundingRect(biggest_contour) 
        # cut the bound rect out of the image with a 5 pixel margin
        x, y, w, h = template
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)  
        # cut the image
        cut = text_rep[y-5:y+h+5, x-5:x+w+5]
        

        ##print(rect)
        # draw the rectangle on the image
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(tr2, [box], 0, (0, 255, 0), 3)

        textiledata.append((count,rect,cut))
    return textiledata

def force_search(images,truths,ranges):
    this_cost = []
    #set lowest cost to max value of an int
    
    lowest_cost=int(1e10) 
    lowest_try = ""
    last_try = ""

    #print("searching for the lowest cost")
    at_maxp=ranges[0][1]*ranges[1][1]*ranges[2][1]*ranges[3][1]*ranges[4][1] 
    for i in range(ranges[0][0], ranges[0][1], ranges[0][2]):
        
        for j in range(ranges[1][0], ranges[1][1], ranges[1][2]):

            
            for k in range(ranges[2][0], ranges[2][1], ranges[2][2]):
                for l in range(ranges[3][0], ranges[3][1], ranges[3][2]):
                    for m in range(ranges[4][0], ranges[4][1], ranges[4][2]):
                        #print(str((i*j*k*l*m)/at_maxp)+"%")
                        #print("lowest",lowest_try)
                        #print("last",last_try)

                        for n in range(ranges[5][0], ranges[5][1], ranges[5][2]):
                            for o in range(ranges[6][0], ranges[6][1], ranges[6][2]):
                                sum_cost=0
                                for x in range(len(images)):
                                    image= images[x]
                                
                                    truth = truths[x]

                                    cutouts = get_image_cutouts(image, [i,j,k,l,m,n,o],truth)[0]
                                     
                                    # apply xor to the truth and the cutout
                                    c_rep= cutouts[0]-truth
                                    # take the sum of all values in the image
                                    cost = cv2.countNonZero(c_rep)
                                     
                                    
                                    ##print("cost: ", cost)
                                    sum_cost+=cost
                                this_try=[[sum_cost],[i,j,k,l,m,n,o]]
                                last_try=this_try
                                ##print(this_try)
                                if sum_cost<lowest_cost:
                                    lowest_cost=sum_cost
                                    lowest_try=this_try
                               
                                this_cost.append(this_try)

    return this_cost,lowest_cost,lowest_try

def find_intersection_and_angle(vx1, vy1, x1, y1, vx2, vy2, x2, y2):
    

    x3=x1+vx1
    y3=y1+vy1

    x4=x2+vx2
    y4=y2+vy2

    crossp=(x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if crossp!=0 :
                    
                    ##print("lines are not parallel")
                    # find the intersection point
        x = ((x1*y2-x2*y1)*(x3-x4)-(x1-x2)*(x3*y4-x4*y3))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
        y = ((x1*y2-x2*y1)*(y3-y4)-(y1-y2)*(x3*y4-x4*y3))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
                    
        dirx1= x2-x1
        diry1= y2-y1
        dirx2= x4-x3
        diry2= y4-y3
                    
        angle =np.arccos((dirx1*dirx2 + diry1*diry2 )/(( np.sqrt(dirx1**2 +diry1**2) )*(np.sqrt(dirx2**2 +diry2**2))))
                    #print(angle)
 
        return [x,y],angle
    return [0,0],0
    
    
    """
    Finds the intersection point of two lines and the angle at which they intersect.

    Parameters:
    vx1, vy1, x1, y1: Direction vector and a point on the first line.
    vx2, vy2, x2, y2: Direction vector and a point on the second line.

    Returns:
    intersection: (px, py) - The intersection point of the two lines.
    angle: The angle (in radians) at which the lines intersect.
    """
    # Line equations in parametric form:
    # Line 1: (x, y) = (x1, y1) + t1 * (vx1, vy1)
    # Line 2: (x, y) = (x2, y2) + t2 * (vx2, vy2)

    # Solve for t1 and t2 where the lines intersect
    A = np.array([[vx1, -vx2], [vy1, -vy2]])
    b = np.array([x2 - x1, y2 - y1])

    try:
        t = np.linalg.solve(A, b)
        t1, t2 = t

        # Calculate the intersection point
        px = x1 + t1 * vx1
        py = y1 + t1 * vy1
        intersection = (px, py)

        # Calculate the angle between the two lines
        dot_product = vx1 * vx2 + vy1 * vy2
        mag1 = np.sqrt(vx1**2 + vy1**2)
        mag2 = np.sqrt(vx2**2 + vy2**2)
        angle = np.arccos(dot_product / (mag1 * mag2))

        return intersection, angle
    except np.linalg.LinAlgError:
        # Lines are parallel or coincident
        return None, None
# Example usage

def find_line_intersections(search_lines,target_angle,anglethreshold):
    posible_corners=[]

    for line in search_lines:
            ##print("1")
            x1, y1, x2, y2 = line[0]
            for line2 in search_lines:
                ##print("2")
                x3, y3, x4, y4 = line2[0]
                
                ##print("3")
                # check if the lines intersect
                crossp=(x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
                if crossp!=0 :
                    
                    ##print("lines are not parallel")
                    # find the intersection point
                    x = ((x1*y2-x2*y1)*(x3-x4)-(x1-x2)*(x3*y4-x4*y3))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
                    y = ((x1*y2-x2*y1)*(y3-y4)-(y1-y2)*(x3*y4-x4*y3))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
                    
                    dirx1= x2-x1
                    diry1= y2-y1
                    dirx2= x4-x3
                    diry2= y4-y3
                    
                    angle =np.arccos((dirx1*dirx2 + diry1*diry2 )/(( np.sqrt(dirx1**2 +diry1**2) )*(np.sqrt(dirx2**2 +diry2**2))))
                    #print(angle)
 
                    if abs(angle-target_angle)<anglethreshold :
                        ##print("lines are perpendicular")
                        posible_corners.append((x,y,np.arctan2(diry1,dirx1))) 

    return posible_corners

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    # Define vectors
    A = np.array([line1[1][0]-line1[0][0], line1[1][1]-line1[0][1]])
    B = np.array([line2[1][0]-line2[0][0], line2[1][1]-line2[0][1]])

# Calculate dot product
    dot_product = A[0]*B[0]+A[1]*B[1]

# Calculate magnitudes (lengths of the vectors)
    magnitude_A = np.linalg.norm(A)
    magnitude_B = np.linalg.norm(B)

# Calculate angle in radians
    angle_radians = np.arccos(dot_product / (magnitude_A * magnitude_B))

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return [0,0],0

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y],angle_radians


def fit_polygon_to_graphs(lines_foc,lines_int,edge_img):
    shape=edge_img.shape
    intersections=[]
    angles=[]
    origin_lines=[]
    lid=0
    # chek lines for intersections
    for lin_1 in lines_foc:
        intersections_loc=[]
        angles_loc=[]
        ol_loc=[]
        #print(lin_1)
        #[vx1, vy1, x1, y1,size] = lin_1
        vx1=lin_1[0]
        vy1=lin_1[1]
        x1=lin_1[2]
        y1=lin_1[3]
        size1=lin_1[4]
        lsd=0
        for lin_2 in lines_int:
            #[vx2, vy2, x2, y2,size] = lin_2
            vx2=lin_2[0]
            vy2=lin_2[1]
            x2=lin_2[2]
            y2=lin_2[3]
            size2=lin_2[4]
 
            
                #print("lines could intersect")
            itc,angle = line_intersection([[x1+vx1*50,y1+ vy1*50],[ x1, y1]],[[x2+ vx2*50,y2+ vy2*50],[ x2, y2]])
            #angle=np.pi/2  

            #print(itc)
            if hasattr(itc, '__iter__'):
                if(lin_1!=lin_2):
                    if (1<itc[0] and itc[0]<shape[1]-1) and (1<itc[1] and itc[1]<shape[0]-1):
                        #print("lines intersect at"+str(itc))
                        intersections_loc.append(itc)
                        angles_loc.append(angle)
                        ol_loc.append(lsd)
                        cimg=edge_img.copy()
                        #cv2.circle(cimg, (int(itc[0]), int(itc[1])), 30, (255, 255, 255), -1)
                        #cv2.line(cimg,(int(x1),int(y1)),(int(x1+vx1*50),int(y1+vy1*50)),(120, 120, 120) ,3)
                        #cv2.line(cimg,(int(x2),int(y2)),(int(x2+vx2*50),int(y2+vy2*50)),(120, 120, 120) ,3)
                        #cv2.imshow("edgepoint",cimg)
                        #cv2.waitKey(0)
                        #cv2.destroyAllWindows()


                else:
                    print
                    #print(f"lines intersect outside image at {itc}")
            
            #print(f"line chekking line {lid} against line {lsd}")
            lsd+=1
        lid+=1
        intersections.append(intersections_loc)
        angles.append(angles_loc)
        origin_lines.append(ol_loc)
        #print(f"found intersections {len(intersections_loc)} for edge {lid-1}")
    #calculate confidence
    all_lines=[]
    for i in range(len(lines_foc)):
        intersects=intersections[i]# contains xy cordinates of each intersecton
        angled=angles[i]
        pxya=[]
        #P2=[]

        for d in range(len(origin_lines[i])):
            #it=origin_lines[i][d]
            pxya.append([intersects[d][0],intersects[d][1],angled[d]])
            #if it[0]==x:
                #intersects.append(it[1])
                #pxya.append([intersections[i][0],intersections[i][1],angles[i]])
            #elif it[1]==x:
                #intersects.append(it[0])
                #pxya.append([intersections[i][0],intersections[i][1],angles[i]])
        conf_lines=[]
        #print(f"found {len(pxya)} valid intersection")
        Cheked_combos=[]
        for x in pxya:
            for y in pxya:
                if(x!=y):
                    if Cheked_combos.count([x,y])+Cheked_combos.count([y,x])<1:
                        Cheked_combos.append([x,y])
                        all_lines.append([x,y,get_conf_line(lines_foc[i],x,y,736,np.pi/2,0.2,edge_img,[1,1,1,1])])
        #all_lines.append([conf_lines])
    return all_lines

def draw_confidence(img,top,bot):
    image=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    for lt in top:
        #print("drew_top_line")
        p1=lt[0]
        p2=lt[1]
        cv2.line(image,(int(p1[0][0]),int(p1[1][0])),(int(p2[0][0]),int(p2[1][0])),(0,255,0),3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (int((p1[0][0]+p2[0][0])/2), int((p1[1][0]+p2[1][0])/2))
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        cv2.putText(image, str(lt[2]), org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    for lb in bot:
        #print("drew_bot_line")
        p1=lb[0]
        p2=lb[1]
        cv2.line(image,(int(p1[0][0]),int(p1[1][0])),(int(p2[0][0]),int(p2[1][0])),(0,0,255),3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (int((p1[0][0]+p2[0][0])/2), int((p1[1][0]+p2[1][0])/2))
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        cv2.putText(image, str(lb[2]), org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    return image
    


def get_conf_line(line,p1,p2,E_len,E_ang,int_res,edges,weights):
    p1_cx,p1_cy,tval_1=closest_point_on_line(line[0],line[1],line[2],line[3],p1[0],p1[1])
    p2_cx,p2_cy,tval_2=closest_point_on_line(line[0],line[1],line[2],line[3],p2[0],p2[1])
    stp=[] # start from the lowest value
    edp=[]
    #print()
    if(tval_1<tval_2):
        stp=[p1,p1_cy,tval_1]
        edp=[p2_cx,p2_cy,tval_2]  
    else:
        stp=[p2_cx,p2_cy,tval_2]
        edp=[p1_cx,p1_cy,tval_1]
    #print(f"original points were p1 {p1} and p2 {p2}")
    #print(f"start {stp}")
    #print(f"ends {edp}")
    #print("confidence")
    C_len=np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    int_conf=1
    #print(int_conf)
    int__len=(E_len-abs(E_len-C_len))/E_len
    #print(int_conf)
    int_ang1=(E_ang-abs(E_ang-p1[2]))/E_ang
    #print(int_conf)
    int_ang2=(E_ang-abs(E_ang-p2[2]))/E_ang
    #print(int_conf)
    lint_x=int(p1[0])
    lint_y=int(p1[1])
    WH_C=1
    BL_C=1

    #print(f"interpolating at rez {int((C_len)/int_res)} steps on a length of {C_len}")

    for x in range(int((C_len)/int_res)):
        cval=x*int_res/C_len
        #print(cval)
        cint_x=int(p1[0]*cval+p2[0]*(1-cval))
        cint_y=int(p1[1]*cval+p2[1]*(1-cval))
        if(cint_x!=lint_x or cint_y!=lint_y):
            lint_x=cint_x
            lint_y=cint_y
            if(edges[cint_y,cint_x]>1):
                WH_C+=1
            else:
                BL_C+=1
    
    int_sup=WH_C/(WH_C+BL_C)
    
    #int_conf=int__len*weights[0]+int_ang1*weights[1]+int_ang2*weights[2]+int_sup*weights[3]
    #int_conf=int_conf/np.sum(weights)
    int_conf=int__len*int_ang1*int_ang2*int_sup
    #print(int_conf)
    return int_conf



    
      

            


def get_img_cost_f(truth,generated,weights):
    truth=truth.astype(np.uint8)
    generated=generated.astype(np.uint8)
    generated=cv2.inRange(generated, 120, 255)
    truth=cv2.inRange(truth, 120, 255)
    c_rep1=generated-truth
    c_rep2=truth-generated
    c_rep2=cv2.inRange(c_rep2, 120, 255)
    c_rep1=cv2.inRange(c_rep1, 120, 255)
    
    count= cv2.countNonZero(c_rep1)
    count2= cv2.countNonZero(c_rep2)

    count=count*weights[0]+count2*weights[1]
    return count
def get_image_cutouts(image, args, target):
    threshedl_contour=image.copy()
    threshedl_contour = threshedl_contour 
    #cv2.imshow("threshedl_contour", threshedl_contour)
    threshedl_contour = cv2.medianBlur(threshedl_contour, args[0])
    threshedl_contour = cv2.medianBlur(threshedl_contour, args[1])
    threshedl_contour = cv2.medianBlur(threshedl_contour, args[2])
    #cv2.imshow("threshedl_contour_1", threshedl_contour)
    threshedl_contour = cv2.inRange(threshedl_contour, args[3], args[4])
    threshedl_contour = cv2.morphologyEx(threshedl_contour, cv2.MORPH_CLOSE,np.ones((args[5],args[5])))
     
     
    # the 
    # apply a threshold to the image 
    #threshedl_contour = cv2.inRange(threshedl_contour, args[1], args[2])
    #cv2.imshow("threshedl_contour_2", threshedl_contour)
    # apply canny edge detection to the image
    threshedl_contour =cv2.Canny(threshedl_contour, args[6], args[7])
    #cv2.imshow("threshedl_contour_3", threshedl_contour)
     
    threshedl_contour = cv2.dilate(threshedl_contour, np.ones((args[8],args[8]), np.uint8), iterations=args[9])
    
    #cv2.imshow("threshedl_contour_4", threshedl_contour)
    #cv2.imshow("threshedl_contour_6", mask*255)
    conts = cv2.findContours(threshedl_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    threshedl_contour=threshedl_contour* mask*255
    threshedl_contour = (threshedl_contour*255).astype(np.uint8)
    #cv2.imshow("threshedl_contour_7", threshedl_contour)
    
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
      
    count=get_img_cost_f(target,threshedl_contour,[1,1])

    return threshedl_contour, conts,count
def get_image_cutouts_edgde_cut(image, args,target) :
    # this pice of code extracts the top layer based on the found contours
    # it does this via two main asumptions, 
    #       the generated edges are complete and connected
    #       
    #       as its is asumeed there will be overlap between the two textiles
    #       and the main part cuts out outer edges and top edges,
    #       both textiles will be in view, if both are the same size.
    #       the top part will be the largest region, as the overlap will reduce the size of
    #       all other regions, this only works for sewing similar pieces together,
    #           in order to generalize it, it could be possible to compare the features 
    #           of the connected componets with an established templates
    #           the connected components functions has variants which retrive
    #               size,boundingbox,circularity and so forth.

    # apply the mask to isolate the workspace
    threshedl_contour=image.copy()
    #threshedl_contour = threshedl_contour * mask
    
    # blur the image with medianblur to retain edges bigger than the kernel
    threshedl_contour = cv2.medianBlur(threshedl_contour, args[0])
    threshedl_contour = cv2.medianBlur(threshedl_contour, args[1])
    threshedl_contour = cv2.medianBlur(threshedl_contour, args[2])
    threshedl_contour = cv2.medianBlur(threshedl_contour, args[3])
    
     # treshold the image to find edges of a cerntain intensity
    mask_1 = cv2.inRange(threshedl_contour, args[4], args[5])
    # appy lcosing in case the edges are incomplete
        # this might be the cause of vanishing edges
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_CLOSE,np.ones((args[6],args[6])))
    
     # find edges in the image
    threshedl_contour =cv2.Canny(threshedl_contour, args[7], args[8])
    
    # expand edges to increas the structure of the contour
    threshedl_contour = cv2.dilate(threshedl_contour, np.ones((args[9],args[9]), np.uint8), iterations=args[10])
    #cv2.imshow("threshedl_contour_4", threshedl_contour)
    # apply the mask and set the image to the correct type
    threshedl_contour = threshedl_contour * mask_1
    threshedl_contour = (threshedl_contour*255).astype(np.uint8)
    #cv2.imshow("threshedl_contour_5", threshedl_contour)
    # find only the top 
    # invert the image to seperate each region of the found edge
    #threshedl_contour=cv2.morphologyEx(threshedl_contour, cv2.MORPH_CLOSE,np.ones((args[11],args[11])))
    threshedl_contour = threshedl_contour * mask
    retn=cv2.morphologyEx(threshedl_contour, cv2.MORPH_CLOSE,np.ones((args[11],args[11])))
    blobs=~retn # mark everythin that isnt an edge
    
    # make shure regions are within reach of the mask
    blobs= blobs*mask_1 # remove the outside of the surface, this makes it floats
    blobs=(blobs*255).astype(np.uint8)# bring it back to normal int format
    # make regions smaller to avoid accidental connections
    blobs=cv2.erode(blobs,np.ones((args[12],args[12]), np.uint8),args[13])
    #cv2.imshow("threshedl_contour_4", blobs)
    levels, image_ret,stats,centroids = cv2.connectedComponentsWithStats(blobs,4,cv2.CV_32S)
    # ignoring the first blob which accounts for 0 values, pick the layer id, of the layer with the most pixels
    ##print(stats)
    #save the stats of each region
    labels=stats[:,0]
    sizes=stats[:,4]
    #sizes[0]=0
    label=np.argmax(sizes)
    sizes[label]=0
    label=np.argmax(sizes)

    #print(stats)
    largest=np.where(image_ret == label, 1, 0)
    largest = largest.astype(np.uint8)*255 
    largest_d=cv2.dilate(largest,np.ones((args[14],args[14]), np.uint8),args[15])
    top_edges=cv2.Canny(largest_d, args[16], args[17])
    top_edges=cv2.dilate(top_edges,np.ones((args[18],args[18]), np.uint8),args[19])
    top_edges=cv2.medianBlur(top_edges, args[21])
    top_edges=cv2.medianBlur(top_edges, args[22])
    top_edges=cv2.morphologyEx(top_edges, cv2.MORPH_CLOSE,np.ones((args[23],args[23])))
    # knowing both the top visible edge, and the top textile edge means we can extact 
    # visible edges of the bottom textile via subtraction
     
    count=get_img_cost_f(target,threshedl_contour,[1,1])
    return threshedl_contour, top_edges,largest,count 

def get_obscured_edges(image, args,target,top_dilated):
     
    threshedl_contour=image.copy()
    threshedl_contour = cv2.medianBlur(threshedl_contour, args[0])
    threshedl_contour = cv2.medianBlur(threshedl_contour, args[1])
    #cv2.imshow("threshedl_contour", threshedl_contour)
    mask_baselayer =  cv2.inRange(threshedl_contour, args[2], args[3])
    mask_baselayer= np.where(mask_baselayer > 1, 1, 0).astype(np.uint8)
    mask_baselayer_inv= np.where(mask_baselayer == 0, 1, 0).astype(np.uint8)
    mask_baselayer_inv = cv2.morphologyEx(mask_baselayer_inv, cv2.MORPH_CLOSE,np.ones((args[4],args[4])))
    #cv2.imshow("masked_base", mask_baselayer*255 )
    #cv2.imshow("masked_base_inv", mask_baselayer_inv*255 )
    #cv2.imshow("blured_baselayer", blured_baselayers[0] )
    # the soroundings should be unchanged, but the textile should be made acording to the baselayer
    masked_baselayer = (mask_baselayer)+((blured_baselayers[0])*(mask_baselayer_inv)) 
    
    threshedl_contour=threshedl_contour*masked_baselayer
    threshedl_contour=threshedl_contour.astype(np.uint8)
    #cv2.imshow("threshedl_contour_0.4", threshedl_contour)
    #cv2.waitKey(0)
    threshedl_contour = cv2.medianBlur(threshedl_contour, args[5])
    threshedl_contour = cv2.medianBlur(threshedl_contour, args[6])
    #threshedl_contour = cv2.medianBlur(threshedl_contour, args[0]+4)
    #threshedl_contour = cv2.medianBlur(threshedl_contour, args[0]+6)
    cv2.imshow("threshedl_contour_1", threshedl_contour)
    mask_1 = cv2.inRange(threshedl_contour, args[7], args[8])
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_CLOSE,np.ones((args[9],args[9])))
    threshedl_contour = threshedl_contour * mask_1/255
    threshedl_contour = (threshedl_contour*255).astype(np.uint8)
    threshedl_contour = cv2.Canny(threshedl_contour,args[10],args[11])
    #dilate the edges 
    #cv2.imshow("threshedl_contour_2", threshedl_contour)
    #cv2.imshow("mask_2", top_dilated)
    top_eroded=cv2.erode(top_dilated, np.ones((args[12],args[12]), np.uint8),iterations =args[13])
    threshedl_contour = np.bitwise_and( threshedl_contour,top_eroded)

    threshedl_contour = cv2.dilate(threshedl_contour, np.ones((args[14],args[14]), np.uint8),iterations =args[15])
    threshedl_contour = cv2.medianBlur(threshedl_contour, args[16] )
    #threshedl_contour = cv2.medianBlur(threshedl_contour, args[17] )
    #threshedl_contour = cv2.medianBlur(threshedl_contour, args[18] )
    #threshedl_contour = cv2.medianBlur(threshedl_contour, args[19] )
    #top_dilated = cv2.dilate(top_edges, np.ones((args[6],args[6]), np.uint8),iterations =4)
    #top_dilated = cv2.morphologyEx(top_dilated, cv2.MORPH_CLOSE,np.ones((64,64)))
    #threshedl_contour = threshedl_contour - top_dilated
    threshedl_contour = cv2.inRange(threshedl_contour, 120,255)
    #threshedl_contour = cv2.morphologyEx(threshedl_contour, cv2.MORPH_CLOSE,np.ones((3,3)))
    #threshedl_contour = cv2.erode(threshedl_contour, np.ones((3,3), np.uint8),iterations =2)
     
    #cv2.imshow("threshedl_contour_3", threshedl_contour)
    #conts = cv2.findContours(threshedl_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    conts=0
    count=get_img_cost_f(target,threshedl_contour,[1,1])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return threshedl_contour,conts,count

def get_obs_edg_2(image,args,mask,baselayer):
    maks=np.where(mask>0,1,0)
    maks=maks.astype(np.uint8)
    img=(image*maks).astype(np.uint8)
    masked_baselayer = (1- maks)+((baselayer/255)*(maks))
    base_img=(masked_baselayer*img*255*(255/130)).astype(np.uint8)

    img_blur=cv2.medianBlur(base_img, 15)
    img_blur=cv2.medianBlur(img_blur, 11)
    #img_blur=cv2.GaussianBlur(img_blur,(15,15),0)
    #img_blur=cv2.medianBlur(img_blur, 13)
    #img_blur=cv2.medianBlur(img_blur, 15)
    img_blur=cv2.GaussianBlur(img_blur,(11,11),0)
    
     
    timg=cv2.adaptiveThreshold(img_blur, 255,
	        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 1)
    timg=timg*maks
    
    
    #timg=cv2.medianBlur(timg, 3)
    timg=cv2.dilate( timg, np.ones((5,5), np.uint8),7)
    timg=cv2.erode( timg, np.ones((5,5), np.uint8),5)
    timg=cv2.medianBlur(timg, 3)
    levels, image_ret,stats,centroids = cv2.connectedComponentsWithStats(timg,8,cv2.CV_32S) # label each of the blob
    threshold_size=250
    res_img=timg.copy()*0
    for x in range(levels)[1:]:
        if(stats[x,4]>threshold_size):
            res_img=res_img+np.where(image_ret==x,255,0)
    res_img=res_img*cv2.erode( maks, np.ones((3,3), np.uint8),1)
    res_img=res_img.astype(np.uint8)
    res_img=cv2.morphologyEx(res_img, cv2.MORPH_CLOSE,np.ones((5,5)))
    botoom_centroid=get_centeroid_img(image)
    cp_dir_dil=filter_away_from_midpoint(res_img,botoom_centroid)
     
    cp_dir_dil-=image
    cp_dir_dil=cp_dir_dil.astype(np.uint8)
    cp_dir_dil=cv2.dilate(cp_dir_dil,np.ones((5,5),np.uint8),11)
    cp_dir_dil=cp_dir_dil+image
    cp_dir_dil=cp_dir_dil.astype(np.uint8)
    cp_dir_dil=cv2.erode(cp_dir_dil,np.ones((3,3),np.uint8),17)
    res_img=res_img.astype(np.uint8)
     
    
    return res_img

def get_obs_edg_3(image,args,mask,baselayer):
    maks=np.where(mask>0,1,0)
    maks=maks.astype(np.uint8)
    img=(image*maks).astype(np.uint8)
    masked_baselayer = (1- maks)+((baselayer/255)*(maks))
    base_img=(masked_baselayer*img*255).astype(np.uint8)

    img_blur=cv2.medianBlur(base_img, 15)
    img_blur=cv2.medianBlur(img_blur, 11)
    #img_blur=cv2.GaussianBlur(img_blur,(15,15),0)
    #img_blur=cv2.medianBlur(img_blur, 13)
    #img_blur=cv2.medianBlur(img_blur, 15)
    img_blur=cv2.GaussianBlur(img_blur,(11,11),0)
    hist = cv2.calcHist([img_blur], [0], None, [254], [2, 256])
    ind = [i for i, val in enumerate(hist) if val != 0]
    
    lowest_val=ind[0]
    interval=ind[-1]-lowest_val
    scale=255/interval
    img_blur=(img_blur*scale-lowest_val).astype(np.uint8)

    #plt.plot(hist)
    #plt.title("Grayscale Histogram")
    #plt.xlabel("Pixel Intensity")
    #plt.ylabel("Frequency")
    #plt.show()
     
    timg=cv2.adaptiveThreshold(img_blur, 255,
	        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 1)
    timg=timg*maks
    
    
    #timg=cv2.medianBlur(timg, 3)
    timg=cv2.dilate( timg, np.ones((5,5), np.uint8),7)
    timg=cv2.erode( timg, np.ones((5,5), np.uint8),5)
    timg=cv2.medianBlur(timg, 3)
    levels, image_ret,stats,centroids = cv2.connectedComponentsWithStats(timg,8,cv2.CV_32S) # label each of the blob
    threshold_size=250
    res_img=timg.copy()*0
    for x in range(levels)[1:]:
        if(stats[x,4]>threshold_size):
            res_img=res_img+np.where(image_ret==x,255,0)
    res_img=res_img*cv2.erode( maks, np.ones((3,3), np.uint8),1)
    res_img=res_img.astype(np.uint8)
    res_img=cv2.morphologyEx(res_img, cv2.MORPH_CLOSE,np.ones((5,5)))
     
    #for x in range(20):
     #   timg=cv2.adaptiveThreshold(img_blur, 255,
	  #      cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, x)
      #  cv2.imshow(f"thres_{100+x*5}",timg)
    #can_x= cv2.Canny(img,50,70)


    #sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=args[1])
    #sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=args[1])
    
    #can_x= cv2.Canny(sobelx,100,155)
    #can_y= cv2.Canny(sobely,100,155)

    #can_x= cv2.dilate( can_x, np.ones((3,3), np.uint8),5)
    #can_y= cv2.dilate( can_y, np.ones((3,3), np.uint8),5)
    
    #sobelxn=np.bitwise_not(sobelx)
    #sobelyn=np.bitwise_not(sobely)
    
    #xy_sum=sobelx+sobely
    #xy_neg_sum=sobelxn+sobelyn
    #cv2.imshow("img",img)
    #cv2.imshow("img blr",img_blur)
    #cv2.imshow("xc",masked_baselayer)
    #cv2.imshow("yc",base_img)
    #cv2.imshow("thresh",timg) 
    #cv2.imshow("rimg",res_img)
     
    #cv2.imshow("sum",xy_sum)
    #cv2.imshow("bsum",xy_neg_sum)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return res_img



def get_obs_edg_2_ags(image,args,mask,baselayer):
    maks=np.where(mask>0,1,0)
    maks=maks.astype(np.uint8)
    img=(image*maks).astype(np.uint8)
    masked_baselayer = (1- maks)+((baselayer/255)*(maks))
    base_img=(masked_baselayer*img*255*(255/args[0])).astype(np.uint8)

    img_blur=cv2.medianBlur(base_img, args[1])
    img_blur=cv2.medianBlur(img_blur, args[2])
    #img_blur=cv2.GaussianBlur(img_blur,(15,15),0)
    #img_blur=cv2.medianBlur(img_blur, 13)
    #img_blur=cv2.medianBlur(img_blur, 15)
    img_blur=cv2.GaussianBlur(img_blur,(args[3],args[3]),args[4])
    #hist = cv2.calcHist([img_blur], [0], None, [254], [2, 256])
    #plt.plot(hist)
    #plt.title("Grayscale Histogram")
    #plt.xlabel("Pixel Intensity")
    #plt.ylabel("Frequency")
    #plt.show()
     
    timg=cv2.adaptiveThreshold(img_blur, 255,
	        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, args[5], args[6])
    timg=timg*maks
    
    
    #timg=cv2.medianBlur(timg, 3)
    timg=cv2.dilate( timg, np.ones((args[7],args[7]), np.uint8),args[8])
    timg=cv2.erode( timg, np.ones((args[9],args[9]), np.uint8),args[10])
    timg=cv2.medianBlur(timg, args[11])
    levels, image_ret,stats,centroids = cv2.connectedComponentsWithStats(timg,8,cv2.CV_32S) # label each of the blob
    threshold_size=args[12]
    res_img=timg.copy()*0
    for x in range(levels)[1:]:
        if(stats[x,4]>threshold_size):
            res_img=res_img+np.where(image_ret==x,255,0)
    res_img=res_img*cv2.erode( maks, np.ones((args[13],args[13]), np.uint8),args[14])
    res_img=res_img.astype(np.uint8)
    botoom_centroid=get_centeroid_img(image)
    cp_dir_dil=filter_away_from_midpoint(res_img,botoom_centroid)
     
    cp_dir_dil-=res_img
    cp_dir_dil=cp_dir_dil.astype(np.uint8)
    cp_dir_dil=cv2.dilate(cp_dir_dil,np.ones((args[15],args[15]),np.uint8),args[16])
    #cp_dir_dil=cp_dir_dil+res_img
    #cp_dir_dil=cp_dir_dil.astype(np.uint8)
    cp_dir_dil=cv2.erode(cp_dir_dil,np.ones((args[17],args[17]),np.uint8),args[18])
    res_img=cp_dir_dil.astype(np.uint8)
    #res_img=cv2.cv2.medianBlur(res_img, 7)
    #res_img=cv2.cv2.medianBlur(res_img, 7)
    #res_img=cv2.cv2.medianBlur(res_img, 7)
    res_img=cv2.morphologyEx(res_img, cv2.MORPH_CLOSE,np.ones((15,15)))
 
  
    return res_img
    

    



def consolidate_edges(edges,anglethres,distthresh):
    # for edges
    corrected_edges=[]
    for edge in range(len(edges)):
        vx, vy, x, y,size = edges[edge]
        angle=np.arctan2(vx,vy)
        Considered_edges=[edge]
        for edge_2 in range(len(edges)):
            vx_2, vy_2, x_2, y_2,size = edges[edge_2]
            angle_2=np.arctan2(vx_2,vy_2)
            if Considered_edges.count(edge_2)!=1:
                if (abs(angle-angle_2)<anglethres or abs(angle-angle_2+np.pi)<anglethres ):
                    distance = abs(vy * x_2 - vx * y_2 + (vx * y - vy * x)) / np.sqrt(vx**2 + vy**2)
                    if(distance<distthresh):
                        Considered_edges.append(edge_2)
        wa_ang_x=0
        wa_ang_y=0
        total_weight=0
        wa_pos_x=0
        wa_pos_y=0
        for cons_ed in Considered_edges:
            vx_c, vy_c, x_c, y_c,size_c = edges[cons_ed]
             
            wa_ang_x+=vx_c*size_c
            wa_ang_y+=vy_c*size_c
            wa_pos_x+=x_c*size_c
            wa_pos_y+=y_c*size_c
            total_weight+=size_c
        wvx=wa_ang_x/total_weight
        wvy=wa_ang_y/total_weight
        wx=wa_pos_x/total_weight
        wy=wa_pos_y/total_weight
        wsize=total_weight
        corrected_edges.append([wvx,wvy,wx,wy,wsize ])

        

                # the edge is not considerd aready
    return corrected_edges

  
def proces_contour_to_edge(image_src,args):
    labels=[]
    
    img_c=cv2.inRange(image_src,args[0],args[1])
    img_c=cv2.inRange(img_c,-1,20)
    levels, image_inv,stats_i,centroids = cv2.connectedComponentsWithStats(img_c,8,cv2.CV_32S) # label each of the blob
    
    Single_level_hole=np.where(image_inv == 1, 1, 0).astype(np.uint8)*255
    #print(stats_i)
    #Single_level_hole=~(Single_level_hole)
    Single_level_hole=cv2.inRange(Single_level_hole,-1,20)
    #Single_level_hole=Single_level_hole.astype(np.uint8)
    #cv2.imshow("edges", Single_level_hole)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    levels, image_ret,stats,centroids = cv2.connectedComponentsWithStats(Single_level_hole,8,cv2.CV_32S) # label each of the blob
    ##print(stats)
    ##print(levels)


    edge_lines=[]
    
    for d in range(levels)[1:]:
        size=stats[d,4]
        if(size>args[2]):
        
            tmp_IM= np.where(image_ret == d, 1, 0).astype(np.uint8)*255
        #cv2.imshow("edges", tmp_IM)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
            contours,hierarchy = cv2.findContours(tmp_IM, 1, 2)
            [xv,yv,x,y] = cv2.fitLine(contours[0], cv2.DIST_L2,0,0.01,0.01)
            centroid= centroids[d]
            ncx,ncy,Tv=closest_point_on_line(xv,yv,x,y,centroid[0],centroid[1])

            ##print("values were"+str([xv,yv,ncx,ncy]))
            edge_lines.append([xv,yv,ncx,ncy,size])
            #cv2.imshow("isolated"+str(x), tmp_IM)
            #cv2.waitKey(0)

    RGB_IM=cv2.cvtColor(image_src,cv2.COLOR_GRAY2BGR)
    rows,cols = RGB_IM.shape[:2]
    #if(len(edge_lines)<1):
     #   return []
    for d in edge_lines:
     #   #print(d)
        
        vx, vy, x, y,size = d
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        try:
            cv2.line(RGB_IM,(cols-1,righty),(0,lefty),(0,255,0),1)
        except: 
            print(f"line_failed_with_values: righty:{righty}, lefty:{lefty}")
    
    #cv2.imshow("image_with_lines", RGB_IM)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    ##print(edge_lines)
    
    return edge_lines

def closest_point_on_line(vx, vy, x, y, cx, cy):
   
    # Normalize the direction vector
    mag = np.sqrt(vx**2 + vy**2)
    vx /= mag
    vy /= mag

    # Vector from the line point to the given point
    dx = cx - x
    dy = cy - y

    # Project the vector onto the line's direction vector
    dot_product = dx * vx + dy * vy

    # Compute the closest point
    px = x + dot_product * vx
    py = y + dot_product * vy

    return px, py,dot_product
def draw_lines_on_image(in_image,lines,color):
    rows,cols = in_image.shape[:2]
    #if(len(lines[0])<2):
    #    return in_image
    ##print("drawing_lines")
    ##print(lines)
    ##print(len(lines))
    cont=0
    for d in lines:
        ##print(cont)
        ##print(d) 
        cont+=1
        
        
        vx, vy, x, y,size = d
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        ##print(lefty)
        ##print(righty)
        ##print(cols)

        if(type(lefty)==int and type(righty)==int):
            try:
                cv2.line(in_image,(int(cols-1),int(righty)),(int(0),int(lefty)),color,3)
            except:
                print(f"line_failed_with_values: righty:{righty}, lefty:{lefty}")
    return in_image
def draw_hugh_lines_on_image(in_image,lines,color):
    rows,cols = in_image.shape[:2]
    #if(len(lines[0])<2):
    #    return in_image
    ##print("drawing_lines")
    ##print(lines)
    ##print(len(lines))
    cont=0
    for d in lines:
        ##print(cont)
        ##print(d) 
        cont+=1
        
        
        x1, y1, x2, y2  = d[0]
      

       
        cv2.line(in_image,(x1,y1),(x2,y2),color,3)
           
    return in_image

def establish_truths(points):
    
        
     
        # the points list is a series of points where each point is connected to the next point, the last point is connected to the first point
        # the points list contains multiple lists on the format   {""points"":[[55.22625282204546,7.764650430492144],...,[52.581635429937,15.896894565549836]],""closed"":true,""polygonlabels"":[""top_textile_edge""],""original_width"":2048,""original_height"":1088}
        # convert the points to a list of point lists
    points=points.replace("true", 'True')   
    points = eval(points)
    ##print(points[0])
    cont_data_top =points[0]
    points_top = cont_data_top["points"]
    ow= cont_data_top["original_width"]
    oh= cont_data_top["original_height"]
    cont_data_bot= []
    points_bot = []
    # if points[1] is in the list then it is a bottom contour
    if len(points)>1:
        cont_data_bot = points[1]
        points_bot = cont_data_bot["points"]
        ##print(points_top["points"])
        # transform points to a image coordinates
    for i in range(len(points_top)):
        
        points_top[i][0] = int(points_top[i][0]*ow*0.01)
        points_top[i][1] = int(points_top[i][1]*oh*0.01)
        # repeat for the bottom points 
    for i in range(len(points_bot)):
        
        points_bot[i][0] = int(points_bot[i][0]*ow*0.01)
        points_bot[i][1] = int(points_bot[i][1]*oh*0.01)
    # create a blank image to draw the contours on 
   
    truth = np.zeros((int(oh), int(ow)), np.uint8)
    truth_top= truth.copy()*0
    truth_bot= truth.copy()*0
    truth_solid= truth.copy()*0
        # draw the contours on the truth image
    for i in range(len(points_top)):
            ##print(points_top[i])
            ##print(type(points_top[i]))
        s_id=i
        e_id=(i+1)%len(points_top)

        s_poi=(points_top[s_id][0],  points_top[s_id][1]) 
        e_poi=(points_top[e_id][0] ,  points_top[e_id][1])

            # draw the lines on the image
        cv2.line(truth, s_poi,e_poi, (255, 255, 255), 3)
        cv2.line(truth_top, s_poi,e_poi, (255, 255, 255), 3)
        # do the same for the bottom contour
    for i in range(len(points_bot)):
        ##print(points_bot[i])
            ##print(type(points_bot[i]))
        
        s_id=i
        e_id=(i+1)%len(points_bot)
        s_poi=(points_bot[s_id][0],  points_bot[s_id][1]) 
        e_poi=(points_bot[e_id][0] ,  points_bot[e_id][1])

            # draw the lines on the image
        cv2.line(truth, s_poi,e_poi, (255, 255, 255), 3)
        cv2.line(truth_bot, s_poi,e_poi, (255, 255, 255), 3)
    # fill in the area between the lines
    cv2.fillPoly(truth_solid, [np.array(points_top)], (255, 255, 255))
    # isolate only the edges which can be seen without depth vision
    truth_top_edges = truth_bot.copy()
    tmp=truth_solid
    
    truth_top_edges = truth_top_edges - cv2.erode(tmp, np.ones((3,3), np.uint8),iterations =2)  
    
    truth_top_edges = truth_top_edges.astype(np.uint8)

    truth_top_edges = truth_top_edges + truth_top*2
    truth_bot_edges = np.bitwise_and(truth,cv2.erode(tmp, np.ones((3,3), np.uint8),iterations =3)  ) 
    if len(points_bot)>0:
        # 
        cv2.fillPoly(truth_solid, [np.array(points_bot)], (255, 255, 255))
    # 
    truth_solid_edges = cv2.Canny(truth_solid, 100, 200)
    # dilate the edges
    truth_solid_edges = cv2.dilate(truth_solid_edges, np.ones((3,3), np.uint8), iterations=1)
    
    
    #cv2.imshow("whole truthe", truth)
    #cv2.imshow(" outer truthe", truth_solid_edges)
    #cv2.imshow("upper truth", truth_top_edges)
    #cv2.imshow("lower truth", truth_bot_edges)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return [truth, truth_top, truth_bot, truth_solid,truth_solid_edges,truth_top_edges, truth_bot_edges]

def get_centeroid_img(img):
    moments = cv2.moments(img)
    cx=0
    cy=0
# Calculate the centroid
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])  # x-coordinate of the centroid
        cy = int(moments["m01"] / moments["m00"])  # y-coordinate of the centroid
        #print(f"Centroid: ({cx}, {cy})")
    else:
        print("No nonzero pixels found.")
    return [cx,cy]
def filter_away_from_midpoint(img,Centroid):
    height, width = img.shape
    k1=np.zeros((3,3),np.uint8)
    k1[0,2]=1
    #k1[1,2]=1
    #k1[0,1]=1 
    # point is at the bottom left, use for upper right
    k2=np.zeros((3,3),np.uint8)
    k2[2,2]=1
    #k2[1,2]=1
    #k2[2,1]=1
    #point is at lower right user for upper left
    k3=np.zeros((3,3),np.uint8)
    k3[2,0]=1
    #k3[1,0]=1
    #k3[2,1]=1
    #point iis at upper right use for lower left
    k4=np.zeros((3,3),np.uint8)
    k4[0,0]=1
    #k4[1,0]=1
    #k4[0,1]=1
    # point is at upper left, use for lower right
    height, width = img.shape[:2]
    [cx, cy] = Centroid

    # Split the image into 4 quadrants
    top_left = img[0:cy, 0:cx]
    #top_left = img[0:cx, 0:cy]
    top_right = img[0:cy, cx:width]
    #top_right = img[cx:width,0:cy]
    bottom_left = img[cy:height, 0:cx]
    #bottom_left = img[ 0:cx,cy:height ]
    bottom_right = img[cy:height, cx:width]
    #bottom_right = img[cx:width,cy:height ]

    # Apply the filter to each quadrant
    top_left = cv2.filter2D(top_left,-1,k2) # corect
    top_right = cv2.filter2D(top_right,-1,k3)
    bottom_left = cv2.filter2D(bottom_left,-1,k1)
    bottom_right = cv2.filter2D(bottom_right,-1,k4)

    # Stitch the quadrants back together
    top = np.hstack((top_left, top_right))
    bottom = np.hstack((bottom_left, bottom_right))
    stitched_image = np.vstack((top, bottom))
    #stitched_image= cv2.dilate(stitched_image,np.ones((3,3),np.uint8),1)
    return stitched_image
    


def get_base_shapes_and_edges(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask_0=mask.astype(np.float32)
        ##print(np.max(mask_0))
        # mask image to isolate workspac
        # isolate the workspace
    masked_1 = img_rep * mask
        # isolate the textiles by making all white pixels black
        # set values above 250 to 0 otherwise 255
    mask_1 = cv2.inRange(masked_1, (0), (254))/255 # make a mask to the textiles out of the workspace
        # make mask one a float

    mask_1= mask_1.astype(np.float32)/255
        ##print(np.max(mask_1))

        
        
        # correct the image based on lighting
    masked_2 = masked_1 *  mask_1
        # make a cutout of the base layer using the masks
        
    cutout = ~blured_baselayers[0] 
        #iv_grad_mask= cutout.astype(np.float32)/255
        # make the cuttout into floats between 0 and 1 
    masked_3=masked_2 * cutout/255
        #plot histogram of the cutout using plt
        # detect corners with the goodFeaturesToTrack function. 
        
         
        
        #chek if values of an image is higher by a margin than another image on a pixel by pixel basis


        # sort the bottom layer out

        

          
         
    masked_2  = (masked_2*255).astype(np.uint8)
    masked_3  = (masked_3*255).astype(np.uint8)

    masked_2_blured = cv2.medianBlur(masked_2, 9)
    masked_2_blured = cv2.medianBlur(masked_2_blured, 11)
        # do histogram stretching
    masked_3_blured = cv2.medianBlur(masked_3, 9)
    masked_3_blured = cv2.medianBlur(masked_3_blured, 11)

    th2= masked_2 - blured_baselayers[2]-50 # the bottom layer is brighter than the top layer
        # so the top layer will persist
        #th2 = cv2.threshold(th2, 40, 255, cv2.THRESH_BINARY)[1]
    edges_1 = cv2.Canny(masked_2_blured, 20,30)
    edges_2 = cv2.Canny(masked_3_blured, 40,80)
    edges_3 = cv2.Canny(masked_3_blured, 5,15)
    edges_3_dilated = cv2.dilate(edges_1, np.ones((3,3), np.uint8), iterations=1)
    lines= cv2.HoughLinesP(edges_3_dilated, 1, np.pi/180, 100, minLineLength=30, maxLineGap=2)
        # draw the lines on the image
        #for line in lines:
         #   x1, y1, x2, y2 = line[0]
          #  cv2.line(masked_1, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # extend lines in both directions untill they intersect another line
    posible_corners = []
    for line in lines:
            ##print("1")
        x1, y1, x2, y2 = line[0]
        for i in lines:
                ##print("2")
            x3, y3, x4, y4 = i[0]
                
                ##print("3")
                # check if the lines intersect
            crossp=(x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
            if crossp!=0 :
                    
                    ##print("lines are not parallel")
                    # find the intersection point
                x = ((x1*y2-x2*y1)*(x3-x4)-(x1-x2)*(x3*y4-x4*y3))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
                y = ((x1*y2-x2*y1)*(y3-y4)-(y1-y2)*(x3*y4-x4*y3))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
                    # draw a circle at the intersection point
                    
                    #cv2.circle(masked_1, (int(x), int(y)), 2, (0, 255, 0), -1)
                     
                    # get the angle between the lines
                    
                    # using the cross product dot product rule get the angle between the lines
                dirx1= x2-x1
                diry1= y2-y1
                dirx2= x4-x3
                diry2= y4-y3
                    
                angle =np.arccos((dirx1*dirx2 + diry1*diry2 )/(( np.sqrt(dirx1**2 +diry1**2) )*(np.sqrt(dirx2**2 +diry2**2))))

                if abs(angle-np.pi/2)<0.01 :
                        ##print("lines are perpendicular")
                    posible_corners.append((x,y,np.arctan2(diry1,dirx1))) 

                    cv2.circle(masked_1, (int(x), int(y)), 5, (0, 125, 0), -1)
    corners_img = edges_3.copy()*0
        
    #print("found corners"+str(len(posible_corners)))
    for i in range(len(posible_corners)):
        x,y,angle=posible_corners[i]
            # draw a circle at the intersection point
            # apply a cross pattern to the image only at the intersection points

        temp= edges_3.copy()
        size=11 
        sh=size//2+5
        cross_patt=cross_pattern(size, 3, angle)
            # aply filter kernel only at the intersection point
            # cut out the area around the intersection point while taking image size into account
            #give me the syntax if shorthand if else statement in python
            # x_lower=if(int(x)-sh):0 else: int(x)-sh 
            #
        x_lower=int(x)-sh if((int(x)-sh)<0) else 0
        y_lower=int(y)-sh if((int(y)-sh)<0) else 0
        x_upper=int(x)+sh if((int(x)+sh)>edges_3.shape[1]) else edges_3.shape[1]
        y_upper=int(y)+sh if((int(y)+sh)>edges_3.shape[0]) else edges_3.shape[0]
            # cut out the area around the intersection point while taking image size into account
        temp =  temp[y_lower:y_upper, x_lower:x_upper]
            # aply the kernel to the cutout
        temp =cv2.filter2D(temp, -1 , cross_patt)
            # put the cutout back in the image
            
        corners_img[y_lower:y_upper, x_lower:x_upper] += temp
            # draw a circle at the intersection point
        cv2.circle(corners_img, (int(x), int(y)), 5, (0, 255, 0), -1)
        
            
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(masked_1, (x1, y1), (x2, y2), (0, 0, 0), 2)
        
        corners = cv2.goodFeaturesToTrack(edges_1, 27, 0.01, 10) 
        corners = np.int0(corners) 
  
        # we iterate through each corner,  
        # making a circle at each point that we think is a corner. 

    for i in corners: 
        x, y = i.ravel() 
        cv2.circle(edges_1, (x, y), 3, 255, -1) 

def optimize_cuts(images,truthes):
    cost_lists= []
     
    
    # mean kernel size, min threshold, max threshold, canny min, canny max, dilate size, iterations
    #asumptions=[13, 5, 5, 254, 255, 23, 25, 255, 3, 1, 1]
    asumptions=[13, 3, 5, 254, 255, 23, 25, 255, 3, 1, 1]
    # mean kernels size, min threshold, max threshold,clossing kn sie, canny min, canny max, dilate size, iterations
    asumptions_2=[3, 9, 3, 3, 82 , 250, 13, 1 , 1 , 23, 11, 13, 6, 13, 6 , 115, 185, 13, 3, 5, 5]
    asumptions_2=[1, 1, 1, 1, 10, 254, 1, 40, 50, 1, 1, 1, 1, 1, 1, 100, 185, 1, 1, 1]
    asumptions_2=[11, 11, 21, 1, 2, 254, 1, 115, 20, 5, 1, 1, 1, 1, 15, 100, 1, 1, 1, 1]
    asumptions_2=[11, 11, 21, 1, 2, 254, 1, 115, 20, 5, 1, 1, 5, 1, 5, 105, 5, 1, 1, 1,1]
    # mean kernels size, min threshold, max threshold,clossing kn sie, canny min, canny max, dilate size, iterations
    
    #asumptions_3=[254,255,13,15,250, 255,7,100,200,3,1,3,1,5,5,7,9] 

    #asumptions_3=[7,17,240,255,1,17,9,100, 255,7,15,20,3,1,3,1,3,7,9,11]
    #asumptions_3=[5, 27, 210, 305, 7, 19, 3, 115, 270, 1, 30, 5, 13, 6, 1, 6, 1, 1, 19, 21]
    #asumptions_3=[7, 7, 210, 305, 7, 19, 3, 115, 270, 1, 30, 5, 13, 6, 1, 6, 1, 1, 19, 21]
    asumptions_3=[130,15,11,11,0,13,1,5,7,5,5,3,250,3,1,5,11,3,17]
    #asumptions_3=[150, 5, 1, 1, 0, 3, 1, 3, 2, 15, 1, 13, 275, 13, 1, 1, 21, 8, 27]
    asumptions_3=[130, 15, 11, 11, 0, 7, 1, 5, 2, 5, 1, 3, 250, 7, 1, 1, 21, 1, 27]
    asumptions_3=[126, 11, 13, 7, 0, 7, 1, 5, 1, 5, 1, 5, 245, 13, 1, 1, 31, 1, 37]

    asumptions_3=[126, 11, 13, 7, 0, 7, 1, 5, 1, 5, 1, 5, 245, 13, 1, 1, 1, 1, 1]
    scales=[2,2,2,25,1,2,1,1,1,2,1]
    scales=[2,2,2,2,9,4,2,15,15,2,1,2,1,2,1,15,15,2,2,2,2,2,2,2,2]
    scales=[2,2,10,10,2,2,2,15,15,2,15,15,2,1,2,1,2,2,2,2,2,2,2]
    scales=[2,2,10,10,2,2,2,3,3,2,5,5,2,1,2,1,2,2,2,2,2,2,2]
    scales=[4,2,2,2,1,2,1,2,1,2,1,2,5,2,1,2,2,1,2,1]

    directions=[-1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,1,1,-1,1,1,1,1,1,1,1,1,1]
    directions=[-1,-1,1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,1,-1,-1,1,1,1,1,1]
    directions=[1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,-1]


    lastcost=0
    truths=[]
    top_conts_1=[]
    top_conts_2=[]
    top_masks_1=[]
    usedassum=asumptions_3
    for i in range(len(images)):
        these_truths=establish_truths(image_contours[i])
        truths.append(these_truths)
        im_conts_edges,top_edges,top_mask , cost2  = get_image_cutouts_edgde_cut(images[i], asumptions_2,these_truths[5])
        #get_obs_edg_2(images[i],asumptions_3,top_mask,blured_baselayers[0])
        
        top_conts_1.append(im_conts_edges)
        top_conts_2.append(top_edges)
        top_masks_1.append(top_mask)

    for index in range(len(usedassum))[:]: #len(usedassum)
        # chek above and below the current
        print(f"id:{index}")
        dircosts=np.array([0,0,0])
        increasing=True
        iter=0

        while(increasing==True and iter<5):
            iter+=1
            dircosts=np.array([0,0,0])
            for dir in range(3):
                a_cops=usedassum.copy()
                a_cops[index]=usedassum[index]+(dir-1)*scales[index]
                if(a_cops[index]<1):
                        a_cops[index]=1
                sumcost=0
                #print(f"dir is {dir-1}")
                for i in range(len(images)):
                    ##print("making truth ", i)
                    img_cp_1=images[i].copy()
         
                #truths.append(establish_truths(image_contours[i])[4])
                    these_truths=truths[i]
                    ##print("making truth ", i)
                    #im_conts_edges,top_edges,top_mask , cost2  = get_image_cutouts_edgde_cut(images[i], asumptions_2,these_truths[5])
                    im_conts_obscured   = get_obs_edg_2_ags(images[i],a_cops,top_masks_1[i],blured_baselayers[0])
                    im_conts_obscured = cv2.inRange(im_conts_obscured, 1, 255)
                    im_conts_obscured=im_conts_obscured.astype(np.uint8)
                    #im_conts_obscured=cv2.inRange(im_conts_obscured, 120, 255)
                
                    sumcost+=get_img_cost_f(these_truths[6],im_conts_obscured,[1,5])
                #print(sumcost)
                dircosts[dir]=sumcost
            #im_conts_edges = cv2.inRange(im_conts_edges, lower, upper)
            #im_conts_top_edges = cv2.inRange(top_edges, lower, upper)
            #im_conts_obscured = cv2.inRange(im_conts_obscured, lower, upper)
            #print(dircosts-dircosts[1])
            dircosts=dircosts-dircosts[1]
            if(usedassum[index]-1*scales[index]<1):
                dircosts[0]+=20
            print(dircosts)
            if( dircosts[0]>dircosts[2] and dircosts[2]< dircosts[1]):
                
                usedassum[index]+=1*scales[index]
                print(f"increasing value")

            elif(dircosts[0]< dircosts[2] and dircosts[0]< dircosts[1]):
                usedassum[index]-=1*scales[index]
                if(usedassum[index]<1):
                    usedassum[index]=1
                    increasing=False
                print(f"decreasing value")
            else:
                print("no_obvious_best_direction")
                #usedassum[index]+=1*scales[index]*directions[index]
                if(dircosts[0]==0 and dircosts[2]==0):
                    usedassum[index]+=1*scales[index]*directions[index]
                    if(usedassum[index]<1):
                        usedassum[index]=1
                        #increasing=False
                else:
                    increasing=False

                #increasing=False
                lastcost=dircosts[1]
        print(f"index: {index} was set to: {usedassum[index]} ")
    print(usedassum)
    print(lastcost)
    for ima in range(len(images)):
        im_conts,mask1 , cost               = get_image_cutouts(images[ima], asumptions,these_truths[4])
        im_conts_edges,top_edges,top_mask , cost2  = get_image_cutouts_edgde_cut(images[ima], asumptions_2,these_truths[5])
        im_conts_obscured   = get_obs_edg_2_ags(images[ima],usedassum,top_mask,blured_baselayers[0])
        top_edges = cv2.inRange(top_edges, 120, 255)
         
        im_conts_edges = cv2.inRange(im_conts_edges, 120, 255)
        im_conts_obscured=cv2.inRange(im_conts_obscured, 120, 255)
        cv2.imshow("top view", im_conts_edges)
        cv2.imshow("top textile", top_edges)
        cv2.imshow("bottom_vis", im_conts)
        cv2.imshow("bottom_obs", im_conts_obscured)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return asumptions
         

def optimize_lines(images,truthes):
    cost_lists= []
     
    
    # mean kernel size, min threshold, max threshold, canny min, canny max, dilate size, iterations
    #asumptions=[13, 5, 5, 254, 255, 23, 25, 255, 3, 1, 1]
    asumptions=[13, 3, 5, 254, 255, 23, 25, 255, 3, 1, 1]
    # mean kernels size, min threshold, max threshold,clossing kn sie, canny min, canny max, dilate size, iterations
    asumptions_2=[3, 9, 3, 3, 82 , 250, 13, 1 , 1 , 23, 11, 13, 6, 13, 6 , 115, 185, 13, 3, 5, 5]
    asumptions_2=[1, 1, 1, 1, 10, 254, 1, 40, 50, 1, 1, 1, 1, 1, 1, 100, 185, 1, 1, 1]
    asumptions_2=[11, 11, 21, 1, 2, 254, 1, 115, 20, 5, 1, 1, 1, 1, 15, 100, 1, 1, 1, 1]
    asumptions_2=[11, 11, 21, 1, 2, 254, 1, 115, 20, 5, 1, 1, 5, 1, 5, 105, 5, 1, 1, 1,1]
    # mean kernels size, min threshold, max threshold,clossing kn sie, canny min, canny max, dilate size, iterations
    
    #asumptions_3=[254,255,13,15,250, 255,7,100,200,3,1,3,1,5,5,7,9] 

    #asumptions_3=[7,17,240,255,1,17,9,100, 255,7,15,20,3,1,3,1,3,7,9,11]
    #asumptions_3=[5, 27, 210, 305, 7, 19, 3, 115, 270, 1, 30, 5, 13, 6, 1, 6, 1, 1, 19, 21]
    #asumptions_3=[7, 7, 210, 305, 7, 19, 3, 115, 270, 1, 30, 5, 13, 6, 1, 6, 1, 1, 19, 21]
    asumptions_3=[130,15,11,11,0,13,1,5,7,5,5,3,250,3,1,5,11,3,17]
    #asumptions_3=[150, 5, 1, 1, 0, 3, 1, 3, 2, 15, 1, 13, 275, 13, 1, 1, 21, 8, 27]
    asumptions_3=[130, 15, 11, 11, 0, 7, 1, 5, 2, 5, 1, 3, 250, 7, 1, 1, 21, 1, 27]
    asumptions_3=[126, 11, 13, 7, 0, 7, 1, 5, 1, 5, 1, 5, 245, 13, 1, 1, 31, 1, 37]

    asumptions_3=[126, 11, 13, 7, 0, 7, 1, 5, 1, 5, 1, 5, 245, 13, 1, 1, 1, 1, 1]
    asumptions_4=[1,260,60,50,1,90,20,20,50,255,300]
    asumptions_4=[2, 260, 60, 50, 1, 90, 20, 18, 10, 255, 225]# 1918
    asumptions_4=[2, 260, 60, 50, 1, 90, 20, 18, 10, 255, 150]
    
    asumptions_5=[3,2,1,260,70,70,12,1,260,60,50,7,90,60,10,255,300,20,50,20,50]
    asumptions_5=[3,2,100,300,3,2,1,260,70,70,12,1,260,60,50,7,90,60,25,25,10,255,300,10,255,150,20,50,20,50]

    scales=[2,2,2,25,1,2,1,1,1,2,1]
    scales=[2,2,2,2,9,4,2,15,15,2,1,2,1,2,1,15,15,2,2,2,2,2,2,2,2]
    scales=[2,2,10,10,2,2,2,15,15,2,15,15,2,1,2,1,2,2,2,2,2,2,2]
    scales=[2,2,10,10,2,2,2,3,3,2,5,5,2,1,2,1,2,2,2,2,2,2,2]
    scales=[4,2,2,2,1,2,1,2,1,2,1,2,5,2,1,2,2,1,2,1]
    scales=[4,2,2,2,1,2,1,2,1,2,1,2,5,2,1,2,2,1,2,1,2]
    scales=[1,1,1,12,10,10,4,1,12,10,10,1,3,3,1,15,15,3,3,3,3]

    scales=[1,5,4,4,4,2,2,2,10,10,15,2,2,2]
    scales=[2,1,5,5,2,1,1,12,10,10,4,1,12,10,10,1,3,3,1,15,15,3,3,3,3,3,3,3,3,3,3]

    directions=[-1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,1,1,-1,1,1,1,1,1,1,1,1,1]
    directions=[-1,-1,1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,1,-1,-1,1,1,1,1,1]
    directions=[1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,-1]
    directions=[1,-1,1,1,1,-1,1,1,-1,-1,-1,1,1,1]
    directions=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

    lastcost=0
    truths=[]
    top_conts_1=[]
    top_conts_2=[]
    top_masks_1=[]
    truth_lines_top=[]
    truth_lines_bot=[]
    usedassum=asumptions_5
    for i in range(len(images)):
        these_truths=establish_truths(image_contours[i])
        truths.append(these_truths)
        im_conts_edges,top_edges,top_mask , cost2  = get_image_cutouts_edgde_cut(images[i], asumptions_2,these_truths[5])
        tl_tp,tl_bt=establish_truth_on_Edges(these_truths[1],these_truths[2])
        truth_lines_top.append(tl_tp)
        truth_lines_bot.append(tl_bt)

        
        #get_obs_edg_2(images[i],asumptions_3,top_mask,blured_baselayers[0])
        
        top_conts_1.append(im_conts_edges)
        top_conts_2.append(top_edges)
        top_masks_1.append(top_mask)

    for index in range(len(usedassum))[:]: #len(usedassum)
        # chek above and below the current
        print(f"id:{index}")
        dircosts=np.array([0,0,0],np.float64)
        increasing=True
        iter=0

        while(increasing==True and iter<5):
            iter+=1
            dircosts=np.array([0,0,0])
            for dir in range(3):
                a_cops=usedassum.copy()
                a_cops[index]=usedassum[index]+(dir-1)*scales[index]
                if(a_cops[index]<1):
                        a_cops[index]=1
                sumcost=0
                #print(f"dir is {dir-1}")
                for i in range(len(images)):
                    ##print("making truth ", i)
                    img_cp_1=images[i].copy()
         
                #truths.append(establish_truths(image_contours[i])[4])
                    these_truths=truths[i]
                    ##print("making truth ", i)
                    #im_conts_edges,top_edges,top_mask , cost2  = get_image_cutouts_edgde_cut(images[i], asumptions_2,these_truths[5])
                    
                    im_conts,mask1 , cost               = get_image_cutouts(images[i], asumptions,these_truths[4])
                    im_conts_edges,top_edges,top_mask , cost2  = get_image_cutouts_edgde_cut(images[i], asumptions_2,these_truths[5])
                    im_conts_obscured   = get_obs_edg_2_ags(images[i],asumptions_3,top_mask,blured_baselayers[0])
                    #im_conts_obscured=cv2.inRange(im_conts_obscured, 120, 255)

                    #top_lines,bot_lines=find_and_consolidate_edges_arged(im_conts_obscured,im_conts,im_conts_edges,top_edges,im_conts,top_mask,usedassum)
                    top_lines=find_and_consolidate_top(top_edges,asumptions_4)
                    bot_lines=find_and_consolidate_bot(im_conts_edges,top_edges,im_conts,im_conts_obscured,a_cops)
                    bot_edges_conf_1=fit_polygon_to_graphs(bot_lines,bot_lines,top_edges)
                    bot_edges_conf_2=fit_polygon_to_graphs(bot_lines,top_lines,top_edges)
                    if (hasattr(bot_edges_conf_2,'__iter__') and hasattr(bot_edges_conf_1,'__iter__')):
                        bot_edges_conf_1.extend(bot_edges_conf_2)
                    
                    
                    sumcost+=conf_line_cost_func(truth_lines_bot[i],bot_edges_conf_1,[1,1])
                print(sumcost)
                dircosts[dir]=sumcost
                
            #im_conts_edges = cv2.inRange(im_conts_edges, lower, upper)
            #im_conts_top_edges = cv2.inRange(top_edges, lower, upper)
            #im_conts_obscured = cv2.inRange(im_conts_obscured, lower, upper)
            #print(dircosts-dircosts[1])
            print(dircosts)
            dircosts=dircosts-dircosts[1]
            if(usedassum[index]-1*scales[index]<1):
                dircosts[0]+=20
            print(dircosts)
            if( dircosts[0]>dircosts[2] and dircosts[2]< dircosts[1]):
                
                usedassum[index]+=1*scales[index]
                print(f"increasing value")

            elif(dircosts[0]< dircosts[2] and dircosts[0]< dircosts[1]):
                usedassum[index]-=1*scales[index]
                if(usedassum[index]<1):
                    usedassum[index]=1
                    increasing=False
                print(f"decreasing value")
            else:
                print("no_obvious_best_direction")
                #usedassum[index]+=1*scales[index]*directions[index]
                if(dircosts[0]==0 and dircosts[2]==0):
                    usedassum[index]+=1*scales[index]*directions[index]
                    if(usedassum[index]<1):
                        usedassum[index]=1
                        #increasing=False
                else:
                    increasing=False

                #increasing=False
                lastcost=dircosts[1]
        print(f"index: {index} was set to: {usedassum[index]} ")
    print(usedassum)
    print(lastcost)
    for ima in range(len(images)):
        im_conts,mask1 , cost               = get_image_cutouts(images[ima], asumptions,these_truths[4])
        im_conts_edges,top_edges,top_mask , cost2  = get_image_cutouts_edgde_cut(images[ima], asumptions_2,these_truths[5])
        im_conts_obscured   = get_obs_edg_2_ags(images[ima],asumptions_3,top_mask,blured_baselayers[0])
        top_edges = cv2.inRange(top_edges, 120, 255)
        
        top_lines=find_and_consolidate_top(top_edges,a_cops)
        im_conts_edges = cv2.inRange(im_conts_edges, 120, 255)
        im_conts_obscured=cv2.inRange(im_conts_obscured, 120, 255)
        #cv2.imshow("top view", im_conts_edges)
        cv2.imshow(f"top textile_{ima}", top_edges)
        #cv2.imshow("bottom_vis", im_conts)
        #cv2.imshow("bottom_obs", im_conts_obscured)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return asumptions
  
def show_edges(arg1,arg2,arg3,images,truths):
    for ima in range(len(images)):
        these_truths=truths[i]
        im_conts,mask1 , cost               = get_image_cutouts(images[i], arg1,these_truths[4])
        im_conts_edges,top_edges,top_mask , cost2  = get_image_cutouts_edgde_cut(images[i], arg2,these_truths[5])
        im_conts_obscured,conts_obs,cost3   = get_obscured_edges(images[i],arg3,these_truths[6],top_mask)
        
        im_conts_edges = cv2.inRange(top_edges, 120, 255)
        im_conts_obscured=cv2.inRange(im_conts_obscured, 120, 255)
        cv2.imshow("top view", im_conts_edges)
        cv2.imshow("top textile", top_edges)
        cv2.imshow("bottom_vis", im_conts)
        cv2.imshow("bottom_obs", im_conts_obscured)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main_2():
    cost_lists= []
     
    truths=[]
    # mean kernel size, min threshold, max threshold, canny min, canny max, dilate size, iterations
    asumptions=[5, 254, 255, 25, 255, 3, 1]
    # mean kernels size, min threshold, max threshold,clossing kn sie, canny min, canny max, dilate size, iterations
    asumptions_2=[7, 10, 250, 5,40, 110, 3, 1,3,1,3,1,40, 110,3,1,5]
    # mean kernels size, min threshold, max threshold,clossing kn sie, canny min, canny max, dilate size, iterations
    asumptions_3=[254,255,13,15,250, 255,7,100,200,3,1,3,1,5,5,7,9] 
    for i in range(len(images)):
        #print("making truth ", i)
        img_cp_1=images[i].copy()
         
        #truths.append(establish_truths(image_contours[i])[4])
        these_truths=establish_truths(image_contours[i])
        #print("making truth ", i)
        im_conts,mask1 , cost               = get_image_cutouts(images[i], asumptions,these_truths[4])
        im_conts_edges,top_edges,top_mask , cost2  = get_image_cutouts_edgde_cut(images[i], asumptions_2,these_truths[5])
        im_conts_obscured,conts_obs,cost3   = get_obscured_edges(images[i],asumptions_3,these_truths[6],top_mask)
        lower=120
        upper=255
        im_conts = cv2.inRange(im_conts, lower, upper)
        im_conts_edges = cv2.inRange(im_conts_edges, lower, upper)
        im_conts_top_edges = cv2.inRange(top_edges, lower, upper)
        im_conts_obscured = cv2.inRange(im_conts_obscured, lower, upper)
        #cv2.imshow("top view", im_conts_edges)
        #cv2.imshow("top textile", top_edges)
        #cv2.imshow("bottom_vis", im_conts)
        #cv2.imshow("bottom_obs", im_conts_obscured)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        ##print(np.unique(im_conts))
        ##print(np.unique(im_conts_edges))
        ##print(np.unique(im_conts_top_edges))
        ##print(np.unique(im_conts_obscured))
        top_lines = cv2.HoughLinesP(im_conts_top_edges, 1, np.pi/260, 60, minLineLength=50, maxLineGap=1)
        
        #im_conts_obscured= cv2.erode(im_conts_obscured,np.ones((3,3), np.uint8),iterations =1)
        im_conts_top_only=im_conts_edges-cv2.dilate(im_conts,np.ones((3,3), np.uint8),iterations =1)
        im_conts_obscured = cv2.inRange(im_conts_obscured, 120, 255)
        edge_lines,top_lines,bot_lines=[],[],[]

        #get the edges from the outline of the original photo
        #edge_lines = cv2.HoughLinesP(im_conts, 1, np.pi/180, 60, minLineLength=50, maxLineGap=1)
        
        # get the edges from the edge detection of the top fabric
        args=[50,255,500]
        top_lines = cv2.HoughLinesP(top_edges, 1, np.pi/260, 60, minLineLength=50, maxLineGap=1)
        top_corners=find_line_intersections(top_lines,90)
        top_separated=top_edges.copy()
        for f in top_corners:
            (x,y,angle)=f
            cv2.circle(top_separated, (int(x), int(y)), 30, (0, 0, 0), -1)
        top_edge_lines=proces_contour_to_edge( top_separated  , args )

        # separate the parts of the edges which are not the top textile edges
        exagerated_top=cv2.dilate(top_edges,np.ones((3,3), np.uint8),iterations =2).astype(np.uint8)
        exagerated_outer=cv2.dilate(im_conts,np.ones((3,3), np.uint8),iterations =2).astype(np.uint8)

        bottom_outer_sepparated=  (im_conts_edges - exagerated_top).astype(np.uint8) 
        bottom_obscured_sepparated= im_conts_obscured
        #cv2.imshow("top textile ", im_conts)
        #cv2.imshow("top textile_exag", exagerated_top)
        #cv2.imshow("outer edges_exag", exagerated_outer)
        #cv2.imshow("bottom_vis", bottom_outer_sepparated)
        
        #cv2.imshow("bottom_obs", bottom_obscured_sepparated)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        bot_lines=[]
        bot_lines_1 = cv2.HoughLinesP(bottom_outer_sepparated, 2, np.pi/260, 70, minLineLength=70, maxLineGap=12)
        bot_lines_2 = cv2.HoughLinesP(im_conts_obscured, 2, np.pi/260, 70, minLineLength=70, maxLineGap=12)
        if hasattr(bot_lines_1, '__iter__'):
            bot_lines.extend(bot_lines_1)
        #print(bot_lines_1)
        if hasattr(bot_lines_2, '__iter__'):
            bot_lines.extend(bot_lines_2)
        bot_corners_1 = find_line_intersections(bot_lines,90)
        #bot_corners_2 = find_line_intersections(bot_lines_2,90)
        bot_out_separated=bottom_outer_sepparated.copy()
        bot_obs_separated=im_conts_obscured.copy()
        for f in bot_corners_1:
            (x,y,angle)=f
            cv2.circle(bot_obs_separated, (int(x), int(y)), 20, (0, 0, 0), -1)
            cv2.circle(bot_out_separated, (int(x), int(y)), 20, (0, 0, 0), -1)
        bot_edge_lines_1=proces_contour_to_edge( bot_out_separated  , args )
         
        #for f in bot_corners_2:
        #    (x,y,angle)=f
        #    cv2.circle(bot_obs_separated, (int(x), int(y)), 20, (0, 0, 0), -1)
        bot_edge_lines_2=proces_contour_to_edge( bot_obs_separated  , args )
        bot_edge_lines_2=consolidate_edges(bot_edge_lines_2,np.deg2rad(45),300)
        bot_edge_lines_1.extend(bot_edge_lines_2)
        #print((bot_edge_lines_1))
        bot_edge_lines_1=consolidate_edges(bot_edge_lines_1,np.deg2rad(45),300)
        img_cp_1=cv2.cvtColor(img_cp_1,cv2.COLOR_GRAY2BGR)

        img_cp_1=draw_lines_on_image(img_cp_1,top_edge_lines,(0, 255, 0))
        img_cp_1=draw_lines_on_image(img_cp_1,bot_edge_lines_1,(0, 0, 255))
        img_cp_1=draw_lines_on_image(img_cp_1,bot_edge_lines_2,(255, 0, 0))

        # confidence is based on, 
        # how much of the line has marked edge pixels under it
        # how close the angle to sorounding edges are to 90 degrees 
            #thisone only works for our specific case

        # establish graphs and graph parameters for each edge
        # find intersections between each graph, 
        #   ignore intersections outside the image
        # use intersection points as endpoints of the graph

        #walk along the graph cheking the value of pixels underneath it
        # set initial confidence to whitepixelcount-blackpixelcount/whitepixelcount


        #form each layer into a set of functions and intersections

        # evaluate how well each line fits the points of the given edge

        # find the centerpoint and angle of the edge with highest confidence

        # secondly, find a placement point,
            # if more than three exposed bottom edges have been found
                # find the centerpoint and angle of the line with highest confidence and lenght
                # send the startpoint and endpoint to the robots controller
            # if ther isnt more than three edges of the bottom fabric
                # command the robot to go to an intermedeary position 
                    #outside the cameras view
                # extract the bottom fabrics placement and evaluate the
                    #the best placement position
                    # send the placement postion to the robot

         
      
        


        # aquire 3 sets of edges first outer edges, then top edges, then obscured edges 
        # by isolating lines in these it should be possible to get an understanding of the position of both textiles
        # extract lines using hughs lines
        # extract cornerpoints by finding lines which have angels between 90 and 70 in relation to eachother 
        
        # cut out corners to make each line into a seperate contour 
        # for each of these edge contours do a least squares regresion model 
        # find cornerpoints where each regresion model cross eachoter 
        # 
        #outer_edge_contours_isolated=cv2.findContours(im_conts, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #top_edge_contours_isolated=cv2.findContours(im_conts_top_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #bottom_edge_contours_isolated=cv2.findContours(im_conts_obscured, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #edge_graphs =[]
        #args=[50,255,200]
        #edge_lines_1=[]
        #edge_lines_1=proces_contour_to_edge( im_conts  , args )
        #edge_lines.extend(proces_contour_to_edge( im_conts_top_only   , args ))
        #edge_lines.extend(proces_contour_to_edge( im_conts_obscured , args ))
        
        #rows,cols = img_cp_1.shape[:2]
        #for x in edge_lines_1:
        #    #print("line")
        #    vx, vy, x, y = x
        #    lefty = int((-x*vy/vx) + y)
        #    righty = int(((cols-x)*vy/vx)+y)
        #    cv2.line(img_cp_1,(cols-1,righty),(0,lefty),(0,0,0),3)
        # for me tomorow, 
        # images dont account for areas where the textiles cross at a shalow angle



        #for x in top_lines:
           # x1, y1, x2, y2 = x[0]
          #  cv2.line(img_cp_2, (x1, y1), (x2, y2), (0, 0, 0), 1)
        #for x in bot_lines:
           # x1, y1, x2, y2 = x[0]
           # cv2.line(img_cp_2, (x1, y1), (x2, y2), (0, 0, 0), 1)
        ##print(edge_lines_1)
        #for d in edge_lines:
         #   #print(d)
          #  [vx,vy,x,y]=d
           # lefty = int((-x*vy/vx) + y)
           # righty = int(((cols-x)*vy/vx)+y)
           # cv2.line(img_cp_3,(cols-1,righty),(0,lefty),(0,255,0),2)


        #print("cost: ", cost2)
        # show the imag
        cv2.imshow("image", images[i])
        #cv2.imshow("truth1", these_truths[4])
        #cv2.imshow("contours1", im_conts)
        #cv2.imshow("truth2", these_truths[5])
        #cv2.imshow("contours_edges", im_conts_top_only)
        #cv2.imshow("truth3", these_truths[6])
        #cv2.imshow("contours_obscured", im_conts_obscured)

        cv2.imshow("drawn_lines_outer", img_cp_1)
        #cv2.imshow("drawn_lines_upper", img_cp_2)
        #cv2.imshow("drawn_lines_lower", img_cp_3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #cv2.imshow("contours", conts[0])
    #
    

    ranges=[( max(asumptions[0]-2,1),asumptions[0]+2,1),(max(asumptions[1]-30,1),asumptions[1]+30,5),(max(asumptions[2]-30,1),asumptions[2]+30,5),(max(asumptions[3]-10,1),asumptions[3]+10,5),(max(asumptions[4]-30,1),asumptions[4]+30,5),(max(asumptions[5]-2,1),asumptions[5]+2,1),(1,5,1)] 
    costs,lowest_v,lowest_a=force_search(images, truths, ranges)
    # convert the costs to a csv file
    costs_df = pd.DataFrame(costs, columns=["cost", "args"])
    costs_df.to_csv(os.path.join(directory+"costs.csv"), index=False)
    # costs are formated as [[cost],[i,j,k,l,m,n]], find the index in costs where the cost is the lowest
    # find the index of the lowest cost
    #print("lowest cost: ", lowest_v)
    #print("lowest args: ", lowest_a)


def find_and_consolidate_edges(obs_bot,vis_bot,vis_top,top_only,out_edg,top_mask,args):
    
        #top_lines = cv2.HoughLinesP(im_conts_top_edges, 1, np.pi/260, 60, minLineLength=50, maxLineGap=1)
        
        #im_conts_obscured= cv2.erode(im_conts_obscured,np.ones((3,3), np.uint8),iterations =1)
    im_conts_top_only=vis_top-cv2.dilate(out_edg,np.ones((3,3), np.uint8),iterations =1)
    im_conts_obscured = cv2.inRange(im_conts_obscured, 120, 255)
    edge_lines,top_lines,bot_lines=[],[],[]
    print("1")
    args=[50,255,500]
    top_lines = cv2.HoughLinesP(top_only, 1, np.pi/260, 60, minLineLength=50, maxLineGap=1)
    top_corners=find_line_intersections(top_lines,np.deg2rad(90),np.deg2rad(20))
    top_separated=top_only.copy()
    print("2")
    for f in top_corners:
        (x,y,angle)=f
        cv2.circle(top_separated, (int(x), int(y)), 30, (0, 0, 0), -1)
    top_edge_lines=proces_contour_to_edge( top_separated  , args )
    print("3")

        # separate the parts of the edges which are not the top textile edges
    exagerated_top=cv2.dilate(top_only,np.ones((3,3), np.uint8),iterations =2).astype(np.uint8)
    exagerated_outer=cv2.dilate(out_edg,np.ones((3,3), np.uint8),iterations =2).astype(np.uint8)

    bottom_outer_sepparated=  (vis_top - exagerated_top).astype(np.uint8) 
    bottom_obscured_sepparated= im_conts_obscured
    print("4")
 
    bot_lines=[]
     
    bottom_obscured_sepparated=obs_bot.copy()

    bot_lines_1 = cv2.HoughLinesP(bottom_outer_sepparated, 2, np.pi/260, 70, minLineLength=70, maxLineGap=12)
    bot_lines_2 = cv2.HoughLinesP(bottom_obscured_sepparated, 1, np.pi/260, 60, minLineLength=50, maxLineGap=7)

    if hasattr(bot_lines_1, '__iter__'):
        bot_lines.extend(bot_lines_1)
        #print(bot_lines_1)
    if hasattr(bot_lines_2, '__iter__'):
        bot_lines.extend(bot_lines_2)
    


    print
    bot_corners_1 = find_line_intersections(bot_lines,np.deg2rad(90),np.deg2rad(60))
        #bot_corners_2 = find_line_intersections(bot_lines_2,90)
    bot_out_separated=bottom_outer_sepparated.copy()
    bot_obs_separated=im_conts_obscured.copy()
    for f in bot_corners_1:
        (x,y,angle)=f
        cv2.circle(bot_obs_separated, (int(x), int(y)), 12, (0, 0, 0), -1)
        cv2.circle(bot_out_separated, (int(x), int(y)), 12, (0, 0, 0), -1)
    bot_edge_lines_1=proces_contour_to_edge( bot_out_separated  , [10,255,300] )
         
    bot_edge_lines_2=proces_contour_to_edge( bot_obs_separated  , args )
    bot_edge_lines_2=consolidate_edges(bot_edge_lines_2,np.deg2rad(20),50)
    bot_edge_lines_1.extend(bot_edge_lines_2)
      
    bot_edge_lines_1=consolidate_edges(bot_edge_lines_1,np.deg2rad(20),50)
    
    #img_cp_1= draw_lines_on_image(img_cp_1,top_edge_lines,(0,255,0))
    #img_cp_1= draw_lines_on_image(img_cp_1,bot_edge_lines_1,(0,0,255))
 

def find_and_consolidate_edges_arged(obs_bot,vis_bot,vis_top,top_only,out_edg,top_mask,args):
    
        #top_lines = cv2.HoughLinesP(im_conts_top_edges, 1, np.pi/260, 60, minLineLength=50, maxLineGap=1)
        
        #im_conts_obscured= cv2.erode(im_conts_obscured,np.ones((3,3), np.uint8),iterations =1)
    im_conts_top_only=vis_top-cv2.dilate(out_edg,np.ones((args[0],args[0]), np.uint8),iterations =args[1])
    im_conts_obscured = cv2.inRange(im_conts_obscured, args[2],args[3])
    edge_lines,top_lines,bot_lines=[],[],[]
    print("1")
    a_gs=[50,255,500]
    top_lines = cv2.HoughLinesP(top_only, args[4], args[5], args[6], minLineLength=args[7], maxLineGap=args[8])
    top_corners=find_line_intersections(top_lines,np.deg2rad(args[9]),np.deg2rad(args[10]))
    top_separated=top_only.copy()
    print("2")
    for f in top_corners:
        (x,y,angle)=f
        cv2.circle(top_separated, (int(x), int(y)), args[11], (0, 0, 0), -1)
    top_edge_lines=proces_contour_to_edge( top_separated  , [args[12],args[13],args[14]] ) 
    print("3")

        # separate the parts of the edges which are not the top textile edges
    exagerated_top=cv2.dilate(top_only,np.ones((args[15],args[15]), np.uint8),iterations =args[16]).astype(np.uint8)
    #exagerated_outer=cv2.dilate(out_edg,np.ones((3,3), np.uint8),iterations =2).astype(np.uint8)

    bottom_outer_sepparated=  (vis_top - exagerated_top).astype(np.uint8) 
    bottom_obscured_sepparated= im_conts_obscured
  
 
    bot_lines=[]
     
    bottom_obscured_sepparated=obs_bot.copy()

    bot_lines_1 = cv2.HoughLinesP(bottom_outer_sepparated, args[17], args[18], args[19], minLineLength=args[20], maxLineGap=args[21])
    bot_lines_2 = cv2.HoughLinesP(bottom_obscured_sepparated, args[22], args[23], args[24], minLineLength=args[25], maxLineGap=args[26])

    if hasattr(bot_lines_1, '__iter__'):
        bot_lines.extend(bot_lines_1)
        #print(bot_lines_1)
    if hasattr(bot_lines_2, '__iter__'):
        bot_lines.extend(bot_lines_2)
    


    print
    bot_corners_1 = find_line_intersections(bot_lines,np.deg2rad(args[27]),np.deg2rad(args[28]))
        #bot_corners_2 = find_line_intersections(bot_lines_2,90)
    bot_out_separated=bottom_outer_sepparated.copy()
    bot_obs_separated=im_conts_obscured.copy()
    for f in bot_corners_1:
        (x,y,angle)=f
        cv2.circle(bot_obs_separated, (int(x), int(y)), args[29], (0, 0, 0), -1)
        cv2.circle(bot_out_separated, (int(x), int(y)), args[29], (0, 0, 0), -1)
    bot_edge_lines_1=proces_contour_to_edge( bot_out_separated  , [args[30],args[31],args[32]] )
         
    bot_edge_lines_2=proces_contour_to_edge( bot_obs_separated  , [args[33],args[34],args[35]] )
    bot_edge_lines_2=consolidate_edges(bot_edge_lines_2,np.deg2rad(args[36]),args[37])
    bot_edge_lines_1.extend(bot_edge_lines_2)
      
    bot_edge_lines_1=consolidate_edges(bot_edge_lines_1,np.deg2rad(args[38]),args[39])
    
    #img_cp_1= draw_lines_on_image(img_cp_1,top_edge_lines,(0,255,0))
    #img_cp_1= draw_lines_on_image(img_cp_1,bot_edge_lines_1,(0,0,255))
    return top_edge_lines,bot_edge_lines_1

def find_and_consolidate_top(top_only,args):
    print

         #top_lines = cv2.HoughLinesP(im_conts_top_edges, 1, np.pi/260, 60, minLineLength=50, maxLineGap=1)
        
        #im_conts_obscured= cv2.erode(im_conts_obscured,np.ones((3,3), np.uint8),iterations =1)
    #im_conts_top_only=vis_top-cv2.dilate(out_edg,np.ones((args[0],args[0]), np.uint8),iterations =args[1])
    #im_conts_obscured = cv2.inRange(im_conts_obscured, args[2],args[3])
    #top_lines =None
    top_lines = cv2.HoughLinesP(top_only, args[0], np.pi/args[1], args[2], minLineLength=args[3], maxLineGap=args[4])
    if not hasattr(top_lines, '__iter__'):
        return
    top_corners=find_line_intersections(top_lines,np.deg2rad(args[5]),np.deg2rad(args[6]))
    top_separated=top_only.copy()

    for f in top_corners:
        (x,y,angle)=f
        cv2.circle(top_separated, (int(x), int(y)), args[7], (0, 0, 0), -1)
    top_edge_lines=proces_contour_to_edge( top_separated  , [args[8],args[9],args[10]] ) 

    return top_edge_lines

def find_and_consolidate_bot(vis_top,top_only,out_edg,obs_edg,args):
    im_conts_top_only=vis_top-cv2.dilate(out_edg,np.ones((args[0],args[0]), np.uint8),iterations =args[1])
    im_conts_obscured = cv2.inRange(obs_edg, args[2],args[3])
    
         # separate the parts of the edges which are not the top textile edges
    exagerated_top=cv2.dilate(top_only,np.ones((args[4],args[4]), np.uint8),iterations =args[5]).astype(np.uint8)
    #exagerated_outer=cv2.dilate(out_edg,np.ones((3,3), np.uint8),iterations =2).astype(np.uint8)

    
    bottom_outer_sepparated=  (vis_top - exagerated_top).astype(np.uint8) 
    bottom_obscured_sepparated= im_conts_obscured
    
 
    bot_lines=[]
     
    bottom_obscured_sepparated=obs_edg.copy()

    bot_lines_1 = cv2.HoughLinesP(bottom_outer_sepparated, args[6], args[7], args[8], minLineLength=args[9], maxLineGap=args[10])
    bot_lines_2 = cv2.HoughLinesP(bottom_obscured_sepparated, args[11], args[12], args[13], minLineLength=args[14], maxLineGap=args[15])

    if hasattr(bot_lines_1, '__iter__'):
        bot_lines.extend(bot_lines_1)
        #print(bot_lines_1)
    if hasattr(bot_lines_2, '__iter__'):
        bot_lines.extend(bot_lines_2)
    


    print
    bot_corners_1 = find_line_intersections(bot_lines,np.deg2rad(args[16]),np.deg2rad(args[17]))
        #bot_corners_2 = find_line_intersections(bot_lines_2,90)
    bot_out_separated=bottom_outer_sepparated.copy()
    bot_obs_separated=im_conts_obscured.copy()
    for f in bot_corners_1:
        (x,y,angle)=f
        cv2.circle(bot_obs_separated, (int(x), int(y)), args[18], (0, 0, 0), -1)
        cv2.circle(bot_out_separated, (int(x), int(y)), args[19], (0, 0, 0), -1)
    bot_edge_lines_1=proces_contour_to_edge( bot_out_separated  , [args[20],args[21],args[22]] )
         
    bot_edge_lines_2=proces_contour_to_edge( bot_obs_separated  , [args[23],args[24],args[25]] )
    bot_edge_lines_2=consolidate_edges(bot_edge_lines_2,np.deg2rad(args[26]),args[27])
    bot_edge_lines_1.extend(bot_edge_lines_2)
      
    bot_edge_lines_1=consolidate_edges(bot_edge_lines_1,np.deg2rad(args[28]),args[29])
    return bot_edge_lines_1
    print
def re_find_and_consolidate_bot(bot_edges,args):
    bottom_edges_asembled=bot_edges.copy()
    bot_lines_1 = cv2.HoughLinesP(bottom_edges_asembled, args[0], np.deg2rad(args[1]), args[2], minLineLength=args[3], maxLineGap=args[4])
    bot_corners_1 = find_line_intersections(bot_lines_1,np.deg2rad(args[5]),np.deg2rad(args[6]))
    bot_all_separated=bottom_edges_asembled.copy()
    for f in bot_corners_1:
        (x,y,angle)=f
        cv2.circle(bot_all_separated, (int(x), int(y)), args[7], (0, 0, 0), -1)
    bot_edge_lines_1=proces_contour_to_edge( bot_all_separated  , [args[8],args[9],args[10]] )
    bot_edge_lines_1=consolidate_edges(bot_edge_lines_1,np.deg2rad(args[11]),args[12])
    return bot_edge_lines_1
def establish_truth_on_Edges(top_edge_truth,bot_edge_truth):
    top_edge_truth=cv2.inRange(top_edge_truth,100,260)
    bot_edge_truth=cv2.inRange(bot_edge_truth,100,260)
    
    bot_seperated=bot_edge_truth.copy()
    top_seperated=top_edge_truth.copy()

    bot_lines = cv2.HoughLinesP(bot_edge_truth, 1, np.pi/360, 100, minLineLength=60, maxLineGap=2)
    top_lines= cv2.HoughLinesP(top_edge_truth,  1, np.pi/360, 100, minLineLength=60, maxLineGap=2)
    if hasattr(bot_lines, '__iter__'):
        intersect_bot=find_line_intersections(bot_lines,np.deg2rad(90),np.deg2rad(60))
        for i in intersect_bot:
            cv2.circle(bot_seperated, (int(i[0]), int(i[1])), 45  , (100, 0, 0), -1)
    
    
    #for lin in bot_lines:
    #    [bx1,by1,bx2,by2]=lin[0]
    #    ln1=[[bx2,by2],[bx1+bx2,by1+by2]]
    #    for lin2 in bot_lines:
    #        [lx1,ly1,lx2,ly2]=lin2[0]
    #        ln2=[[lx2,ly2],[lx1+lx2,ly1+ly2]]
    #        intersect,ang=find_line_intersections(lin[,ln2)
    #        if(abs(ang-np.deg2rad(90))>np.deg2rad(30)):
    #            cv2.circle(bot_seperated, (int(intersect[0]), int(intersect[1])), 20  , (100, 100, 100), -1)
    
    #cv2.imshow("edges", bot_seperated)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    if hasattr(top_lines, '__iter__'):
        intersect_top=find_line_intersections(top_lines,np.deg2rad(90),np.deg2rad(60))
        for i in intersect_top:
            cv2.circle(top_seperated, (int(i[0]), int(i[1])), 45  , (100, 0, 0), -1)
    #cv2.imshow("edges", top_seperated)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    top_seperated=cv2.inRange(top_seperated,120,260)
    bot_seperated=cv2.inRange(bot_seperated,120,260)

    edge_lines_bot=proces_contour_to_edge(bot_seperated,[10,256,0])
    edge_lines_top=proces_contour_to_edge(top_seperated,[10,256,0])
    bot_conf_line=None
    top_conf_line=None
    if hasattr(edge_lines_bot, '__iter__'):
        bot_conf_line=fit_polygon_to_graphs(edge_lines_bot,edge_lines_bot,bot_edge_truth)
    if hasattr(edge_lines_top, '__iter__'):
        top_conf_line=fit_polygon_to_graphs(edge_lines_top,edge_lines_top,top_edge_truth)

    return top_conf_line,bot_conf_line

    print
def is_empty(val):
  return not val

def conf_line_cost_func(truth_lines,gen_lines,weights):
    shape_t=len(truth_lines)#.shape()
    shape_g=len(gen_lines)#.shape()
    found_edges=np.zeros((shape_g,shape_t,2))
    unfound_cost=0
    for gen in range(shape_g):# for each generated confidence line
        gn_line=gen_lines[gen]
        p3=gn_line[0]
        p4=gn_line[1]
         
        for truth in range(shape_t): # get the cost of each line so they can be sorted later
            #if(found_edges[gen,truth,0]!=0):
            tr_line=truth_lines[truth]
            p1=tr_line[0]
            p2=tr_line[1]

            endpoint_dist_1=np.sqrt((p1[0]-p3[0])**2+(p1[1]-p3[1])**2)+np.sqrt((p2[0]-p4[0])**2+(p2[1]-p4[1])**2)
            endpoint_dist_2=np.sqrt((p1[0]-p4[0])**2+(p1[1]-p4[1])**2)+np.sqrt((p2[0]-p3[0])**2+(p2[1]-p3[1])**2)
            endpoint_dist=0
            if(endpoint_dist_1<endpoint_dist_2):
                endpoint_dist=endpoint_dist_1
            else:
                endpoint_dist=endpoint_dist_2

                

            cost=endpoint_dist
            found_edges[gen,truth,0]=cost
    g_matched=[]
    t_matched=[]
    for i in range(shape_t):
        cut=found_edges[:,i,0]# extract all generated edges cost compared to this truth edge
        if(np.any(cut)):
            id=cut.argmin() # find the generated edge with lowest cost comparatively
        #
            found_edges[id,i,1]=found_edges[id,i,0]# save the cost to the second layer
            found_edges[id,:,0]=99**20 # set a high value for the matched generated edge so it wont be matched to other edges
            if(found_edges[id,i,1]<99**19):
                g_matched.append(id)
                t_matched.append(i)
    
     
    unmatched_g = set(range(shape_g)[:]) - set(g_matched)# make a list of all generated lines which were not matched
    unmatched_t = set(range(shape_t)[:]) - set(t_matched)   
    for i in unmatched_g:# loop over them and generate the overshoot cost
        p1=gen_lines[i][0]
        p2=gen_lines[i][1]

        uc1=np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)*(gen_lines[i][2]-0.5)

        if uc1<0:
            uc1=0
        unfound_cost += uc1

    for i in unmatched_t:
        p1=truth_lines[i][0]
        p2=truth_lines[i][1]

        uc1=np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)*(truth_lines[i][2])

        if uc1<0:
            uc1=0
        unfound_cost+= uc1


    found_cost=np.sum(found_edges[:,:,1])
    if(found_cost>1.8e+15):
        found_cost=0
    if(hasattr(unfound_cost,'__iter__')):
        unfound_cost=unfound_cost[0]
    print(f"cost_found:{found_cost} cost_unfound:{unfound_cost} total: {found_cost+unfound_cost}")
    return unfound_cost+found_cost


def dot_p(a,b):
    return(a[0]*b[0]+a[1]*b[1])        

def conf_line_to_move_target(line,centroid):
    print(line)
    print(f"center at{centroid}")
    img=blured_baselayers[0]*0
    img=cv2.line(img,(int(line[0][0]),int(line[0][1])),(int(line[1][0]),int(line[1][1])),(255,255,255))
    
    centerpoint=centroid
    #centerpoint=[(np.sum(lines[:][0][0])+np.sum(lines[:][1][0]))/(len(lines)*2),(np.sum(lines[:][0][1])+np.sum(lines[:][1][1]))/(len(lines)*2)]
    #centerpoint=[(np.sum(lines[:,0,0])+np.sum(lines[:,1,0]))/(len(lines)*2),(np.sum(lines[:,0,1])+np.sum(lines[:,1,1]))/(len(lines)*2)]
    img=cv2.drawMarker(img,(int(centerpoint[0]),int(centerpoint[1])),(100,100,100))

    angle=np.arctan2(line[1][0]-line[0][0],-(line[1][1]-line[0][1]))
    print
    midpoint=[(line[0][0]+line[1][0])/2,(line[0][1]+line[1][1])/2]
    img=cv2.drawMarker(img,(int(midpoint[0]),int(midpoint[1])),(80,80,80))
    
    line_to_center=[centerpoint[0]-midpoint[0],centerpoint[1]-midpoint[1]]
    posrot=[np.cos(angle ),np.sin(angle )]
    negrot=[np.cos(angle+np.deg2rad(180) ),np.sin(angle+np.deg2rad(180) )]
    
    #img=cv2.line(img,(int(midpoint[0]),int(midpoint[1])),(int(midpoint[0]+posrot[0]*70),int(midpoint[1]+posrot[1]*70)),(255,255,255))
    #img=cv2.line(img,(int(midpoint[0]),int(midpoint[1])),(int(midpoint[0]+negrot[0]*70),int(midpoint[1]+negrot[1]*70)),(255,255,255))
    dp=dot_p(line_to_center,posrot)
    dn=dot_p(line_to_center,negrot)
    
    dir=0
    # chek if the vector from center to edge point, 
    # is pointing toward the negative 90 degree turned edge or the positive
    
    if(dp>0):
        dir=angle+np.deg2rad(180)
        img=cv2.line(img,(int(midpoint[0]),int(midpoint[1])),(int(midpoint[0]+negrot[0]*70),int(midpoint[1]+negrot[1]*70)),(255,255,255))

         
    elif(dn>0):
        dir=angle
        img=cv2.line(img,(int(midpoint[0]),int(midpoint[1])),(int(midpoint[0]+posrot[0]*70),int(midpoint[1]+posrot[1]*70)),(255,255,255))
        
    cv2.imshow("picked_edge",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return [midpoint[0],midpoint[1], dir,line[2]]





def get_image_edge_and_confidence(image):
    
     
    truths=[]
    # mean kernel size, min threshold, max threshold, canny min, canny max, dilate size, iterations
    #asumptions=[5, 254, 255, 25, 255, 3, 1]
    
    #asumptions=[3, 3, 9, 254, 255, 1, 1, 270, 5, 1, 1]
    # mean kernels size, min threshold, max threshold,clossing kn sie, canny min, canny max, dilate size, iterations
    #asumptions_2=[7, 10, 250, 5,40, 110, 3, 1,3,1,3,1,40, 110,3,1,5]
    #asumptions_2=[3, 11, 13, 1, 2, 258, 1, 175, 50, 5, 1, 15, 4, 1, 9, 330, 5, 1, 5, 1, 1, 3, 3, 1]
    # mean kernels size, min threshold, max threshold,clossing kn sie, canny min, canny max, dilate size, iterations
    #asumptions_3=[160, 7, 21, 7, 2, 7, 1, 5, 1, 5, 1, 3, 200, 1, 1, 9, 1, 2, 1]

    #asumptions_4=[5, 259, 60, 46, 1, 85, 35, 50, 10, 255, 95]

    #asumptions_5=[7, 258, 55, 36, 7, 81, 34, 170, 10, 255, 75]

    asumptions=[3, 3, 9, 254, 255, 1, 1, 270, 5, 1, 1]
    asumptions_2=[3, 11, 13, 1, 2, 258, 1, 175, 50, 5, 1, 15, 4, 1, 9, 330, 5, 1, 5, 1, 1, 3, 3, 1]
    asumptions_3=[160, 7, 21, 7, 2, 7, 1, 5, 1, 5, 1, 3, 200, 1, 1, 9, 1, 2, 1]
    asumptions_4=[2, 260, 60, 50, 1, 90, 20, 18, 10, 255, 150]
    asumptions_5=[10, 3, 330, 175, 15, 70, 24, 220, 1, 255, 210, 1, 400]
    
        #print("making truth ", i)
    img_cp_1=image.copy()
    img_cp_2=image.copy()
         
        #truths.append(establish_truths(image_contours[i])[4])
    these_truths=image.copy()*0
        #print("making truth ", i)
    im_conts,mask1 , cost               = get_image_cutouts(image, asumptions,these_truths)
    im_conts_edges,top_edges,top_mask , cost2  = get_image_cutouts_edgde_cut(image, asumptions_2,these_truths)
    im_conts_obscured   = get_obs_edg_2_ags(image,asumptions_3,top_mask,blured_baselayers[0])
    #im_conts_obscured,conts_obs,cost3   = get_obscured_edges(image,asumptions_3,these_truths,top_mask)
        
    cv2.imshow("input_image", image)
    cv2.imshow("top_visible_edges", im_conts_edges)
    cv2.imshow("top_top_textile_edges", top_edges)
    cv2.imshow("bottom_vis_edges", im_conts-top_edges)
    cv2.imshow("bottom_textile_edges", im_conts_obscured)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    top_lines=find_and_consolidate_top(top_edges,asumptions_4)
    
    bot_edges=(im_conts-cv2.dilate(top_edges, np.ones((5,5), np.uint8), iterations=3))+im_conts_obscured
    bot_edges=cv2.inRange(bot_edges,20,255)
    bot_edges=bot_edges.astype(np.uint8)
    bot_lines=re_find_and_consolidate_bot(bot_edges,asumptions_5)

    bot_edges_conf_nar=fit_polygon_to_graphs(bot_lines,bot_lines,bot_edges)
    top_edges_conf_reg=fit_polygon_to_graphs(top_lines,top_lines,top_edges)
    all_edges=[]
    if hasattr(top_lines, '__iter__'):
        if(len(top_lines)>2):
            all_edges=top_lines
    if hasattr(bot_lines, '__iter__'):
        if(len(bot_lines)>1):
            all_edges.extend(bot_lines) 
    bot_edges_conf_wide=fit_polygon_to_graphs(bot_lines,all_edges,(bot_edges+top_edges).astype(np.uint8))
    
    if hasattr(bot_edges_conf_nar, '__iter__'):
        img_cp_1= draw_confidence(img_cp_1, top_edges_conf_reg,bot_edges_conf_nar)
    
    #if hasattr(top_edges_conf_reg, '__iter__'):    
    #    img_cp_1= draw_confidence(img_cp_1, top_edges_conf_reg,(0,255,0))
     
        #img_cp_2= draw_confidence(img_cp_2, top_edges_conf_reg,(0,255,0))
    if hasattr(bot_edges_conf_wide, '__iter__'): 
        img_cp_2= draw_confidence(img_cp_2, top_edges_conf_reg,bot_edges_conf_wide)
    
    cv2.imshow("edges_strict", img_cp_1)
    cv2.imshow("edges_lose", img_cp_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    c_bot=get_centeroid_img(bot_edges)
    c_top=get_centeroid_img(top_edges)

    return bot_edges_conf_nar,top_edges_conf_reg,bot_edges_conf_wide,c_bot,c_top

def get_1_layer_edge_and_confidence(image):
    print

def generate_edge_movement(image):
    bot_conf_e,top_conf_e,bot_conf_wide,c_bot,c_top=get_image_edge_and_confidence(image)
    # find the pick up location
    top_conf_thresh=0.7
    p_e_top=[]
    for edg_lc in range(len(top_conf_e)):
        if (top_conf_e[edg_lc][2]>top_conf_thresh):
            p_e_top.append(top_conf_e[edg_lc])
    p_e_bot=[]
    bot_conf_thresh=0.7
    for edg_lc in range(len(bot_conf_e)):
        if (bot_conf_e[edg_lc][2]>bot_conf_thresh):
            p_e_bot.append(bot_conf_e[edg_lc])
    p_e_bot_wide=[]
    bot_conf_thresh_wide=0.6
    for edg_lc in range(len(bot_conf_wide)):
        if (bot_conf_wide[edg_lc][2]>bot_conf_thresh_wide):
            p_e_bot_wide.append(bot_conf_wide[edg_lc])

    print(f"{len(top_conf_e)}")
    if(len(p_e_bot)>0 and len(p_e_top)>0):
        # find the two closest posible bottom and top edges
        shortests=100000
        weigts_=[1,0.2]
        start_=[]
        end_=[]
        for lt in p_e_top:
            mp_top=[(lt[0][0]+lt[1][0])/2,(lt[0][1]-lt[1][1])/2]
            ang_top=np.arctan2(lt[1][0]-lt[0][0],lt[1][1]-lt[0][1])
            for lb in p_e_bot:
                mp_bot=[(lb[0][0]+lb[1][0])/2,(lb[0][1]-lb[1][1])/2]
                ang_bot=np.arctan2(lb[1][0]-lb[0][0],lb[1][1]-lb[0][1])
                dist=np.sqrt((mp_top[0]-mp_bot[0])**2,(mp_bot[0]-mp_bot[0])**2)
                dist=dist*weigts_[0]+abs(ang_top-ang_bot)*weigts_[1]
                if(dist<shortests):
                    shortests=dist
                    start_=lt
                    end_=lb
        # send startpoint endpoint
        pickup=conf_line_to_move_target(start_,c_top)
        placement=conf_line_to_move_target(end_,c_bot)
        publish_p(pickup,placement)

        print
    elif(len(p_e_bot_wide)>0 and len(p_e_top)>0):
         # find the two closest posible bottom and top edges
        shortests=100000
        weigts_=[1,0.2]
        start_=[]
        end_=[]
        for lt in p_e_top:
            mp_top=[(lt[0][0]+lt[1][0])/2,(lt[0][1]-lt[1][1])/2]
            ang_top=np.arctan2(lt[1][0]-lt[0][0],lt[1][1]-lt[0][1])
            for lb in p_e_bot_wide:
                mp_bot=[(lb[0][0]+lb[1][0])/2,(lb[0][1]-lb[1][1])/2]
                ang_bot=np.arctan2(lb[1][0]-lb[0][0],lb[1][1]-lb[0][1])
                dist=np.sqrt((mp_top[0]-mp_bot[0])**2,(mp_bot[0]-mp_bot[0])**2)
                dist=dist*weigts_[0]+abs(ang_top-ang_bot)*weigts_[1]
                if(dist<shortests):
                    shortests=dist
                    start_=lt
                    end_=lb
        # send startpoint endpoint
        pickup=conf_line_to_move_target(start_,c_top)
        placement=conf_line_to_move_target(end_,c_bot)
        publish_p(pickup,placement)
         
    else:
        # no solutions could be found above the thrshold
        print("no solutions to path planning")

rob_msg_count=0   
def publish_p(pickup,placement):
    global rob_msg_count
    print("sent a pose")
    F_pub.pose = Float32MultiArray()
    print(f"pick up {pickup}")
    print(f"placement {placement}")
    F_pub.pose.data = [float(pickup[0]),float(pickup[1]),float(pickup[2]),float(pickup[3]),float(placement[0]),float(placement[1]) ,float(placement[2]) ,float(placement[3]),float(rob_msg_count)]
    F_pub.publish_pose()
#c_sub=MinimalSubscriber(Node)
#F_pub=FabricPositionPublisher(Node)

 


if __name__ == '__main__':
    main()


#if __name__ == "__main__":
    #print("Current working directory:", os.getcwd())
    #optimize_cuts(images,0)
    #optimize_lines(images,0)
    #generate_edge_movement(images[0])
    #get_image_edge_and_confidence(images[1])
    #main()
  

             
     
        
 
    

