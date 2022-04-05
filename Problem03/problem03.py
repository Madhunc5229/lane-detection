from turtle import right
import numpy as np
import cv2
import copy


def warpImage(image, src_p, dst_p, img_size):
    M = cv2.getPerspectiveTransform(src_p, dst_p)
    warpImage = cv2.warpPerspective(image, M, img_size)

    return warpImage

def get_hist(img):
    hist = np.sum(img[:,:], axis=0)
    return hist

def slidWin(warped_img):

    image = copy.deepcopy(warped_img)

    #create empty lists to append curve fit parameters
    left_a = []
    left_b = []
    left_c = []

    right_a = []
    right_b = []
    right_c = []
    
    #Get histogram of the frame column wise
    hist = get_hist(image)

    #Get the centre point of the histogram
    hist_ctr = int(hist.shape[0]/2)
    
    #Get the column index of maximum histogram 
    left_hist = np.argmax(hist[:100])
    right_hist = np.argmax(hist[70:]) + hist_ctr

    #Define the window height
    win_h = int(image.shape[0]/24)

    #Initialize the center point of the first window 
    left_current = left_hist
    right_current = right_hist

    #Create lists to append best 
    left_lane_indx = []
    right_lane_indx = []

    #Get the index of all non zero pixels
    nonzerox = np.array(image.nonzero()[1])
    nonzeroy = np.array(image.nonzero()[0])

    for win in range(25):

        #Defining window in y direction
        y_low = image.shape[0] - ((win+1)*win_h)
        y_high = image.shape[0] - (win*win_h)

        #Defining left window in x direction
        x_left_low = left_current - 15
        x_left_high = left_current + 15

        #Defining right window in x direction
        x_right_low = right_current - 15
        x_right_high = right_current + 15

        #plot the windows
        cv2.rectangle(image, (x_left_low,y_low),(x_left_high, y_high), (255), 
        1)
        cv2.rectangle(image, (x_right_low,y_low),(x_right_high, y_high), (255), 
        1)

        #Defining good windows based on number of non points in the window
        good_left_win_indx = ((nonzerox < x_left_high) & (nonzerox >= x_left_low) & (nonzeroy < y_high) & (nonzeroy >= y_low)).nonzero()[0]
        good_right_win_indx =((nonzerox < x_right_high) & (nonzerox >= x_right_low) & (nonzeroy < y_high) & (nonzeroy >= y_low)).nonzero()[0]

        #Append all the good window index
        left_lane_indx.append(good_left_win_indx)
        right_lane_indx.append(good_right_win_indx)

        #If one pixel with desired intensity value is found, 
        #update the starting point of next window by calculating the mean
        if len(good_left_win_indx)>1:
            left_current = (np.mean(nonzerox[good_left_win_indx])).astype(int)
        if len(good_right_win_indx)>1:
            right_current = (np.mean(nonzerox[good_right_win_indx])).astype(int)

    #Concatenate the indicies
    left_lane_indx = np.concatenate(left_lane_indx)
    right_lane_indx = np.concatenate(right_lane_indx)

    #Separate the x and y from the index which are non zero
    leftx = nonzerox[left_lane_indx]
    lefty = nonzeroy[left_lane_indx] 
    rightx = nonzerox[right_lane_indx]
    righty = nonzeroy[right_lane_indx]


    if len(rightx) != 0:
        #Fit a parabola for left and right curve
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)


        left_a.append(left_fit[0])
        left_b.append(left_fit[1])
        left_c.append(left_fit[2])

        right_a.append(right_fit[0])
        right_b.append(right_fit[1])
        right_c.append(right_fit[2])

        #get the average of the parameters
        left_fit[0] = np.mean(left_a[:])
        left_fit[1] = np.mean(left_b[:])
        left_fit[2] = np.mean(left_c[:])
        
        right_fit[0] = np.mean(right_a[:])
        right_fit[1] = np.mean(right_b[:])
        right_fit[2] = np.mean(right_c[:])

        #create points from 0 to 399 to plot the curve
        ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )

        #generate x values from y from eq ax^2 + bx + c
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        return image, (left_fitx, right_fitx), (left_fit, right_fit), ploty
    else:
        return None


def get_curve(img, leftx, rightx):

    #create points from 0 to 399 to plot the curve
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])

    #max of y i.e 399
    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty, leftx, 2)
    right_fit_cr = np.polyfit(ploty, rightx, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    #car position
    car_pos = img.shape[1]/2

    #calculating the lane center
    l_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0]*img.shape[0]**2 + right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) /2
    center = (car_pos - lane_center_position)  / 10

    return (left_curverad, right_curverad, center)



def draw_lanes(img, left_fit, right_fit):

    #create points from 0 to 399 to plot the curve
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])

    #plot the right and left curve
    for i in range(len(left_fit)):

        cv2.circle(img,(int(left_fit[i]), int(ploty[i])),0,(0,255,0),5)

    for i in range(len(right_fit)):

        cv2.circle(img,(int(right_fit[i]), int(ploty[i])),0,(0,0,255),5)


    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))
    
    #plot the lane
    cv2.fillPoly(img, np.int_(points), (0,200,255))

    dst = np.float32([[150,680],[630,425],[730,425],[1180,680]])

    src = np.float32([[0,400],[0,0],[200,0],[200,400]])

    img_size = (1280,720)

    #warp the image back to frame size
    frame_back = warpImage(img,src,dst,img_size)

    return frame_back

def generateFinalimg(orig_img, warp_img, win_img, overlayed_img, curve_rad):

    cv2.putText(orig_img,'(1)', (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 10)
                
    cv2.putText(warp_img,'(2)', (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv2.putText(win_img,'(3)', (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    if curve_rad[0] >0 and curve_rad[1] >0:
        cv2.putText(overlayed_img,'Turn Right', (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    warped_imgs = np.concatenate((warp_img, win_img), axis=1)

    merged = cv2.merge((warped_imgs,warped_imgs,warped_imgs))

    original_frame = cv2.resize(orig_img,(400,320),interpolation = cv2.INTER_AREA)

    right_stack = np.concatenate((original_frame,merged),axis=0)

    image_stack = np.concatenate((overlayed,right_stack), axis=1)

    text_img = np.empty((150,1680,3),dtype= np.uint8)

    text_img[:,:]= (200,124,124)

    text = '(1) : Undistorted Image  (2) : Warped and filtered white and yellow lane'+' '+ '(3) : Sliding windows detecting the curve'
    curvature = 'Left Curvature = '+str(curve_rad[0]) + '  '+ 'Right Curvature = ' + str(curve_rad[1])

    cv2.putText(text_img,text, (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 3)
    
    cv2.putText(text_img,curvature, (10, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 3)


    final_image = np.concatenate((image_stack,text_img),axis=0)

    return final_image



if __name__=='__main__':

    cap = cv2.VideoCapture('predict_turn.mp4')

    ret = True

    fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
    out = cv2.VideoWriter("problem03.mp4", fourcc, 25, (1680, 870))

    while(cap.isOpened()):

        ret, frame = cap.read()
        
        if ret == True:
            
            #create points for warping
            pts1 = np.float32([[150,680],[630,425],[730,425],[1180,680]])

            pts2 = np.float32([[0,400],[0,0],[200,0],[200,400]])

            img_size = (200,400)

            frame_warp = warpImage(frame, pts1, pts2,img_size)
            
            #filter the yellow and white lanes
            frame_hsv = cv2.cvtColor(frame_warp, cv2.COLOR_BGR2HSV)

            mask_yellow = cv2.inRange(frame_hsv, (12, 124, 125), (33, 255, 255))

            frame_gray = cv2.cvtColor(frame_warp, cv2.COLOR_BGR2GRAY)

            ret,frame_bin = cv2.threshold(frame_gray,200,255,cv2.THRESH_BINARY)

            masked = cv2.bitwise_or(frame_bin, mask_yellow)

            #create sliding windows to fit a curve
            fitting = slidWin(masked)

            #if successfully fit a curve
            if fitting is not None:

                img_win, curves, lanes, ploty = fitting

                curve_radii = get_curve(img_win, curves[0],curves[1])

                lane_detect = draw_lanes(frame_warp, curves[0], curves[1])
                
                overlayed = cv2.addWeighted(frame,1,lane_detect,0.5,0)

                #generate the final image showing all images
                final_img = generateFinalimg(frame,masked,img_win,overlayed, curve_radii)

                cv2.imshow("overlayed",final_img)

                out.write(final_img)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
