import numpy as np
import cv2
import statistics as st

def solidOrDash(image):
    left_half = image[:,0:int(image.shape[1]/2)]
    right_half = image[:,int(image.shape[1]/2):]
    if (np.count_nonzero(left_half) < np.count_nonzero(right_half)):
        left_line = 'Dashed'
        right_line = 'Solid'
    else:
        left_line = 'Solid'
        right_line = 'Dashed'
    return right_line, left_line

def roI(image):
    mask = np.zeros_like(image)
    triangle = np.array([[10,image.shape[0]],[485,310],[image.shape[1],image.shape[0]]])
    mask = cv2.fillPoly(mask, np.int32([triangle]) ,255)

    return mask

def getPoints(right_line, left_line):
    r_m, r_c = right_line
    l_m, l_c = left_line

    if (r_m and l_m) !=0: 
        r_y1 = 539
        r_x1 = int((r_y1-r_c)/r_m)
        r_y2 = 320
        r_x2 =  int((r_y2-r_c)/r_m)

        l_y1 = 539
        l_x1 = int((l_y1-l_c)/l_m)
        l_y2 = 320
        l_x2 =  int((l_y2-l_c)/l_m)

        r_points = [r_x1,r_y1,r_x2, r_y2]
        l_points = [l_x1,l_y1,l_x2, l_y2]
    else:
        r_points = [0,0,0,0]
        l_points = [0,0,0,0]        

    return r_points, l_points



def lineCat(lines):

    right_slope = []
    left_slope = []
    right_intercept = []
    left_intercept = []

    for points in lines:
        x1,y1,x2,y2 = points[0]

        m,c = np.polyfit((x1,x2),(y1,y2),1)

        if m > 0:
            right_slope.append(m)
            right_intercept.append(c)
        else:
            left_slope.append(m)
            left_intercept.append(c)
    if len(right_slope) and len(left_slope) != 0:
        right_line = [st.mean(right_slope) , st.mean(right_intercept)]
        left_line = [st.mean(left_slope) , st.mean(left_intercept)]
    else:
        right_line = [0,0]
        left_line = [0,0]

    r_points, l_points = getPoints(right_line,left_line)

    return r_points, l_points


if __name__=='__main__':

    cap = cv2.VideoCapture('whiteline.mp4')

    ret = True

    fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
    out = cv2.VideoWriter("lane_detect_p2.mp4", fourcc, 25, (960, 540))

    while(cap.isOpened()):

        ret, frame = cap.read()

        
        if ret == True:

            # frame = cv2.flip(frame,1)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            ret,frame_bin = cv2.threshold(frame_gray,210,255,cv2.THRESH_BINARY)

            masked = cv2.bitwise_and(frame_bin, roI(frame_bin))

            blurred = cv2.GaussianBlur(masked, (5,5), 0)

            edges = cv2.Canny(blurred, 50, 150)

            right_side,left_side = solidOrDash(edges)
            
            lines = cv2.HoughLinesP(edges,rho = 4,theta = 1*np.pi/180,threshold = 100,minLineLength = 40,maxLineGap = 150)

            r_line, l_line = lineCat(lines)

            if right_side == 'Solid':
                #right is green and left is red
                cv2.line(frame,(r_line[0],r_line[1]),(r_line[2],r_line[3]), (0,255,0),5)
                cv2.line(frame,(l_line[0],l_line[1]),(l_line[2],l_line[3]), (0,0,255),5)
            else:
                #right is red and left is green
                cv2.line(frame,(r_line[0],r_line[1]),(r_line[2],r_line[3]), (0,0,255),5)
                cv2.line(frame,(l_line[0],l_line[1]),(l_line[2],l_line[3]), (0,255,0),5)

            cv2.imshow("original",frame)
            out.write(frame)


        if cv2.waitKey(45) & 0xFF == ord('q'):
            break

    out.release()
