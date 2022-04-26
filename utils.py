import cv2 as cv
import numpy as np



def order(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    # print(np.argmax(s))
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # print(np.argmax(diff))
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def Homograph(p1, p2):
    A = []
    p1 = order(p1)
    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        m, n = p2[i][0], p2[i][1]
        A.append([x, y, 1, 0, 0, 0, -m * x, -m * y, -m])
        A.append([0, 0, 0, x, y, 1, -n * x, -n * y, -n])
    A = np.array(A)
    u, s, v= np.linalg.svd(A)
    H_matrix = v[-1, :]
    H_matrix = H_matrix.reshape(3,3)
    H_matrix_normalized = H_matrix/H_matrix[2,2]
    
    return H_matrix_normalized

def decode_tag(ref_tag_image):
    
    size = 160
    tag_gray = cv.cvtColor(ref_tag_image, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(tag_gray, 230 ,255,cv.THRESH_BINARY)[1]
    thresh_resize = cv.resize(thresh, (size, size))
    grid_size = 8
    stride = int(size/grid_size)
    grid = np.zeros((8,8))
    x = 0
    y = 0
    for i in range(0, grid_size, 1):
        for j in range(0, grid_size, 1):
            cell = thresh_resize[y:y+stride, x:x+stride]
            if cell.mean() > 255//2:
                grid[i][j] = 255
            x = x + stride
        x = 0
        y = y + stride
    inner_grid = grid[2:6, 2:6]


    i = 0
    while not inner_grid[3,3] and i<4 :
        inner_grid = np.rot90(inner_grid,1)
        i = i + 1

    
    info_grid = inner_grid[1:3,1:3]
    info_grid_array = np.array((info_grid[0,0],info_grid[0,1],info_grid[1,1],info_grid[1,0]))
    tag_id = 0
    tag_id_bin = []
    for j in range(0,4):
        if(info_grid_array[j]) :
            tag_id = tag_id + 2**(j)
            tag_id_bin.append(1)
        else:
            tag_id_bin.append(0)

        rect = np.zeros((4, 2), dtype="float32")
        # print(type(inner_grid))

    return inner_grid, tag_id, tag_id_bin,i

def projectionMatrix(h,K):  
    h1 = h[:,0]   
    h2 = h[:,1]
    h3 = h[:,2]

    #Lamda function
    lamda = 2 / (np.linalg.norm(np.matmul(np.linalg.inv(K),h1)) + np.linalg.norm(np.matmul(np.linalg.inv(K),h2)))
    b_t = lamda * np.matmul(np.linalg.inv(K),h)

    #check if determinant is greater than 0 ie. has a positive determinant when object is in front of camera
    det = np.linalg.det(b_t)

    if det > 0:
        b = b_t
    else:              
        b = -1 * b_t  
        
    row1 = b[:, 0]
    row2 = b[:, 1]                      #extract rotation and translation vectors
    row3 = np.cross(row1, row2)
    
    t = b[:, 2]
    Rt = np.column_stack((row1, row2, row3, t))

    P = np.matmul(K,Rt)  
    return P


### Intrinsic Camera Parameters



def warpPerspective(img, H, size):
    h, w = img.shape[:2]
    if len(img.shape) == 3:
        result = np.zeros([size[1], size[0], 3], np.uint8)
    else:
        result = np.zeros([size[1], size[0]], np.uint8)
    x, y = np.indices((w, h))
    x,y = x.flatten(),y.flatten()
    
    img_coords = np.vstack((x,y,[1]*x.size))
    
    new_coords = H @ img_coords
    
    new_coords = new_coords/(new_coords[2]) #+ 1e-6)
    new_x, new_y, _ = np.int0(np.round(new_coords))
    
    new_x[np.where(new_x < 0)] = 0
    new_y[np.where(new_y < 0)] = 0
    new_x[np.where(new_x > size[0] - 1)] = size[0] - 1
    new_y[np.where(new_y > size[1] - 1)] = size[1] - 1
    
    result[new_y,new_x] = img[y,x]
    
    return result
    