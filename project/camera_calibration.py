import cv2 
import numpy as np
import glob


def calibrate_camera(w,h,criteria,imgs_path,show_imgs=False):

    # Vector que vai guardar pontos 3D em espaço mundo real para cada imagem de xadrez
    objpoints = [] 
    # Vector que vai guardar pontos 2D em espaço imagem para cada imagem de xadrez
    imgpoints = [] 

    #definir coordenadas do mundo para pontos 3D
    objp = np.zeros((w*h,3), np.float32)
    objp[:,:2] = np.mgrid[0:h, 0:w].T.reshape(-1, 2)

    images = glob.glob(imgs_path)
    for img_name in images:
        img = cv2.imread(img_name)
        #converter para grayscale
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #Encontrar cantos do xadrez
        ret, corners = cv2.findChessboardCorners(gray_img, (w,h), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        #ret é igual a true se o numero de cantos encontrados for o desejado
        #apos detecao dos cantos, fazer a melhoria dos cantos
        if ret==True:
            objpoints.append(objp)
            #Redefinir os cantos do xadrez
            corners_refined = cv2.cornerSubPix(gray_img, corners, (11,11),(-1,-1), criteria)
            #adicionar corners
            imgpoints.append(corners_refined)
            #desenhar os cantos no xadrez
            img = cv2.drawChessboardCorners(img, (w,h), corners_refined, ret)
            #cv2.imwrite()
        #if show_imgs:
           #  cv2.imshow('img',img)
            # cv2.waitKey(0)
   # cv2.destroyAllWindows()

    
    return objpoints,imgpoints,gray_img
    

def do_undistortion(img, mtx,dist):
    #read image
    #img = cv2.imread(img_path)
    h,  w = img.shape[:2]
    # refine the camera matrix based on a free scaling parameter
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    undistort_img = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    #undistort_img = undistort_img[y:y+h, x:x+w]
    #cv2.imwrite('calibresult.png', undistort_img)
    return undistort_img



def get_re_projection_error(objpoints,imgpoints,rvecs,tvecs,mtx,dist):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    print( "total error: {}".format(mean_error/len(objpoints)) )




'''
#termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints,imgpoints,gray_img = calibrate_camera(7,7,criteria,'./imgs/calibration_3/*.jpg',show_imgs=True)
#realizar camera calibration passando os valores do objpoints (pontos 3D)
#e as coordenadas de pixeis correspondentes dos cantos detetatos
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_img.shape[::-1], None, None)

#img_2_path = './imgs/WIN_20220530_14_19_27_Pro.jpg'
#img_undist = do_undistortion(img_2_path, mtx,dist)

get_re_projection_error(objpoints,imgpoints,rvecs,tvecs,mtx,dist)

'''